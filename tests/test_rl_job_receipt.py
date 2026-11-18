import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from art.serverless.client import Client, TrainingJob
from openai import APITimeoutError

from bufogen.training.rl import _atomic_record, capture_training_job_receipt


def checkpoint_writer(path):
    def checkpoint(**changes):
        prior = json.loads(path.read_text()) if path.exists() else {}
        prior.update(changes)
        _atomic_record(path, prior)

    return checkpoint


async def test_real_sdk_create_id_is_persisted_before_event_poll_and_method_restored(
    tmp_path, monkeypatch
):
    async with Client(api_key="test", base_url="https://training.test/v1") as client:
        resource = client.training_jobs
        original = resource.create
        path = tmp_path / "state.json"
        received = TrainingJob(id="actual-server-job-123")

        async def post(*args, **kwargs):
            assert args == ("/preview/training-jobs",)
            assert kwargs["body"]["model_id"] == "model-123"
            # The durable intent exists before the actual SDK request executes.
            assert json.loads(path.read_text())["training_job"] == {
                "model_id": "model-123",
                "from_step": 1,
                "status": "submitting",
                "id": None,
            }
            return received

        mocked_post = AsyncMock(side_effect=post)
        monkeypatch.setattr(resource, "_post", mocked_post)
        with capture_training_job_receipt(
            SimpleNamespace(_client=client), checkpoint_writer(path), model_id="model-123", step=1
        ):
            result = await resource.create(model_id="model-123", trajectory_groups=[])
            assert result is received
            assert json.loads(path.read_text())["training_job"]["id"] == "actual-server-job-123"
            # The SDK would next enter polling; no poll/optimizer call is retried here.
            with pytest.raises(RuntimeError, match="second"):
                await resource.create(model_id="model-123", trajectory_groups=[])
        assert mocked_post.await_count == 1
        assert resource.create == original and "create" not in vars(resource)


async def test_timeout_retains_intent_without_invented_id_or_retry(tmp_path):
    error = APITimeoutError(request=httpx.Request("POST", "https://training.test"))
    original = AsyncMock(side_effect=error)
    resource = SimpleNamespace(create=original)
    path = tmp_path / "state.json"
    with pytest.raises(APITimeoutError):
        with capture_training_job_receipt(
            SimpleNamespace(_client=SimpleNamespace(training_jobs=resource)),
            checkpoint_writer(path),
            model_id="model",
            step=1,
        ):
            await resource.create(model_id="model", trajectory_groups=[])
    receipt = json.loads(path.read_text())["training_job"]
    assert receipt == {
        "model_id": "model",
        "from_step": 1,
        "status": "create_error",
        "id": None,
        "error_type": "APITimeoutError",
    }
    original.assert_awaited_once()
    assert resource.create is original


async def test_polling_failure_keeps_successful_receipt_and_restores_hook(tmp_path):
    original = AsyncMock(return_value=TrainingJob(id="returned-id"))
    resource = SimpleNamespace(create=original)
    path = tmp_path / "state.json"
    with pytest.raises(TimeoutError):
        with capture_training_job_receipt(
            SimpleNamespace(_client=SimpleNamespace(training_jobs=resource)),
            checkpoint_writer(path),
            model_id="model",
            step=3,
        ):
            await resource.create(model_id="model", trajectory_groups=[])
            raise TimeoutError("event polling interrupted")
    assert json.loads(path.read_text())["training_job"] == {
        "model_id": "model",
        "from_step": 3,
        "status": "created",
        "id": "returned-id",
    }
    assert resource.create is original and original.await_count == 1
