import json

import httpx
import pytest
from test_forge import MODEL, client, prepared

from bufogen.integrations.forge import ForgeError
from bufogen.workflows.forge_loop import AmbiguousOperation, resume_training, submit_training


def imported_state(tmp_path, api):
    path, _ = prepared(tmp_path, api)
    state = json.loads(path.read_text())
    state.update(dataset_id="dataset-1", entries_imported=True)
    path.write_text(json.dumps(state))
    return path


@pytest.mark.parametrize("rate", [None, 2e-5, 1e-4])
def test_training_request_and_durable_intent_preserve_selected_rate(tmp_path, rate):
    requests = []
    expected = {"epochs": 1, "batch_size": 2, "assistant_turns": "all"}
    if rate is not None:
        expected["learning_rate"] = rate

    def handler(request):
        requests.append(request)
        saved = json.loads(path.read_text())
        assert saved["training_config"] == expected
        assert saved["pending_operation"]["name"] == "create_finetune"
        assert json.loads(request.content)["model_config"] == {"config": expected}
        return httpx.Response(202, json={"id": "job-1", "status": "queued"})

    with client(handler) as api:
        path = imported_state(tmp_path, api)
        result = submit_training(path, api, learning_rate=rate)
    assert result["finetune_id"] == "job-1"
    assert result["training_config"] == expected
    assert result["pending_operation"] is None
    assert len(requests) == 1


@pytest.mark.parametrize("rate", [True, False, 0, -0.1, float("nan"), float("inf"), "0.0001", []])
@pytest.mark.parametrize("entrypoint", ["client", "workflow"])
def test_invalid_learning_rate_fails_before_network_or_state_change(tmp_path, rate, entrypoint):
    requests = []
    with client(lambda request: requests.append(request)) as api:
        path = imported_state(tmp_path, api)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="learning_rate"):
            if entrypoint == "client":
                api.create_finetune(
                    dataset_id="dataset-1", model_name="bufo-sft", learning_rate=rate
                )
            else:
                submit_training(path, api, learning_rate=rate)
    assert not requests
    assert path.read_bytes() == before


def test_uncertain_training_write_keeps_rate_and_cannot_be_resubmitted(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("interrupted", request=request)

    with client(handler) as api:
        path = imported_state(tmp_path, api)
        with pytest.raises(ForgeError):
            submit_training(path, api, learning_rate=2e-5)
        saved = json.loads(path.read_text())
        assert saved["training_config"]["learning_rate"] == 2e-5
        assert saved["pending_operation"]["name"] == "create_finetune"
        for rate in [2e-5, 1e-4, None]:
            with pytest.raises(AmbiguousOperation):
                submit_training(path, api, learning_rate=rate)
    assert len(requests) == 1


@pytest.mark.parametrize("stored_rate", [None, 2e-5])
def test_existing_job_cannot_be_presented_as_a_different_rate(tmp_path, stored_rate):
    requests = []
    with client(lambda request: requests.append(request)) as api:
        path = imported_state(tmp_path, api)
        state = json.loads(path.read_text())
        state.update(
            finetune_id="job-1",
            training_config={"epochs": 1, "batch_size": 2, "assistant_turns": "all"},
        )
        if stored_rate is not None:
            state["training_config"]["learning_rate"] = stored_rate
        path.write_text(json.dumps(state))
        before = path.read_bytes()
        with pytest.raises(ValueError, match="configuration"):
            submit_training(path, api, learning_rate=1e-4)
    assert not requests
    assert path.read_bytes() == before


def test_same_rate_submission_and_legacy_poll_never_recreate_job(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "id": "job-1",
                "status": "deployed",
                "base_model": MODEL,
                "model_artifact_ref": "wandb-artifact:///team/p/real:v1",
            },
        )

    with client(handler) as api:
        path = imported_state(tmp_path, api)
        state = json.loads(path.read_text())
        state.update(
            finetune_id="job-1",
            training_config={
                "epochs": 1,
                "batch_size": 2,
                "assistant_turns": "all",
                "learning_rate": 2e-5,
            },
        )
        path.write_text(json.dumps(state))
        submit_training(path, api, learning_rate=2e-5)
        submit_training(path, api)
        result = resume_training(path, api)
    assert len(requests) == 3
    assert result["training_config"]["learning_rate"] == 2e-5
    assert result["model_artifact_ref"] == "wandb-artifact:///team/p/real:v1"
