import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import art
import httpx
import pytest
from openai import APIStatusError, APITimeoutError, AsyncOpenAI

from bufogen.training import rl as train_rl

BASE = "google/gemma-4-26B-A4B-it"


def completion(number):
    return {
        "id": f"real-http-choice-{number}",
        "created": 0,
        "model": BASE,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": f"Actual response {number}"},
                "prompt_token_ids": [1, 2],
                "token_ids": [3, 4],
                "art_completion_logprobs": [-0.2, -0.3],
            }
        ],
        "nvext": {
            "engine_data": {
                "finished": True,
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "completion_logprobs": [-0.2, -0.3],
            }
        },
    }


async def test_retry_preserves_partial_actual_capture_and_retries_only_missing_rollout(
    tmp_path, monkeypatch
):
    from test_rl_contract import scored_row

    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 2:
            raise httpx.ReadTimeout("transient", request=request)
        return httpx.Response(200, json=completion(len(calls)))

    sleep = AsyncMock()
    monkeypatch.setattr(train_rl.asyncio, "sleep", sleep)
    groups = []
    client = AsyncOpenAI(
        api_key="test",
        max_retries=0,
        base_url="https://api.inference.wandb.ai/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def generate(prompt, **kwargs):
        groups.append(kwargs["group_id"])
        for _ in range(2):
            await client.chat.completions.create(
                model=BASE, messages=[{"role": "user", "content": prompt}]
            )
        row, _, _ = scored_row()
        return row

    async with client:
        row, trajectory, path, record = await train_rl.student_rollout(
            art=art,
            agent_factory=lambda: SimpleNamespace(generate=generate, partial_generation=None),
            example={"prompt_id": "p", "prompt": "Draw Bufo"},
            output=tmp_path,
            name="attempt",
            step=1,
            rollout_index=3,
        )
    attempts = sorted((tmp_path / "captures").glob("*.json"))
    assert len(attempts) == 2 and len(calls) == 4
    failed = json.loads(attempts[0].read_text())
    assert failed["status"] == "failed" and failed["error_type"] == "APITimeoutError"
    assert failed["retry_scheduled"] is True
    assert failed["exchanges"]["chat_completions"][0]["response"]["id"] == "real-http-choice-1"
    assert len(failed["exchanges"]["chat_completions"]) == 1
    saved = json.loads(path.read_text())
    assert saved["status"] == "captured" and saved["generation_id"] == row.id
    assert [c["response"]["id"] for c in saved["exchanges"]["chat_completions"]] == [
        "real-http-choice-3",
        "real-http-choice-4",
    ]
    assert len(trajectory.exchanges.chat_completions) == 2
    assert groups == ["attempt:1:3:attempt1", "attempt:1:3:attempt2"]
    sleep.assert_awaited_once_with(2)
    assert record["step"] == 1 and record["rollout_index"] == 3


@pytest.mark.parametrize("status,retries", [(401, 1), (400, 1), (429, 3), (503, 3)])
async def test_only_transient_inference_errors_retry_with_finite_budget(
    tmp_path, monkeypatch, status, retries
):
    error = APIStatusError(
        "no secrets stored",
        response=httpx.Response(status, request=httpx.Request("POST", "https://example.test")),
        body=None,
    )
    generate = AsyncMock(side_effect=error)
    monkeypatch.setattr(train_rl.asyncio, "sleep", AsyncMock())
    with pytest.raises(APIStatusError):
        await train_rl.student_rollout(
            art=art,
            agent_factory=lambda: SimpleNamespace(generate=generate),
            example={"prompt_id": "p", "prompt": "Draw Bufo"},
            output=tmp_path,
            name="attempt",
            step=1,
            rollout_index=0,
        )
    assert generate.await_count == retries
    records = [json.loads(p.read_text()) for p in sorted((tmp_path / "captures").glob("*.json"))]
    assert len(records) == retries
    assert records[-1]["retry_scheduled"] is False
    assert all(r["exchanges"]["chat_completions"] == [] for r in records)


@pytest.mark.parametrize("missing_metadata", [False, True])
async def test_training_failure_is_never_retried_and_existing_attempt_refuses_replay(
    tmp_path, monkeypatch, missing_metadata
):
    import art.serverless
    from test_rl_contract import scored_row

    import wandb
    from bufogen.evaluation.scorers import scene_quality
    from bufogen.integrations import telemetry
    from bufogen.training.rl_contract import build_calibration_report, write_environment_tasks

    positive, yes, config = scored_row()
    negative, no, _ = scored_row(False)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            build_calibration_report(
                [positive.to_dict(), negative.to_dict()], [yes, no], scorer_configuration=config
            )
        )
    )
    write_environment_tasks(
        [
            {
                "id": "p",
                "prompt": positive.prompt,
                "family": "presenter",
                "split": "train",
                "requirements": positive.benchmark_metadata["requirements"],
            }
        ],
        tmp_path,
    )
    model = SimpleNamespace(
        register=AsyncMock(),
        get_step=AsyncMock(return_value=0),
        log=AsyncMock(),
        run_id="actual-run",
        id="actual-model",
    )
    training_error = APITimeoutError(request=httpx.Request("POST", "https://training.test"))
    backend = SimpleNamespace(
        _client=SimpleNamespace(training_jobs=SimpleNamespace(create=AsyncMock())),
        close=AsyncMock(),
        train=AsyncMock(side_effect=training_error),
    )
    monkeypatch.setattr(art, "TrainableModel", lambda **kwargs: model)
    monkeypatch.setattr(art.serverless, "ServerlessBackend", lambda **kwargs: backend)
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(train_rl, "ROOT", tmp_path)
    monkeypatch.setattr(train_rl, "required", lambda key: "test")
    monkeypatch.setattr(train_rl, "Renderer", lambda: SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(wandb, "Api", Mock())
    response = completion(1)
    if missing_metadata:
        del response["nvext"]
    client = AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
        ),
    )
    generated = []
    model.openai_client = Mock(return_value=SimpleNamespace(with_options=Mock(return_value=client)))

    def factory(*args, **kwargs):
        assert kwargs["client"] is client and kwargs["training_logprobs"] is True

        async def generate(prompt, **options):
            generated.append(options)
            await client.chat.completions.create(
                model=BASE, messages=[{"role": "user", "content": prompt}]
            )
            row, _, _ = scored_row()
            return row

        return SimpleNamespace(generate=generate)

    monkeypatch.setattr(train_rl, "BufoAgent", factory)

    async def score(row, *, rescore):
        assert not missing_metadata, "Judge must not be called for unusable training evidence"
        return row["scores"]["scene_quality"]

    monkeypatch.setattr(
        scene_quality,
        "SceneQuality",
        lambda: SimpleNamespace(configuration=lambda: config, score=score),
    )
    trainer = train_rl.RLTrainer(
        "dataset:v0", name="attempt", base_model=BASE, calibration_path=calibration_path
    )
    trainer.registry = Mock()
    trainer.registry.download.return_value = tmp_path
    async with client:
        with pytest.raises(ValueError if missing_metadata else APITimeoutError):
            await trainer.train(steps=1)
    if missing_metadata:
        assert len(generated) >= 1  # the block's rollouts start together
        model.log.assert_not_awaited()
        backend.train.assert_not_awaited()
        state = json.loads((tmp_path / "runs/attempt/rl-training.json").read_text())
        assert state["status"] == "failed" and "training_from_step" not in state
        return
    assert len(generated) == 4 and backend.train.await_count == 1
    state_path = tmp_path / "runs/attempt/rl-training.json"
    saved = state_path.read_bytes()
    state = json.loads(saved)
    assert state["status"] == "failed" and state["training_from_step"] == 0
    captures = list((state_path.parent / "captures").glob("*.json"))
    assert len(captures) == 4
    assert all(json.loads(p.read_text())["status"] == "scored" for p in captures)
    assert all(json.loads(p.read_text())["reward"] > 0 for p in captures)
    assert len((state_path.parent / "manifest.jsonl").read_text().splitlines()) == 4
    with pytest.raises(ValueError, match="already exists"):
        await trainer.train(steps=1)
    assert state_path.read_bytes() == saved
    assert model.register.await_count == 1 and backend.train.await_count == 1


async def test_missing_parquet_dependency_fails_before_dataset_or_registration(
    tmp_path, monkeypatch
):
    def missing(module):
        assert module == "pyarrow"
        raise ModuleNotFoundError("No module named pyarrow")

    monkeypatch.setattr(train_rl, "import_module", missing)
    trainer = train_rl.RLTrainer(
        "dataset:v0", name="unstarted", base_model=BASE, calibration_path=tmp_path / "not-read.json"
    )
    trainer.dataset = Mock(side_effect=AssertionError("Dataset access must not happen"))
    with pytest.raises(ValueError, match="pyarrow"):
        await trainer.train(steps=1)
    trainer.dataset.assert_not_called()


async def test_installed_art_parquet_writer_preserves_real_capture(tmp_path):
    import pyarrow.parquet as parquet
    from art.utils.trajectory_logging import write_trajectory_groups_parquet

    train_rl.require_rl_logging_dependencies()
    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=completion(7)))
        ),
    ) as client:
        with art.Trajectory() as captured:
            await client.chat.completions.create(
                model=BASE, messages=[{"role": "user", "content": "Draw Bufo"}]
            )
    captured.reward = 0.5
    path = tmp_path / "actual-art.parquet"
    write_trajectory_groups_parquet([art.TrajectoryGroup([captured])], path)
    rows = parquet.read_table(path).to_pylist()
    assert len(rows) == 1
    restored = art.Trajectory.model_validate_json(rows[0]["trajectory_json"])
    assert restored.exchanges.chat_completions[0].response.id == "real-http-choice-7"
    assert restored.reward == 0.5
