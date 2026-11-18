"""Real ART/OpenAI transport and preprocessing with deterministic HTTP fixtures."""

import copy
import json
from types import SimpleNamespace

import art
import httpx
import pytest
from agents import ModelSettings, OpenAIChatCompletionsModel
from agents.models.interface import ModelTracing

from bufogen.training import rl as train_rl

BASE = "google/gemma-4-26B-A4B-it"


def engine_response():
    return {
        "id": "provider-response-1",
        "created": 0,
        "object": "chat.completion",
        "model": BASE,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Here is Bufo"},
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


async def captured_dynamo(payload=None):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload or engine_response())

    model = art.TrainableModel(name="unit", run_name="unit", project="unit", base_model=BASE)
    model.inference_api_key = "test"
    # Serverless registration defaults to the training gateway; route only
    # inference to its documented endpoint without losing ART's wrapper.
    model.inference_base_url = "https://api.training.wandb.ai/v1"
    parent = model.openai_client()
    client = parent.with_options(
        base_url="https://api.inference.wandb.ai/v1",
        default_headers={"OpenAI-Project": "team/project"},
        max_retries=0,
        timeout=180,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        llm = OpenAIChatCompletionsModel(model=BASE, openai_client=client)
        with art.Trajectory() as trajectory:
            await llm.get_response(
                system_instructions="Draw Bufo",
                input="Hold a cup",
                model_settings=ModelSettings(extra_args={"logprobs": True}),
                tools=[],
                output_schema=None,
                handoffs=[],
                tracing=ModelTracing.DISABLED,
            )
        return trajectory, requests
    finally:
        await client.close()
        await parent.close()


async def test_managed_client_keeps_dynamo_options_through_agents_sdk_and_tokenizes():
    trajectory, requests = await captured_dynamo()
    assert len(requests) == 1
    request = requests[0]
    assert request.url.host == "api.inference.wandb.ai"
    body = json.loads(request.content)
    assert request.headers["OpenAI-Project"] == "team/project"
    assert body["logprobs"] is True
    assert body["nvext"]["extra_fields"] == ["engine_data"]
    assert body["return_token_ids"] is True
    assert body["return_tokens_as_token_ids"] is True
    assert body["chat_template_kwargs"]["preserve_thinking"] is True
    assert train_rl.validate_training_tokens(trajectory, BASE) == {
        "exchanges": [{"prompt_tokens": 2, "completion_tokens": 2}],
        "histories": 1,
        "history_tokens": [4],
    }
    # Serialization preserves exact provider evidence and the SDK mapping.
    restored = art.Trajectory.model_validate_json(trajectory.model_dump_json())
    assert train_rl.validate_training_tokens(restored, BASE) == train_rl.validate_training_tokens(
        trajectory, BASE
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unfinished",
        "empty_prompt",
        "empty_output",
        "bad_id",
        "missing_logprobs",
        "short_logprobs",
        "nan",
        "bool",
        "conflicting_mapping",
    ],
)
async def test_missing_or_invalid_training_evidence_is_rejected(mutation):
    trajectory, _ = await captured_dynamo()
    response = trajectory.exchanges.chat_completions[0].response
    engine = response.model_extra["nvext"]["engine_data"]
    if mutation == "missing":
        del response.model_extra["nvext"]
    elif mutation == "unfinished":
        engine["finished"] = False
    elif mutation == "empty_prompt":
        engine["prompt_token_ids"] = []
    elif mutation == "empty_output":
        engine["completion_token_ids"] = []
    elif mutation == "bad_id":
        engine["prompt_token_ids"] = [True, 2]
    elif mutation == "missing_logprobs":
        del engine["completion_logprobs"]
    elif mutation == "short_logprobs":
        engine["completion_logprobs"] = [-0.2]
    elif mutation == "nan":
        engine["completion_logprobs"] = [float("nan"), -0.3]
    elif mutation == "bool":
        engine["completion_logprobs"] = [True, -0.3]
    else:
        response.choices[0].model_extra["token_ids"] = [5, 6]
    with pytest.raises(ValueError):
        train_rl.validate_training_tokens(trajectory, BASE)


def test_same_package_version_without_verified_revision_is_not_accepted(monkeypatch):
    source = {
        "url": train_rl.ART_REPOSITORY,
        "vcs_info": {"vcs": "git", "commit_id": train_rl.ART_REVISION},
    }
    monkeypatch.setattr(train_rl, "version", lambda _: "0.5.20")
    for replacement in ({}, {**source, "vcs_info": {"vcs": "git", "commit_id": "other"}}):
        monkeypatch.setattr(
            train_rl,
            "distribution",
            lambda _: SimpleNamespace(read_text=lambda _: json.dumps(replacement)),
        )
        with pytest.raises(ValueError, match="Dynamo RL runtime"):
            train_rl.require_rl_runtime()
    monkeypatch.setattr(
        train_rl,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: json.dumps(copy.deepcopy(source))),
    )
    assert train_rl.require_rl_runtime()["revision"] == train_rl.ART_REVISION
