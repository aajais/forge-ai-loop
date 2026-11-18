"""Exercise the live studio's settings at the outgoing HTTP boundary."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from bufogen.agents.agent import BufoAgent
from bufogen.app.ui import create_app
from bufogen.models import BackendConfig, load_backends
from bufogen.rendering.renderer import RenderResult


@pytest.fixture
def offline_telemetry(monkeypatch):
    import bufogen.integrations.observability as observability
    import bufogen.integrations.telemetry as telemetry

    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(observability, "record_generation", lambda row: None)
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("WANDB_ENTITY", "test")
    monkeypatch.setenv("BUFO_DISABLE_THINKING", "true")


def response_reply():
    return httpx.Response(
        200,
        json={
            "id": "r1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6.1-sol",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done", "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {"input_tokens": 3, "output_tokens": 9, "total_tokens": 12},
        },
    )


@pytest.mark.parametrize(
    "name", ["qwen_base", "sft_q3", "rl_q3", "closed_quality_bar", "forge_proxy"]
)
def test_studio_requests_enable_reasoning_without_output_caps(
    name, tmp_path, monkeypatch, offline_telemetry
):
    from test_forge_cli import _chat_response

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return (
            response_reply()
            if name == "closed_quality_bar"
            else _chat_response(len(requests), False, "stop")
        )

    def client_for(backend, **kwargs):
        return AsyncOpenAI(
            api_key="test",
            base_url="http://mock.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(BackendConfig, "client", client_for)
    with TestClient(create_app(output_dir=tmp_path)) as client:
        status = {m["id"]: m for m in client.get("/api/status").json()["models"]}
        assert status[name]["runtime_policy"]["max_tokens"] is None
        assert status[name]["runtime_policy"]["disable_thinking"] is False
        result = client.post("/api/generate", json={"prompt": "Bufo", "backend": name})
        assert result.status_code == 200, result.text
        row = result.json()
    assert len(requests) == 2  # Both the original attempt and its automatic repair.
    for request in requests:
        assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"} & request.keys()
        if name in {"qwen_base", "sft_q3", "rl_q3"}:
            assert request["chat_template_kwargs"]["enable_thinking"] is True
        elif name == "closed_quality_bar":
            assert request["reasoning"]["effort"] == "medium"
    assert row["execution_metadata"]["max_turns"] == 100
    assert row["execution_metadata"]["runtime_policy"]["max_tokens"] is None
    assert row["trajectory"]["provenance"]["disable_thinking"] is False


async def test_studio_repair_shares_total_turn_budget(tmp_path, offline_telemetry):
    from agents import MaxTurnsExceeded, trace

    from test_forge_cli import _chat_response

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        # First attempt finishes without drawing; repair keeps using the tool.
        tool = len(requests) > 1
        return _chat_response(len(requests), tool, "tool_calls" if tool else "stop")

    class Renderer:
        async def render(self, svg):
            return RenderResult(None, False, "Invalid drawing")

    async with AsyncOpenAI(
        api_key="test",
        base_url="http://mock.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        agent = BufoAgent(
            load_backends()["qwen_base"],
            Renderer(),
            tmp_path,
            client=client,
            studio_mode=True,
            max_turns=3,
        )
        with trace("test", disabled=True), pytest.raises(MaxTurnsExceeded):
            await agent.generate("Bufo")
    assert len(requests) == 3
