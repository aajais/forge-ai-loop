"""Contracts for evaluated model budgets at the studio/agent boundary."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from bufogen.agents.agent import BufoAgent
from bufogen.app.ui import create_app
from bufogen.models import load_backends


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    for name in (
        "BUFO_DISABLE_THINKING",
        "BUFO_INFERENCE_TIMEOUT_S",
        "QWEN_BASE_MODEL",
        "SFT_Q3_MODEL",
        "RL_Q3_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("name", ["qwen_base", "sft_q3"])
def test_agent_applies_evaluated_qwen_budget(name, tmp_path):
    agent = BufoAgent(load_backends()[name], None, tmp_path)
    assert agent.max_tokens == 20000
    assert agent.disable_thinking is False
    assert agent.runtime_policy.timeout_s == 900


def test_explicit_agent_settings_override_qwen_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("BUFO_INFERENCE_TIMEOUT_S", "345")
    monkeypatch.setenv("BUFO_DISABLE_THINKING", "true")
    agent = BufoAgent(
        load_backends()["qwen_base"], None, tmp_path, max_tokens=7777, disable_thinking=False
    )
    assert agent.max_tokens == 7777
    assert agent.disable_thinking is False
    assert agent.runtime_policy.timeout_s == 345


@pytest.mark.parametrize(("name", "tokens"), [("open_base", 6000), ("closed_quality_bar", 10000)])
def test_other_model_budgets_remain_unchanged(name, tokens, tmp_path):
    agent = BufoAgent(load_backends()[name], None, tmp_path)
    assert agent.max_tokens == tokens
    assert agent.runtime_policy.timeout_s == 180


@pytest.mark.parametrize(
    "model",
    [
        "wandb-artifact:///e/p/PENDING",
        "wandb-artifact:///e/p/m:latest",
        "wandb-artifact:///e/p/m",
        "PENDING",
    ],
)
def test_unpinned_checkpoint_cannot_be_served(model):
    backend = replace(load_backends()["sft_q3"], model=model)
    with pytest.raises(ValueError, match="pinned|PENDING"):
        backend.require_configured()


def test_status_exposes_policy_and_disables_pending_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("RL_Q3_MODEL", "wandb-artifact:///team/project/PENDING")
    with TestClient(create_app(output_dir=tmp_path)) as client:
        models = {m["id"]: m for m in client.get("/api/status").json()["models"]}
        assert models["qwen_base"]["runtime_policy"]["max_tokens"] is None
        assert models["sft_q3"]["runtime_policy"]["timeout_s"] == 900
        assert not models["rl_q3"]["available"]
        assert "PENDING" in models["rl_q3"]["unavailable_reason"]
        assert (
            client.post("/api/generate", json={"prompt": "Bufo", "backend": "rl_q3"}).status_code
            == 409
        )


def test_qwen_client_timeout_tracks_evaluated_policy(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    client = load_backends()["qwen_base"].client()
    assert client.timeout == 900


def test_qwen_explicitly_enables_thinking():
    assert load_backends()["qwen_base"].thinking_extra_body(False) == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_training_can_use_step_alias_without_exposing_it_as_demo(tmp_path):
    backend = replace(load_backends()["sft_q3"], model="wandb-artifact:///e/p/m:step2")
    agent = BufoAgent(backend, None, tmp_path, training_logprobs=True, client=object())
    assert agent.backend.model.endswith(":step2")
    assert agent.execution_metadata["origin"] == "training"
    with pytest.raises(ValueError, match="pinned"):
        BufoAgent(backend, None, tmp_path)


def test_controlled_policy_ignores_environment_overrides(monkeypatch):
    from bufogen.runtime import resolve_inference_policy

    monkeypatch.setenv("BUFO_INFERENCE_TIMEOUT_S", "300")
    monkeypatch.setenv("BUFO_DISABLE_THINKING", "true")
    policy = resolve_inference_policy(load_backends()["qwen_base"], use_environment=False)
    assert policy.timeout_s == 900
    assert policy.disable_thinking is False


@pytest.mark.parametrize(
    "options", [{"max_tokens": 0}, {"max_tokens": True}, {"inference_timeout_s": 0}]
)
def test_invalid_runtime_budget_rejected_before_serving(tmp_path, options):
    with pytest.raises(ValueError):
        BufoAgent(load_backends()["qwen_base"], None, tmp_path, **options)


async def test_qwen_http_requests_and_saved_provenance_use_same_policy(tmp_path, monkeypatch):
    from test_forge_cli import _run_open_base

    monkeypatch.setenv("OPEN_MODEL", "Qwen/Qwen3.8-27B")
    row, requests = await _run_open_base(
        tmp_path, monkeypatch, [(True, "tool_calls"), (False, "stop")]
    )
    assert all(request["max_tokens"] == 20000 for request in requests)
    assert all(request["chat_template_kwargs"] == {"enable_thinking": True} for request in requests)
    policy = row.trajectory["provenance"]["runtime_policy"]
    assert policy["max_tokens"] == 20000
    # This helper injects an OpenAI client with the SDK default read timeout.
    assert policy["timeout_s"] == 600
    assert row.trajectory["provenance"]["timeout_source"] == "injected_client"
    assert policy["disable_thinking"] is False


async def test_agent_persists_original_root_and_runtime_metadata(tmp_path, monkeypatch):
    import json

    import httpx
    from agents import trace as agents_trace
    from openai import AsyncOpenAI
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from test_forge_cli import _chat_response

    import bufogen.integrations.observability as observability
    import bufogen.integrations.telemetry as telemetry
    from bufogen.rendering.renderer import RenderResult

    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(observability, "record_generation", lambda row: None)

    async def publish(*args):
        return None

    monkeypatch.setattr(telemetry, "publish_render_png", publish)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return _chat_response(
            len(requests), len(requests) == 1, "tool_calls" if len(requests) == 1 else "stop"
        )

    class Renderer:
        async def render(self, svg):
            return RenderResult(b"png", True, None)

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
            web_search_enabled=False,
            benchmark_metadata={"split": "validation"},
        )
        with agents_trace("test", disabled=True):
            row = await agent.generate("Draw Bufo", group_id="thread-1")
    roots = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    ]
    assert len(roots) == 1
    assert row.native_trace_id == format(roots[0].context.trace_id, "032x")
    assert row.native_span_id == format(roots[0].context.span_id, "016x")
    assert agent.partial_generation.native_span_id == row.native_span_id
    assert row.execution_metadata["runtime_policy"]["max_tokens"] == 20000
    assert roots[0].attributes["bufo.max_tokens"] == 20000
    provider.shutdown()


@pytest.mark.parametrize("read_timeout", [180, 245.5, None])
async def test_injected_training_client_records_actual_timeout(tmp_path, read_timeout):
    import httpx
    from openai import AsyncOpenAI

    configured = httpx.Timeout(connect=5, read=read_timeout, write=60, pool=10)
    async with AsyncOpenAI(api_key="test", timeout=configured) as client:
        agent = BufoAgent(
            load_backends()["qwen_base"],
            None,
            tmp_path,
            client=client,
            training_logprobs=True,
        )
        assert agent.runtime_policy.timeout_s == read_timeout
        assert agent.execution_metadata["runtime_policy"]["timeout_s"] == read_timeout
        assert agent.execution_metadata["client_timeout"] == {
            "connect": 5,
            "read": read_timeout,
            "write": 60,
            "pool": 10,
        }
        assert client.timeout == configured


async def test_injected_scalar_timeout_overrides_requested_policy_for_provenance(tmp_path):
    from openai import AsyncOpenAI

    async with AsyncOpenAI(api_key="test", timeout=180) as client:
        agent = BufoAgent(
            load_backends()["qwen_base"], None, tmp_path, client=client, inference_timeout_s=900
        )
        assert agent.runtime_policy.timeout_s == 180
        assert client.timeout == 180


@pytest.mark.parametrize("timeout", ["soon", "0"])
def test_invalid_timeout_is_reported_as_unavailable_in_studio(tmp_path, monkeypatch, timeout):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("BUFO_INFERENCE_TIMEOUT_S", timeout)
    with TestClient(create_app(output_dir=tmp_path), raise_server_exceptions=False) as client:
        response = client.get("/api/status")
        assert response.status_code == 200
        models = {model["id"]: model for model in response.json()["models"]}
        assert models["qwen_base"]["runtime_policy"] is None
        assert not models["qwen_base"]["available"]
        assert "BUFO_INFERENCE_TIMEOUT_S" in models["qwen_base"]["unavailable_reason"]
        generation = client.post("/api/generate", json={"prompt": "Bufo", "backend": "qwen_base"})
        assert generation.status_code == 409
        assert "BUFO_INFERENCE_TIMEOUT_S" in generation.json()["detail"]
    with pytest.raises(ValueError, match="BUFO_INFERENCE_TIMEOUT_S"):
        BufoAgent(load_backends()["qwen_base"], None, tmp_path)
