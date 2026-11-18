"""Real SDK runner/tool callbacks and native OTel spans, with offline model/render boundaries."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import weave
import weave.trace.api as weave_api
from agents import Model
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from bufogen.agents.prompting import style_references
from bufogen.curation.trajectories import export_training_example
from bufogen.models import BackendConfig
from bufogen.rendering.renderer import RenderResult


def module():
    name = "contact_agent_test"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "docs/forge-contact-pilot/contact_agent.py"
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


def scene(m):
    return m.Scene.model_validate({
        "bufo": {"x": 150, "y": 80, "scale": 2},
        "objects": [{
            "id": "mug", "svg": '<rect width="20" height="30" fill="#2374FF"/>',
            "anchors": [{"name": "grip", "x": 10, "y": 15}],
            "transform": {"x": 370, "y": 250},
        }],
        "hands": [{"side": "right", "to": "mug.grip"}],
    }).model_dump(mode="json")


def call(value, call_id):
    return ResponseFunctionToolCall(
        type="function_call", name="render_scene", call_id=call_id,
        arguments=json.dumps({"scene": value, "emoji_name": ":bufo-blue-mug:"}),
    )


def reply():
    return ResponseOutputMessage(
        id="reply", role="assistant", status="completed", type="message",
        content=[ResponseOutputText(type="output_text", text="Here's Bufo with a blue mug!", annotations=[])],
    )


class OfflineModel(Model):
    def __init__(self, outputs):
        self.outputs, self.requests = list(outputs), []

    async def get_response(self, **kwargs):
        self.requests.append(kwargs)
        if not self.outputs:
            raise AssertionError("Unexpected extra model call")
        return ModelResponse(
            output=[self.outputs.pop(0)], usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            response_id=f"response-{len(self.requests)}",
        )

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def offline(monkeypatch):
    m = module()
    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(m, "init_weave", lambda: None)
    logged = []
    monkeypatch.setattr(m, "record_generation", logged.append)
    uri = "weave:///offline/project/object/Content:rendered"
    monkeypatch.setattr(weave, "publish", lambda content: SimpleNamespace(uri=lambda: uri))
    monkeypatch.setattr(weave_api, "publish", lambda content: SimpleNamespace(uri=uri))
    png = style_references()[0].read_bytes()

    class OfflineRenderer:
        def __init__(self):
            self.inputs = []

        async def render(self, svg):
            self.inputs.append(svg)
            return RenderResult(png, True, None)

    backend = BackendConfig(
        name="open_base", label="google/gemma-4-26B-A4B-it",
        model="google/gemma-4-26B-A4B-it", model_env="", provider="coreweave",
        base_url="https://unused.invalid", api_key_env="", base_model="google/gemma-4-26B-A4B-it",
    )
    yield m, provider, exporter, logged, uri, OfflineRenderer, backend
    provider.shutdown()


async def test_real_runner_calls_compiler_and_captures_scene_then_text_png(offline, monkeypatch, tmp_path):
    m, _, exporter, logged, uri, Renderer, backend = offline
    value = scene(m)
    model = OfflineModel([call(value, "real-call-1"), reply()])
    monkeypatch.setattr(m, "OpenAIChatCompletionsModel", lambda **kwargs: model)
    renderer = Renderer()
    agent = m.ContactAgent(backend, renderer, tmp_path, client=object(), benchmark_metadata={"split": "train"})
    row = await agent.generate("Bufo holding a blue mug", prompt_id="mug", group_id="conversation")
    assert row.valid and row.response_text == "Here's Bufo with a blue mug!"
    assert row.emoji_name == ":bufo-blue-mug:"
    assert row.prompt_version.startswith("bufo-contact-v1-")
    assert row.trajectory["system_instructions"] == m.instructions()
    assert row.trajectory["provenance"]["tool_contract"] == m.TOOL_CONTRACT
    assert row.trajectory["provenance"]["scene"] == value
    assert json.loads(Path(row.trajectory["provenance"]["scene_path"]).read_text()) == value
    messages = row.trajectory["messages"]
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"])["scene"] == value
    assert messages[2]["tool_calls"][0]["function"]["name"] == "render_scene"
    assert json.loads(messages[3]["content"])["verification"] == "geometry_only"
    assert row.svg == m.compile_scene(value)["svg"]
    assert all(svg == row.svg for svg in renderer.inputs)
    assert row.png_content_ref == uri and Path(row.png_path).is_file()
    assert logged == [row] and agent.partial_generation.valid
    assert row.trajectory["diagnostics"] == []
    with pytest.raises(ValueError, match="render_svg"):
        export_training_example(row)
    assert "pattern" not in json.dumps(row.trajectory["tools"])
    assert model.requests[0]["system_instructions"] == m.instructions()
    assert model.requests[0]["input"] == [{"role": "user", "content": "Bufo holding a blue mug"}]
    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent"]
    tools = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "execute_tool"]
    chats = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat"]
    assert len(roots) == 1 and roots[0].parent is None
    assert {s.name for s in tools} == {"execute_tool render_scene", "execute_tool render_png"}
    assert len(chats) == 2
    final_chat = max(chats, key=lambda s: s.start_time)
    assert all(s.end_time <= final_chat.start_time for s in tools)
    output = json.loads(final_chat.attributes["gen_ai.output.messages"])
    assert output[0]["parts"][-1]["uri"] == uri
    assert "Here's Bufo" in roots[0].attributes["gen_ai.output.messages"]
    assert len({s.context.trace_id for s in spans}) == 1


async def test_invalid_scene_is_actual_error_then_corrected_call(offline, monkeypatch, tmp_path):
    m, _, exporter, _, _, Renderer, backend = offline
    good = scene(m)
    bad = deepcopy(good)
    bad["hands"][0]["to"] = "absent.grip"
    model = OfflineModel([call(bad, "bad-call"), call(good, "repair-call"), reply()])
    monkeypatch.setattr(m, "OpenAIChatCompletionsModel", lambda **kwargs: model)
    row = await m.ContactAgent(backend, Renderer(), tmp_path, client=object()).generate("Hold a mug")
    assert row.valid and row.tool_calls == 2
    results = [json.loads(msg["content"]) for msg in row.trajectory["messages"] if msg["role"] == "tool"]
    assert results[0]["valid"] is False and "absent" in results[0]["error"]
    assert results[1]["valid"] is True
    attempts = [s for s in exporter.get_finished_spans() if s.name == "execute_tool render_scene"]
    assert len(attempts) == 2
    assert json.loads(attempts[0].attributes["gen_ai.tool.call.result"])["valid"] is False


async def test_media_failure_keeps_valid_render_and_optional_image_is_preserved(offline, monkeypatch, tmp_path):
    m, _, _, _, _, Renderer, backend = offline
    model = OfflineModel([call(scene(m), "image-call"), reply()])
    monkeypatch.setattr(m, "OpenAIChatCompletionsModel", lambda **kwargs: model)

    async def unavailable(*args):
        raise RuntimeError("media unavailable")

    monkeypatch.setattr(m, "publish_render_png", unavailable)
    image = "data:image/png;base64,YQ=="
    row = await m.ContactAgent(backend, Renderer(), tmp_path, client=object()).generate("Use this scene", image=image)
    assert row.valid and Path(row.png_path).exists()
    assert row.png_content_ref is None
    assert model.requests[0]["input"][0]["content"][1]["image_url"] == image
    assert row.trajectory["source_items"][0]["content"][1]["image_url"] == image
    assert row.input_image_path and Path(row.input_image_path).read_bytes() == b"a"


def test_web_search_cannot_silently_change_pilot(offline, tmp_path):
    m, _, _, _, _, Renderer, backend = offline
    with pytest.raises(ValueError, match="Web search"):
        m.ContactAgent(backend, Renderer(), tmp_path, web_search_enabled=True)


async def test_schema_rejection_is_traced_and_does_not_disable_local_identifier_validation(offline, monkeypatch, tmp_path):
    m, _, exporter, _, _, Renderer, backend = offline
    good = scene(m)
    bad = deepcopy(good)
    bad["objects"][0]["id"] = "invalid identifier"
    model = OfflineModel([call(bad, "invalid-schema"), call(good, "fixed-schema"), reply()])
    monkeypatch.setattr(m, "OpenAIChatCompletionsModel", lambda **kwargs: model)
    row = await m.ContactAgent(backend, Renderer(), tmp_path, client=object()).generate("Hold a mug")
    assert row.valid and row.tool_calls == 2
    results = [json.loads(msg["content"]) for msg in row.trajectory["messages"] if msg["role"] == "tool"]
    assert results[0]["valid"] is False and results[1]["valid"] is True
    attempts = [s for s in exporter.get_finished_spans() if s.name == "execute_tool render_scene"]
    assert len(attempts) == 2
    assert "invalid identifier" in attempts[0].attributes["gen_ai.tool.call.arguments"]


async def test_followup_replays_authentic_scene_json_without_invented_svg_tool(offline, monkeypatch, tmp_path):
    m, _, _, _, _, Renderer, backend = offline
    original = scene(m)
    revised = deepcopy(original)
    revised["objects"][0]["svg"] = '<rect width="20" height="30" fill="#ffcc00"/>'
    model = OfflineModel([call(original, "first-scene"), reply(), call(revised, "second-scene"), reply()])
    monkeypatch.setattr(m, "OpenAIChatCompletionsModel", lambda **kwargs: model)
    agent = m.ContactAgent(backend, Renderer(), tmp_path, client=object())
    first = await agent.generate("Hold a blue mug", group_id="same-conversation")
    second = await agent.generate("Make the mug yellow", group_id="same-conversation", history=[first.to_dict()])
    history_calls = [item for item in model.requests[2]["input"] if item.get("type") == "function_call"]
    assert len(history_calls) == 1
    assert history_calls[0]["call_id"] == "first-scene"
    assert history_calls[0]["name"] == "render_scene"
    assert json.loads(history_calls[0]["arguments"])["scene"] == original
    assert second.trajectory["provenance"]["scene"] == revised
    assert second.trajectory["provenance"]["prior_conversation"] is True
    assert len([m for m in second.trajectory["messages"] if m.get("tool_calls")]) == 2


async def test_missing_render_triggers_only_the_existing_bounded_repair_invocation(offline, monkeypatch, tmp_path):
    m, _, _, _, _, Renderer, backend = offline
    model = OfflineModel([reply(), reply()])
    monkeypatch.setattr(m, "OpenAIChatCompletionsModel", lambda **kwargs: model)
    row = await m.ContactAgent(backend, Renderer(), tmp_path, client=object()).generate("Hold a mug")
    assert not row.valid and row.png_path is None
    assert row.response_text == "I couldn't finish rendering this Bufo."
    assert len(model.requests) == 2
    assert "Render status:" in model.requests[1]["input"][-1]["content"]
