import json
from types import SimpleNamespace

import pytest
from agents.models.interface import ModelTracing
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from bufogen.agents.reply_model import ImageReplyModel
from bufogen.integrations.telemetry import generation_turn


async def test_image_is_on_real_assistant_chat_after_render_tool(monkeypatch):
    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    uri = "weave:///test/project/object/Content:image"
    calls = []

    class Model:
        async def get_response(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output=[
                    ResponseOutputMessage(
                        id="reply",
                        role="assistant",
                        status="completed",
                        content=[
                            ResponseOutputText(
                                type="output_text",
                                text="<|channel>final\n<channel|>Here's Bufo!",
                                annotations=[],
                            )
                        ],
                        type="message",
                    )
                ],
                response_id="response",
                usage=SimpleNamespace(input_tokens=12, output_tokens=6, total_tokens=18),
            )

    try:
        with generation_turn("Bufo", "gemma", "thread", "Draw") as turn:
            with turn.start_tool(name="render_png") as tool:
                tool.result = {"png": uri}
            model = ImageReplyModel(Model(), turn, "gemma", "coreweave", lambda: uri)
            result = await model.get_response(input="Draw", tracing=ModelTracing.ENABLED)
            assert result.output[0].content[0].text.endswith("Here's Bufo!")
        spans = exporter.get_finished_spans()
        chat = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat")
        tool = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "execute_tool")
        assert tool.end_time <= chat.start_time
        parts = json.loads(chat.attributes["gen_ai.output.messages"])[0]["parts"]
        assert parts[0]["content"] == "Here's Bufo!"
        assert parts[1]["uri"] == uri
        assert list(chat.attributes["weave.content_refs"]) == [uri]
        assert uri not in chat.attributes["gen_ai.input.messages"]
        assert chat.attributes["gen_ai.usage.input_tokens"] == 12
        assert calls[0]["tracing"] == ModelTracing.DISABLED
    finally:
        provider.shutdown()


@pytest.mark.parametrize("image_uri", [None, "weave:///test/project/object/Content:previous"])
async def test_tool_request_records_context_without_attaching_a_final_image(monkeypatch, image_uri):
    from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem

    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)

    class Model:
        async def get_response(self, **kwargs):
            return SimpleNamespace(
                output=[
                    ResponseReasoningItem(
                        id="reason",
                        type="reasoning",
                        summary=[
                            {"type": "summary_text", "text": "Arrange the scene before rendering."}
                        ],
                    ),
                    ResponseFunctionToolCall(
                        id="tool",
                        call_id="plan-1",
                        type="function_call",
                        name="plan_scene",
                        arguments='{"x":90}',
                    ),
                ],
                response_id="first-response",
                usage=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=20,
                    total_tokens=32,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=7),
                ),
            )

    try:
        with generation_turn("Bufo", "qwen", "thread", "Draw") as turn:
            model = ImageReplyModel(Model(), turn, "qwen", "coreweave", lambda: image_uri)
            await model.get_response(
                input="Draw",
                system_instructions="Draw the requested frog.",
                tracing=ModelTracing.ENABLED,
            )
            with turn.start_tool(name="plan_scene", tool_call_id="plan-1") as tool:
                tool.result = {"valid": True}
        spans = exporter.get_finished_spans()
        chats = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat"]
        assert len(chats) == 1
        chat = chats[0]
        assert "Draw the requested frog." in chat.attributes["gen_ai.system_instructions"]
        assert "plan_scene" in chat.attributes["gen_ai.output.messages"]
        assert "plan-1" in chat.attributes["gen_ai.output.messages"]
        assert chat.attributes["gen_ai.usage.reasoning_tokens"] == 7
        assert "Arrange the scene" in str(dict(chat.attributes))
        assert "weave.content_refs" not in chat.attributes
        tool = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "execute_tool")
        assert chat.end_time <= tool.start_time
    finally:
        provider.shutdown()


async def test_failed_first_model_call_keeps_inputs_and_error_in_native_trace(monkeypatch):
    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)

    class Model:
        async def get_response(self, **kwargs):
            raise TimeoutError("Inference timed out")

    try:
        with generation_turn("Bufo", "qwen", "thread", "Draw") as turn:
            model = ImageReplyModel(Model(), turn, "qwen", "coreweave", lambda: None)
            with pytest.raises(TimeoutError):
                await model.get_response(
                    input="Draw", system_instructions="Draw a frog.", tracing=ModelTracing.ENABLED
                )
        chats = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes.get("gen_ai.operation.name") == "chat"
        ]
        assert len(chats) == 1
        assert chats[0].status.status_code == trace.StatusCode.ERROR
        assert "Draw" in chats[0].attributes["gen_ai.input.messages"]
        assert "Draw a frog." in chats[0].attributes["gen_ai.system_instructions"]
    finally:
        provider.shutdown()
