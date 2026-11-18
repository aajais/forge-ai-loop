from opentelemetry import trace
from weave.trace import weave_init


async def test_native_tool_failure_retains_call_id_and_error_after_sdk_wrappers_are_disabled(
    monkeypatch,
):
    from agents import function_tool
    from agents.tool_context import ToolContext
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from bufogen.integrations.telemetry import (
        generation_turn,
        render_tool_failure,
        trace_function_tool,
    )

    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setenv("WANDB_API_KEY", "secret-test-key")

    async def broken_render(svg: str) -> str:
        raise RuntimeError("Renderer failed secret-test-key")

    tool = trace_function_tool(
        function_tool(broken_render, failure_error_function=render_tool_failure)
    )
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="failed-render",
        tool_arguments='{"svg":"<svg/>"}',
    )
    try:
        with generation_turn("Bufo", "qwen", "thread", "Draw"):
            result = await tool.on_invoke_tool(context, context.tool_arguments)
        span = next(
            s
            for s in exporter.get_finished_spans()
            if s.attributes.get("gen_ai.operation.name") == "execute_tool"
        )
        assert span.status.status_code == trace.StatusCode.ERROR
        assert span.attributes["gen_ai.tool.call.id"] == "failed-render"
        assert "Renderer failed" in result
        assert "secret-test-key" not in result
        assert "secret-test-key" not in str(span.status)
        assert "secret-test-key" not in str(span.events)
    finally:
        provider.shutdown()


async def test_render_png_tool_publishes_real_image_and_attaches_media_ref(monkeypatch):
    import json
    from types import SimpleNamespace

    import weave
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from bufogen.agents.prompting import style_references
    from bufogen.integrations.telemetry import generation_turn, publish_render_png
    from bufogen.rendering.renderer import RenderResult

    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    png = style_references()[0].read_bytes()
    uri = "weave:///test/project/object/Content:digest"
    published, inputs = [], []

    def publish(content):
        published.append(content)
        return SimpleNamespace(uri=lambda: uri)

    monkeypatch.setattr(weave, "publish", publish)

    class Renderer:
        async def render(self, svg):
            inputs.append(svg)
            return RenderResult(png, True, None)

    try:
        with generation_turn("Bufo", "gemma", "thread", "Draw") as turn:
            assert await publish_render_png(turn, Renderer(), "<svg/>") == uri
        tools = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes.get("gen_ai.operation.name") == "execute_tool"
        ]
        assert len(tools) == 1 and tools[0].name == "execute_tool render_png"
        assert json.loads(tools[0].attributes["gen_ai.tool.call.arguments"]) == {"svg": "<svg/>"}
        assert json.loads(tools[0].attributes["gen_ai.tool.call.result"])["png"] == uri
        assert list(tools[0].attributes["weave.content_refs"]) == [uri]
        assert inputs == ["<svg/>"] and len(published) == 1
    finally:
        provider.shutdown()


def test_tool_failure_is_actionable_and_redacts_credentials(monkeypatch):
    import json

    from bufogen.integrations.telemetry import render_tool_failure

    monkeypatch.setenv("WANDB_API_KEY", "test-secret-token")
    result = render_tool_failure(None, RuntimeError("Sandbox refused test-secret-token"))
    assert "test-secret-token" not in result
    data = json.loads(result)
    assert data["valid"] is False and data["error_type"] == "RuntimeError"
    assert "Sandbox refused" in data["error"]


def test_agent_exchange_has_its_own_root_inside_http_request(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from weave.conversation import Message

    from bufogen.integrations.telemetry import generation_turn

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    tracer = provider.get_tracer("test")
    try:
        with tracer.start_as_current_span("POST /api/generate") as http:
            for index in range(2):
                with generation_turn(
                    "Bufo open_base",
                    "gemma",
                    "same-conversation",
                    "Draw bufo",
                    image="data:image/png;base64,YQ==" if index else None,
                ) as turn:
                    with tracer.start_as_current_span("chat gemma") as chat:
                        chat.set_attribute("gen_ai.operation.name", "chat")
                    with tracer.start_as_current_span("execute_tool render_svg") as tool:
                        tool.set_attribute("gen_ai.operation.name", "execute_tool")
                    turn.output_messages = [Message(role="assistant", content="Here you go!")]
                assert trace.get_current_span() is http
        spans = exporter.get_finished_spans()
        roots = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent"]
        assert len(roots) == 2
        assert len({s.context.trace_id for s in roots}) == 2
        assert '"mime_type": "image/png"' in roots[1].attributes["gen_ai.input.messages"]
        for root in roots:
            assert root.parent is None
            assert root.name == "invoke_agent Bufo open_base"
            assert root.attributes["gen_ai.agent.name"] == "Bufo open_base"
            assert root.attributes["gen_ai.conversation.id"] == "same-conversation"
            assert "Here you go!" in root.attributes["gen_ai.output.messages"]
            children = [s for s in spans if s.parent and s.parent.span_id == root.context.span_id]
            assert {s.attributes["gen_ai.operation.name"] for s in children} == {
                "chat",
                "execute_tool",
            }
    finally:
        provider.shutdown()


def test_weave_conversation_tracing_initializes_and_changes_project(monkeypatch):
    """Exercise the real exporter integration without exporting spans or using credentials."""
    providers = [trace.ProxyTracerProvider()]
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: providers[-1])
    monkeypatch.setattr(trace, "set_tracer_provider", providers.append)
    monkeypatch.setattr(weave_init, "_conversation_tracer_provider", None)
    monkeypatch.setattr(weave_init, "_conversation_span_exporter", None)
    monkeypatch.setattr(weave_init.env, "weave_trace_server_url", lambda: "http://127.0.0.1:1")

    try:
        weave_init._setup_conversation_tracing("test-entity", "first-project", None)
        provider = weave_init._conversation_tracer_provider
        assert provider is providers[-1]
        assert provider is not None

        # Reinitialization must reroute the same exporter, without recreating the provider.
        weave_init._setup_conversation_tracing("test-entity", "second-project", None)
        assert weave_init._conversation_tracer_provider is provider
        exporter = weave_init._conversation_span_exporter
        assert exporter._session.headers["project_id"] == "test-entity/second-project"
        assert exporter._endpoint == "http://127.0.0.1:1/agents/otel/v1/traces"
    finally:
        for provider in providers[1:]:
            provider.shutdown()
