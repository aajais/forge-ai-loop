"""Real Agents + Weave native span ordering with only HTTP/media publication mocked."""

import asyncio
import json
from contextlib import AsyncExitStack
from types import SimpleNamespace

import httpx
import pytest
from agents.tracing import get_trace_provider, set_trace_provider
from agents.tracing.provider import DefaultTraceProvider
from openai import AsyncOpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from weave.integrations.openai_agents.otel_processor import WeaveOtelTracingProcessor

from bufogen.agents.agent import BufoAgent, instructions
from bufogen.models import load_backends
from bufogen.rendering.renderer import RenderResult


def assert_otlp_exchanges(spans, rows):
    """Validate the actual wire payload against CoreWeave's Agents-view contract."""
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    wire = ExportTraceServiceRequest.FromString(encode_spans(spans).SerializeToString())
    exported = [
        s for resource in wire.resource_spans for scope in resource.scope_spans for s in scope.spans
    ]
    assert {s.trace_id.hex() for s in exported} == {row.native_trace_id for row in rows}
    for row in rows:
        turn = [s for s in exported if s.trace_id.hex() == row.native_trace_id]
        by_id = {s.span_id: s for s in turn}
        roots = [s for s in turn if not s.parent_span_id]
        assert len(roots) == 1
        root = roots[0]
        attrs = {a.key: a.value.string_value for a in root.attributes}
        assert root.span_id.hex() == row.native_span_id
        assert attrs["gen_ai.operation.name"] == "invoke_agent"
        assert attrs["gen_ai.agent.name"].startswith("Bufo ")
        assert attrs["gen_ai.conversation.id"] == row.group_id
        assert attrs["gen_ai.agent.version"] == "bufo-native-v2"
        assert any(m["role"] == "assistant" for m in json.loads(attrs["gen_ai.output.messages"]))
        final_responses = []
        for s in turn:
            fields = {a.key: a.value.string_value for a in s.attributes}
            assert fields["gen_ai.conversation.id"] == row.group_id
            assert fields["gen_ai.operation.name"] in {"invoke_agent", "chat", "execute_tool"}
            if s.parent_span_id:
                parent = by_id[s.parent_span_id]  # No orphan/split parent IDs.
                assert parent.start_time_unix_nano <= s.start_time_unix_nano
                assert s.end_time_unix_nano <= parent.end_time_unix_nano
            if fields["gen_ai.operation.name"] == "chat":
                messages = json.loads(fields.get("gen_ai.output.messages", "[]"))
                if any(
                    m["role"] == "assistant"
                    and any(
                        p.get("type") == "text" and row.response_text in p.get("content", "")
                        for p in m.get("parts", [])
                    )
                    for m in messages
                ):
                    final_responses.append(s)
        assert final_responses, "Missing the final assistant response in the agent trace"
        tools = [
            s
            for s in turn
            if any(
                a.key == "gen_ai.operation.name" and a.value.string_value == "execute_tool"
                for a in s.attributes
            )
        ]
        assert tools
        assert max(s.end_time_unix_nano for s in tools) <= final_responses[-1].start_time_unix_nano


@pytest.mark.parametrize("backend_name", ["qwen_base", "closed_quality_bar"])
async def test_native_timeline_records_prompt_then_plan_render_and_final_reply(
    tmp_path, monkeypatch, backend_name
):
    import weave

    import bufogen.integrations.observability as observability
    import bufogen.integrations.telemetry as telemetry

    backend = load_backends()[backend_name]
    provider, exporter = TracerProvider(), InMemorySpanExporter()
    started = []

    class CaptureLiveSpans(SimpleSpanProcessor):
        def on_start(self, span, parent_context=None):
            started.append(span)

    provider.add_span_processor(CaptureLiveSpans(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    previous_provider = get_trace_provider()
    sdk_provider = DefaultTraceProvider()
    sdk_provider.set_processors([WeaveOtelTracingProcessor()])
    set_trace_provider(sdk_provider)
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(observability, "record_generation", lambda row: None)
    uri = "weave:///test/project/object/Content:image"
    monkeypatch.setattr(weave, "publish", lambda content: SimpleNamespace(uri=lambda: uri))
    requests = []
    live_snapshots = []

    def snapshot():
        live_snapshots.append([(s.name, dict(s.attributes)) for s in started if s.is_recording()])

    def handler(request):
        snapshot()
        requests.append(json.loads(request.content))
        index = len(requests)
        message = {"role": "assistant", "content": "Here is Bufo!"}
        if index <= 2:
            args = (
                {
                    "main_bufo": {"x": 90, "y": 80, "scale": 2.2},
                    "coordinate_frame": "bufo_local",
                    "contact_anchors": [],
                    "paint_order": ["bufo"],
                    "framing": "fullbody",
                }
                if index == 1
                else {"svg": "<svg/>", "emoji_name": ":bufo-test:"}
            )
            message = {
                "role": "assistant",
                "content": None,
                "reasoning": "Place Bufo before adding the scenery.",
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {
                            "name": "plan_scene" if index == 1 else "render_svg",
                            "arguments": json.dumps(args),
                        },
                    }
                ],
            }
        if backend.provider == "openai":
            output = [
                {
                    "type": "reasoning",
                    "id": f"reason-{index}",
                    "summary": [
                        {"type": "summary_text", "text": "Place Bufo before adding the scenery."}
                    ],
                }
            ]
            if index <= 2:
                output.append(
                    {
                        "type": "function_call",
                        "id": f"tool-{index}",
                        "call_id": f"call-{index}",
                        "name": message["tool_calls"][0]["function"]["name"],
                        "arguments": json.dumps(args),
                    }
                )
            else:
                output.append(
                    {
                        "type": "message",
                        "id": "final",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "Here is Bufo!", "annotations": []}
                        ],
                    }
                )
            return httpx.Response(
                200,
                json={
                    "id": f"response-{index}",
                    "object": "response",
                    "created_at": 0,
                    "model": backend.model,
                    "status": "completed",
                    "output": output,
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": f"chat-{index}",
                "object": "chat.completion",
                "created": 0,
                "model": "served-qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if index <= 2 else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
        )

    class Renderer:
        async def render(self, svg):
            snapshot()
            return RenderResult(b"png", True, None)

    try:
        async with AsyncOpenAI(
            api_key="test",
            base_url="http://mock.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            agent = BufoAgent(backend, Renderer(), tmp_path, client=client, studio_mode=True)
            row = await agent.generate("Bufo", group_id="timeline-test")
        spans = exporter.get_finished_spans()
        assert_otlp_exchanges(spans, [row])
        assert all(
            s.attributes.get("gen_ai.operation.name") in {"invoke_agent", "chat", "execute_tool"}
            for s in spans
        ), [(s.name, s.attributes.get("gen_ai.operation.name")) for s in spans]
        assert sum(s.attributes.get("gen_ai.operation.name") == "invoke_agent" for s in spans) == 1
        for snapshot_rows in live_snapshots:
            for name, attrs in snapshot_rows:
                assert attrs.get("gen_ai.operation.name") in {
                    "invoke_agent",
                    "chat",
                    "execute_tool",
                }, name
                if attrs["gen_ai.operation.name"] == "invoke_agent":
                    assert attrs["gen_ai.agent.name"] == f"Bufo {backend.display_name}"
                elif attrs["gen_ai.operation.name"] == "chat":
                    assert attrs["gen_ai.request.model"] == backend.model
                else:
                    assert attrs["gen_ai.tool.name"] in {"plan_scene", "render_svg", "render_png"}
        root = next(s for s in spans if s.context.span_id == int(row.native_span_id, 16))
        assert (
            json.loads(root.attributes["gen_ai.system_instructions"])[0]["content"]
            == instructions()
        )
        definitions = json.loads(root.attributes["gen_ai.tool.definitions"])
        if backend.provider == "openai":
            assert [d["function"]["name"] for d in definitions] == [
                t["name"] for t in requests[0]["tools"]
            ]
        else:
            assert definitions == requests[0]["tools"]
        chats = sorted(
            [s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat"],
            key=lambda s: s.start_time,
        )
        assert len(chats) == len(requests) == 3
        tools = {
            s.attributes.get("gen_ai.tool.name"): s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "execute_tool"
        }
        assert set(tools) == {"plan_scene", "render_svg", "render_png"}
        plan, render, png = (tools[name] for name in ("plan_scene", "render_svg", "render_png"))
        assert plan.attributes["gen_ai.tool.call.id"] == "call-1"
        assert render.attributes["gen_ai.tool.call.id"] == "call-2"
        assert root.start_time <= chats[0].start_time
        assert chats[0].end_time <= plan.start_time <= plan.end_time <= chats[1].start_time
        assert (
            chats[1].end_time
            <= render.start_time
            <= png.start_time
            <= png.end_time
            <= render.end_time
        )
        assert render.end_time <= chats[2].start_time <= chats[2].end_time <= root.end_time
        assert png.parent.span_id == render.context.span_id
        for chat in chats:
            assert chat.context.trace_id == root.context.trace_id
            assert (
                json.loads(chat.attributes["gen_ai.system_instructions"])[0]["content"]
                == instructions()
            )
            assert chat.attributes["gen_ai.usage.input_tokens"] == 10
        assert "plan_scene" in chats[0].attributes["gen_ai.output.messages"]
        assert "Place Bufo" in str(dict(chats[0].attributes))
        assert chats[0].attributes["gen_ai.response.model"] == (
            backend.model if backend.provider == "openai" else "served-qwen"
        )
        assert list(chats[0].attributes["gen_ai.response.finish_reasons"]) == (
            ["stop"] if backend.provider == "openai" else ["tool_calls"]
        )
        assert "call-1" in chats[1].attributes["gen_ai.input.messages"]
        assert uri not in chats[1].attributes["gen_ai.output.messages"]
        assert uri in chats[2].attributes["gen_ai.output.messages"]
        assert json.loads(render.attributes["gen_ai.tool.call.result"])["valid"] is True
    finally:
        set_trace_provider(previous_provider)
        sdk_provider.shutdown()
        provider.shutdown()


async def test_repairs_and_parallel_conversations_keep_complete_turns_in_separate_traces(
    tmp_path, monkeypatch
):
    import weave

    import bufogen.integrations.observability as observability
    import bufogen.integrations.telemetry as telemetry
    from test_forge_cli import _chat_response
    from test_studio_runtime import response_reply

    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    previous_provider = get_trace_provider()
    sdk_provider = DefaultTraceProvider()
    sdk_provider.set_processors([WeaveOtelTracingProcessor()])
    set_trace_provider(sdk_provider)
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(observability, "record_generation", lambda row: None)
    monkeypatch.setattr(
        weave,
        "publish",
        lambda content: SimpleNamespace(uri=lambda: "weave:///test/project/object/Content:image"),
    )

    class Renderer:
        async def render(self, svg):
            await asyncio.sleep(0)
            return RenderResult(b"png", True, None)

    def transport(backend):
        count = 0

        async def handler(request):
            nonlocal count
            count += 1
            index = count
            await asyncio.sleep(0)  # Interleave requests from independent conversations.
            # First run ends without rendering; its repair uses a tool, then replies.
            tool = index % 3 == 2
            if backend.provider != "openai":
                return _chat_response(index, tool, "tool_calls" if tool else "stop")
            body = response_reply().json()
            body["id"] = f"response-{index}"
            if tool:
                body["output"] = [
                    {
                        "type": "function_call",
                        "id": f"tool-{index}",
                        "call_id": f"call-{index}",
                        "name": "render_svg",
                        "arguments": json.dumps({"svg": "<svg/>", "emoji_name": ":bufo-test:"}),
                    }
                ]
            return httpx.Response(200, json=body)

        return httpx.MockTransport(handler)

    try:
        async with AsyncExitStack() as stack:
            agents = []
            for name in ("qwen_base", "closed_quality_bar"):
                backend = load_backends()[name]
                client = await stack.enter_async_context(
                    AsyncOpenAI(
                        api_key="test",
                        base_url="http://mock.invalid/v1",
                        http_client=httpx.AsyncClient(transport=transport(backend)),
                    )
                )
                agents.append(
                    BufoAgent(backend, Renderer(), tmp_path / name, client=client, studio_mode=True)
                )
            rows = await asyncio.gather(
                *[
                    agent.generate("Bufo", group_id=f"conversation-{i}")
                    for i, agent in enumerate(agents)
                ]
            )
            rows.append(
                await agents[1].generate(
                    "Add a scarf", group_id=rows[1].group_id, history=[rows[1].to_dict()]
                )
            )
        assert all(row.valid for row in rows)
        assert rows[1].group_id == rows[2].group_id
        assert len({row.native_trace_id for row in rows}) == 3
        assert_otlp_exchanges(exporter.get_finished_spans(), rows)
    finally:
        set_trace_provider(previous_provider)
        sdk_provider.shutdown()
        provider.shutdown()
