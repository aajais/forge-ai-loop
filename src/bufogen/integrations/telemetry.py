import asyncio
import json
import logging
import os
from contextlib import contextmanager
from functools import lru_cache, wraps

from bufogen.settings import project_path, required

NATIVE_TRACE_VERSION = "bufo-native-v2"


def current_agent_root() -> dict:
    """Capture the active Conversation turn using the public OTel context API.

    Call at the turn boundary, before entering model/tool child spans. Native span IDs
    are not classic Weave call IDs and must never be passed to client.get_call().
    """
    from opentelemetry import trace

    context = trace.get_current_span().get_span_context()
    return {
        "native_trace_id": format(context.trace_id, "032x") if context.is_valid else None,
        "native_span_id": format(context.span_id, "016x") if context.is_valid else None,
    }


@contextmanager
def native_tool_span(name: str, arguments: str, *, tool_call_id: str = "", description: str = ""):
    """Trace a real tool operation, with its type available while it is running."""
    from weave.conversation import Tool, get_current_conversation

    with Tool(
        name=name,
        arguments=arguments,
        tool_call_id=tool_call_id,
        tool_description=description,
        tool_type="function",
    ) as span:
        attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "gen_ai.tool.call.arguments": arguments,
        }
        if tool_call_id:
            attributes["gen_ai.tool.call.id"] = tool_call_id
        if conversation := get_current_conversation():
            attributes["gen_ai.conversation.id"] = conversation.conversation_id
        span.set_attributes(attributes)
        yield span


def trace_function_tool(tool):
    """Instrument the SDK's actual invocation without adding workflow/turn wrappers."""
    invoke = tool.on_invoke_tool

    @wraps(invoke)
    async def traced(context, arguments):
        with native_tool_span(
            tool.name,
            arguments,
            tool_call_id=getattr(context, "tool_call_id", ""),
            description=tool.description,
        ) as span:
            result = await invoke(context, arguments)
            span.result = result
            return result

    tool.on_invoke_tool = traced
    return tool


async def publish_render_png(turn, renderer, svg: str) -> str:
    """Publish the rendered PNG while the drawing tool is executing."""
    import weave

    # Inherit the active render_svg tool as parent, rather than flattening this
    # application operation into a sibling of the model-requested tools.
    with native_tool_span("render_png", json.dumps({"svg": svg})) as tool:
        tool.set_attributes(
            {
                "bufo.tool_origin": "application",
                "bufo.model_requested": False,
                "gen_ai.tool.description": (
                    "Application rendering and media publication inside render_svg; "
                    "not an additional tool requested by the model."
                ),
            }
        )
        rendered = await renderer.render(svg)
        if not rendered.valid or not rendered.png_bytes:
            raise ValueError(rendered.error or "PNG rendering failed.")
        content = weave.Content.from_bytes(rendered.png_bytes, mimetype="image/png")
        ref = await asyncio.to_thread(weave.publish, content)
        uri = str(ref.uri())
        tool.result = {"png": uri, "mime_type": "image/png"}
        tool.set_attributes({"weave.content_refs": [uri]})
        return uri


@contextmanager
def generation_turn(
    agent_name: str,
    model: str,
    group_id: str,
    prompt: str,
    *,
    image: str | None = None,
    attributes: dict | None = None,
):
    """Keep one user exchange in its own agent-rooted trace, even under HTTP."""
    from weave.conversation import BlobPart, Conversation, Message, TextPart

    with Conversation(
        conversation_id=group_id,
        agent_name=agent_name,
        agent_version=NATIVE_TRACE_VERSION,
        model=model,
        continue_parent_trace=False,
    ) as conversation:
        with conversation.start_turn(user_message=prompt) as turn:
            turn.set_attributes(
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": agent_name,
                    "gen_ai.agent.version": NATIVE_TRACE_VERSION,
                    "gen_ai.request.model": model,
                    "gen_ai.conversation.id": group_id,
                }
            )
            if attributes:
                turn.set_attributes(attributes)
            if image:
                turn.messages = [
                    Message(
                        role="user",
                        parts=[
                            TextPart(content=prompt),
                            BlobPart(
                                mime_type="image/png",
                                modality="image",
                                content=image.split(",", 1)[1],
                            ),
                        ],
                    )
                ]
            yield turn


def render_tool_failure(context, error: Exception) -> str:
    """Keep a real tool failure visible and actionable without leaking credentials."""
    from bufogen.app.ui import public_error

    message = public_error(error)
    # The SDK turns this exception into a model-visible result. Mark the active
    # native tool before that happens, even with SDK wrapper spans disabled.
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    span = trace.get_current_span()
    span.set_status(Status(StatusCode.ERROR, message))
    span.set_attribute("error.type", type(error).__name__)
    span.add_event(
        "exception", {"exception.type": type(error).__name__, "exception.message": message}
    )
    logging.getLogger(__name__).warning("render_svg failed (%s): %s", type(error).__name__, message)
    return json.dumps(
        {
            "valid": False,
            "error_type": type(error).__name__,
            "error": message,
            "next_step": "Retry render_svg once. If it still fails, explain the failure without claiming the image is ready.",
        }
    )


@lru_cache(maxsize=1)
def init_weave():
    required("WANDB_API_KEY")
    # Remove the OpenAI exporter BEFORE Weave registers its native SDK processor.
    from agents import set_trace_processors, set_tracing_disabled

    set_trace_processors([])
    set_tracing_disabled(False)
    os.environ["WEAVE_USE_OTEL_V2"] = "true"
    import weave

    return weave.init(project_path())
