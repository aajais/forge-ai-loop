"""Trace every real model call natively, attaching images only to final replies."""

from agents import Model
from agents.models.interface import ModelTracing
from weave.conversation import Message, TextPart, UriPart, Usage, get_current_conversation
from weave.conversation.adapters.openai import (
    message_from_openai_responses_input,
    reasoning_from_openai_responses,
)
from weave.trace.context.call_context import tracing_disabled

from bufogen.rendering.svg import extract_svg


class ImageReplyModel(Model):
    def __init__(self, inner, turn, model_name, provider, image_uri):
        self.inner = inner
        self.turn = turn
        self.model_name = model_name
        self.provider = provider
        self.image_uri = image_uri
        self.call_index = 0

    async def get_response(self, **kwargs):
        uri = self.image_uri()
        self.call_index += 1
        # One native span per real call, including calls before the first image.
        # Tools are independently instrumented under the same native turn.
        with self.turn.start_llm(
            model=self.model_name,
            provider_name=self.provider,
            system_instructions=[kwargs.get("system_instructions") or ""],
        ) as llm:
            attributes = {
                "bufo.model_call_index": self.call_index,
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": self.model_name,
                "gen_ai.provider.name": self.provider,
            }
            if conversation := get_current_conversation():
                attributes["gen_ai.conversation.id"] = conversation.conversation_id
            llm.set_attributes(attributes)
            value = kwargs["input"]
            inputs, attachments = message_from_openai_responses_input(
                [{"role": "user", "content": value}] if isinstance(value, str) else value
            )
            # Assign inputs before awaiting so failed/cancelled calls keep their context.
            llm.record(input_messages=inputs, media_attachments=attachments)
            settings = kwargs.get("model_settings")
            if settings is not None:
                llm.request_temperature = settings.temperature
                llm.request_max_tokens = settings.max_tokens
                llm.request_top_p = settings.top_p
            # Suppress duplicate SDK/HTTP model spans only; tools execute outside this scope.
            with tracing_disabled():
                response = await self.inner.get_response(
                    **{**kwargs, "tracing": ModelTracing.DISABLED}
                )
            items = [item.model_dump(exclude_none=True) for item in response.output]
            outputs, _ = message_from_openai_responses_input(items)
            reasoning = [
                part.content
                for item in items
                if item.get("type") == "reasoning"
                if (part := reasoning_from_openai_responses(item)) is not None
            ]
            call = getattr(self.inner, "last_call", None)
            # A further tool request is not the final reply. Only attach the PNG
            # to text-only assistant completions, never to a repair/tool request.
            if uri and not any(item.get("type") == "function_call" for item in items):
                for message in reversed(outputs):
                    if message.role == "assistant":
                        parts = list(message.parts)
                        if not parts and message.content:
                            parts = [TextPart(content=extract_svg(message.content))]
                        parts.append(UriPart(uri=uri, modality="image", mime_type="image/png"))
                        outputs[outputs.index(message)] = Message(role="assistant", parts=parts)
                        llm.set_attributes({"weave.content_refs": [uri]})
                        break
            input_details = getattr(response.usage, "input_tokens_details", None)
            output_details = getattr(response.usage, "output_tokens_details", None)
            llm.record(
                input_messages=inputs,
                output_messages=outputs,
                reasoning="\n".join(reasoning) or getattr(self.inner, "last_reasoning", None),
                response_id=response.response_id or "",
                response_model=call.served_model if call else None,
                finish_reasons=[call.finish_reason] if call and call.finish_reason else None,
                usage=Usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                    cache_read_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
                    cache_creation_input_tokens=getattr(input_details, "cache_write_tokens", 0)
                    or 0,
                    reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
                ),
            )
            return response

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError("ImageReplyModel uses the studio's non-streaming agent runner.")

    def get_retry_advice(self, request):
        return self.inner.get_retry_advice(request)
