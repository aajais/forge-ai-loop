"""Model wrappers that record one ModelCall per non-streamed model request."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
from openai.types.chat import ChatCompletion
from openai.types.responses import Response


@dataclass
class ModelCall:
    index: int
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None
    reasoning_chars: int
    latency_s: float
    served_model: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _chat_call(index: int, result: ChatCompletion, latency_s: float) -> ModelCall:
    choice = result.choices[0] if result.choices else None
    message = choice.message if choice else None
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    usage = result.usage
    details = usage.completion_tokens_details if usage else None
    return ModelCall(
        index=index,
        finish_reason=choice.finish_reason if choice else None,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        reasoning_tokens=details.reasoning_tokens if details else None,
        reasoning_chars=len(reasoning) if isinstance(reasoning, str) else 0,
        latency_s=latency_s,
        served_model=result.model,
    )


def _responses_call(index: int, result: Response, latency_s: float) -> ModelCall:
    incomplete = result.incomplete_details
    truncated = (
        result.status == "incomplete"
        and incomplete is not None
        and incomplete.reason == "max_output_tokens"
    )
    usage = result.usage
    details = usage.output_tokens_details if usage else None
    return ModelCall(
        index=index,
        finish_reason="length" if truncated else "stop",
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        reasoning_tokens=details.reasoning_tokens if details else None,
        reasoning_chars=0,
        latency_s=latency_s,
        served_model=result.model,
    )


class RecordingChatCompletionsModel(OpenAIChatCompletionsModel):
    def __init__(self, *, model, openai_client, calls: list[ModelCall]):
        super().__init__(model=model, openai_client=openai_client)
        self._calls = calls
        self.last_call = None
        self.last_reasoning = None

    async def _fetch_response(self, *args: Any, **kwargs: Any):
        start = time.monotonic()
        result = await super()._fetch_response(*args, **kwargs)
        if isinstance(result, ChatCompletion):
            self._calls.append(_chat_call(len(self._calls), result, time.monotonic() - start))
            self.last_call = self._calls[-1]
            message = result.choices[0].message if result.choices else None
            self.last_reasoning = getattr(message, "reasoning_content", None) or getattr(
                message, "reasoning", None
            )
        return result


class RecordingResponsesModel(OpenAIResponsesModel):
    def __init__(self, *, model, openai_client, calls: list[ModelCall]):
        super().__init__(model=model, openai_client=openai_client)
        self._calls = calls
        self.last_call = None

    async def _fetch_response(self, *args: Any, **kwargs: Any):
        start = time.monotonic()
        result = await super()._fetch_response(*args, **kwargs)
        if isinstance(result, Response):
            self._calls.append(_responses_call(len(self._calls), result, time.monotonic() - start))
            self.last_call = self._calls[-1]
        return result
