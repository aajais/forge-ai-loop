import httpx
from agents import ModelSettings
from agents.models.interface import ModelTracing
from openai import AsyncOpenAI

from bufogen.agents.recording_model import (
    ModelCall,
    RecordingChatCompletionsModel,
    RecordingResponsesModel,
)
from bufogen.generation import BufoGeneration


def client_for(payload):
    def handler(request):
        return httpx.Response(200, json=payload)

    return AsyncOpenAI(
        api_key="x",
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def chat_client(
    *,
    finish_reason="stop",
    completion_tokens=5,
    details=None,
    message_extra=None,
    model_name="served",
):
    usage = {"prompt_tokens": 3, "completion_tokens": completion_tokens, "total_tokens": 8}
    if details is not None:
        usage["completion_tokens_details"] = details
    message = {"role": "assistant", "content": "hi", **(message_extra or {})}
    return client_for(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }
    )


def responses_client(*, status="completed", incomplete=None, reasoning_tokens=7):
    return client_for(
        {
            "id": "r1",
            "object": "response",
            "created_at": 0,
            "model": "served-resp",
            "status": status,
            "incomplete_details": incomplete,
            "output": [
                {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 9,
                "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        }
    )


def minimal_get_response_kwargs():
    return dict(
        system_instructions="s",
        input="hi",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )


async def test_chat_records_finish_reason_usage_and_reasoning():
    calls = []
    model = RecordingChatCompletionsModel(
        model="m",
        openai_client=chat_client(
            finish_reason="length",
            completion_tokens=6000,
            details={"reasoning_tokens": 4800},
            message_extra={"reasoning_content": "x" * 42},
            model_name="served-gemma",
        ),
        calls=calls,
    )
    await model.get_response(**minimal_get_response_kwargs())
    assert len(calls) == 1
    assert calls[0].index == 0
    assert calls[0].finish_reason == "length" and calls[0].output_tokens == 6000
    assert calls[0].input_tokens == 3
    assert calls[0].reasoning_tokens == 4800 and calls[0].reasoning_chars == 42
    assert calls[0].served_model == "served-gemma" and calls[0].latency_s >= 0
    assert calls[0].to_dict()["finish_reason"] == "length"


async def test_chat_reasoning_alias_field_and_index_increments():
    calls = []
    model = RecordingChatCompletionsModel(
        model="m",
        openai_client=chat_client(message_extra={"reasoning": "abc"}),
        calls=calls,
    )
    await model.get_response(**minimal_get_response_kwargs())
    await model.get_response(**minimal_get_response_kwargs())
    assert [c.index for c in calls] == [0, 1]
    assert calls[0].reasoning_chars == 3


async def test_chat_missing_details_gives_none_not_zero():
    calls = []
    model = RecordingChatCompletionsModel(model="m", openai_client=chat_client(), calls=calls)
    await model.get_response(**minimal_get_response_kwargs())
    assert calls[0].reasoning_tokens is None
    assert calls[0].reasoning_chars == 0
    assert calls[0].finish_reason == "stop"


async def test_responses_incomplete_maps_to_length():
    calls = []
    model = RecordingResponsesModel(
        model="m",
        openai_client=responses_client(
            status="incomplete", incomplete={"reason": "max_output_tokens"}
        ),
        calls=calls,
    )
    await model.get_response(**minimal_get_response_kwargs())
    assert calls[0].finish_reason == "length"
    assert calls[0].reasoning_tokens == 7 and calls[0].output_tokens == 9
    assert calls[0].served_model == "served-resp"


async def test_responses_completed_maps_to_stop():
    calls = []
    model = RecordingResponsesModel(model="m", openai_client=responses_client(), calls=calls)
    await model.get_response(**minimal_get_response_kwargs())
    assert calls[0].finish_reason == "stop"


def test_model_call_to_dict_roundtrip():
    call = ModelCall(0, "stop", 1, 2, None, 0, 0.5, None)
    assert call.to_dict()["reasoning_tokens"] is None


def test_generation_roundtrip_with_new_fields_and_legacy_rows():
    legacy = dict(
        prompt_id="p",
        prompt="x",
        backend="b",
        model="m",
        group_id="g",
        svg="",
        valid=False,
        latency_s=1.0,
    )
    row = BufoGeneration(**legacy)
    assert row.model_calls == [] and row.truncated_calls == 0
    assert BufoGeneration(**row.to_dict()).model_calls == []
