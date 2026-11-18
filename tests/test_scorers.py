import json

import pytest

from bufogen.evaluation.scorers.clip_score import normalized_caption
from bufogen.evaluation.scorers.svg_validity import SvgValidity
from bufogen.evaluation.scorers.vlm_judge import VLMJudge, parse_judgment


def test_clip_caption_normalizes_unknown_brand_and_meme_words():
    caption = normalized_caption("CoreWeave bufo riding a cloud")
    assert caption == "a green cartoon frog, riding a cloud"


@pytest.mark.parametrize("value", [True, -1, 2, float("nan"), float("inf"), "0.8"])
def test_judge_rejects_unusable_scores(value):
    data = dict(green_bufo=value, blue_accent=0.8, theme_match=0.8, visual_quality=0.8)
    with pytest.raises(ValueError):
        parse_judgment(json.dumps(data))


async def test_invalid_svg_receives_zero_judge_without_model_call():
    result = await VLMJudge().score({"valid": False, "svg": "<svg>"})
    assert result["overall"] == 0
    assert not SvgValidity().score({"valid": True, "svg": '<svg width="."/>'})["valid"]


def test_judge_keeps_green_and_blue_criteria_separate():
    result = parse_judgment(
        json.dumps(dict(green_bufo=1, blue_accent=0, theme_match=1, visual_quality=1))
    )
    assert result["green_bufo"] == 1 and result["blue_accent"] == 0
    assert result["overall"] == 0.75


async def test_structured_vision_request_recovers_from_empty_truncated_answer():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from bufogen.evaluation.scorers.vlm_judge import vision_json

    create = AsyncMock(
        side_effect=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=None))
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop", message=SimpleNamespace(content='{"usable": true}')
                    )
                ]
            ),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert await vision_json(client, model="Qwen/Qwen3.8-27B", messages=[], max_tokens=700) == {
        "usable": True
    }
    first, second = create.call_args_list
    assert first.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert second.kwargs["max_tokens"] > first.kwargs["max_tokens"]
