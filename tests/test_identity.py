import pytest

from bufogen.evaluation.scorers.bufo_identity import BufoIdentity, identity_result


def test_identity_failure_does_not_confuse_generic_frog_with_bufo():
    result = identity_result(
        dict(
            frog_present=True,
            bufo_present=False,
            similarity=0.2,
            prompt_alignment=0.95,
            reason="Round head and inset eyes",
        )
    )
    assert result["quality_failure"]
    assert "bufo_missing" in result["failure_modes"]
    assert "low_bufo_similarity" in result["failure_modes"]
    assert "prompt_mismatch" not in result["failure_modes"]


def test_identity_accepts_character_in_new_scene_without_exact_pixel_match():
    result = identity_result(
        dict(
            frog_present=True,
            bufo_present=True,
            similarity=0.8,
            prompt_alignment=0.9,
            reason="Bufo proportions preserved",
        )
    )
    assert not result["quality_failure"]


@pytest.mark.parametrize("bad", [True, -1, 2, float("nan"), "0.8"])
def test_similarity_requires_finite_unit_interval(bad):
    with pytest.raises(ValueError):
        identity_result(
            dict(
                frog_present=True,
                bufo_present=True,
                similarity=bad,
                prompt_alignment=1,
                reason="bad score",
            )
        )


async def test_invalid_render_has_explicit_failure_without_vision_request():
    result = await BufoIdentity().score({"valid": False})
    assert result["failure_modes"] == ["invalid_render"]


def test_explicit_failed_constraint_is_not_hidden_by_high_overall_alignment():
    result = identity_result(
        dict(
            frog_present=True,
            bufo_present=True,
            similarity=0.9,
            prompt_alignment=0.95,
            prompt_checks=[
                {
                    "requirement": "one visible eye",
                    "satisfied": False,
                    "evidence": "two visible eyes",
                },
                {"requirement": "blue rack", "satisfied": True, "evidence": "blue rack visible"},
            ],
        )
    )
    assert result["prompt_alignment"] == 0.5
    assert result["failure_modes"] == ["prompt_mismatch"]


async def test_weave_evaluation_replays_current_judgment_without_loading_local_image():
    judgment = identity_result(
        dict(
            frog_present=True,
            bufo_present=True,
            similarity=0.8,
            prompt_alignment=1.0,
        )
    )
    result = await BufoIdentity().score(
        {
            "generation": {
                "valid": True,
                "png_path": "/does-not-exist.png",
                "scores": {"bufo_identity": judgment},
            }
        }
    )
    assert result == judgment
