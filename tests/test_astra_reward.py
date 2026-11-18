import copy

from bufogen.generation import BufoGeneration
from bufogen.training.common import astra_reward

CONFIG = {"rubric_version": "bufo-astra-v5", "samples": 3, "threshold": 0.65}


def astra_row(**overrides):
    row = BufoGeneration(
        prompt_id="p",
        prompt="Bufo grips a pointer",
        backend="open_base",
        model="student",
        group_id="g",
        svg="<svg/>",
        valid=True,
        latency_s=1,
        benchmark_metadata={"requirements": ["Hand grips pointer"], "status": "completed"},
    )
    row.scores["astra_judge"] = {
        "status": "ok",
        "generation_id": row.id,
        "bufo_present": True,
        "similarity": 0.8,
        "anatomy": 0.9,
        "composition": 0.6,
        "readability": 0.9,
        "requirement_coverage": 0.5,
        "scorer_configuration": copy.deepcopy(CONFIG) | {"calibration_status": "calibrated"},
    }
    for key, value in overrides.items():
        if key in {"valid", "error"}:
            setattr(row, key, value)
        elif key == "metadata":
            row.benchmark_metadata = value
        else:
            row.scores["astra_judge"][key] = value
    return row


def test_reward_is_similarity_times_coverage_times_craft_mean():
    expected = 0.8 * 0.5 * ((0.9 + 0.6 + 0.9) / 3)
    assert abs(astra_reward(astra_row(), scorer_configuration=CONFIG) - expected) < 1e-12


def test_configuration_compared_through_config_key():
    # calibration_status differs between row and CONFIG yet still matches
    assert astra_reward(astra_row(), scorer_configuration=CONFIG) > 0
    other = CONFIG | {"threshold": 0.5}
    assert astra_reward(astra_row(), scorer_configuration=other) == 0.0


def test_zero_cases():
    for row in (
        astra_row(valid=False),
        astra_row(error="boom"),
        astra_row(status="judge_error"),
        astra_row(bufo_present=False),
        astra_row(generation_id="other"),
        astra_row(metadata={"requirements": ["x"], "status": "failed"}),
        astra_row(similarity=None),
    ):
        assert astra_reward(row, scorer_configuration=CONFIG) == 0.0


def test_missing_score_is_zero():
    row = astra_row()
    row.scores.pop("astra_judge")
    assert astra_reward(row, scorer_configuration=CONFIG) == 0.0
