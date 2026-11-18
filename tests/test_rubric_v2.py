import json

import pytest

from bufogen.evaluation.scorers.astra_judge import (
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_V2,
    RUBRIC_V1,
    RUBRIC_V2,
    RUBRIC_V3,
    RUBRIC_V4,
    RUBRIC_V5,
    RUBRICS,
    AstraReferenceJudge,
    astra_result,
    judge_failure,
    schema_name,
)


def judgment(reqs, **overrides):
    checks = [
        {"requirement": r, "satisfied": True, "directly_visible": True, "evidence": "seen"}
        for r in reqs
    ]
    base = {
        "bufo_present": True,
        "similarity": 0.9,
        "anatomy": 0.9,
        "composition": 0.9,
        "readability": 0.9,
        "reason": "ok",
        "prompt_checks": checks,
        "vs_reference": "equal",
        "vs_reference_reason": "same",
        "judge_confidence": 0.9,
    }
    return base | overrides


def test_v1_rubric_and_schema_are_unchanged_by_v2():
    assert RUBRICS["bufo-astra-v1"] is RUBRIC_V1
    assert RUBRIC_V2.startswith(RUBRIC_V1)
    assert "directly_visible" not in json.dumps(RESPONSE_SCHEMA)
    v1 = AstraReferenceJudge().configuration()
    v2 = AstraReferenceJudge(rubric_version="bufo-astra-v2").configuration()
    assert v1["rubric_version"] == "bufo-astra-v1" and v2["rubric_version"] == "bufo-astra-v2"
    assert v1["rubric_hash"] != v2["rubric_hash"] and v1["schema_hash"] != v2["schema_hash"]
    assert set(v1) == set(v2)


def test_v2_schema_requires_directly_visible():
    item = RESPONSE_SCHEMA_V2["properties"]["prompt_checks"]["items"]
    assert "directly_visible" in item["required"]
    assert item["properties"]["directly_visible"]["type"] == "boolean"
    assert schema_name("bufo-astra-v2") == "bufo_astra_v2"
    with pytest.raises(ValueError):
        schema_name("bufo-astra-v9")


def test_v2_rubric_states_the_visibility_rule():
    for phrase in (
        "VISIBILITY RULE (binding)",
        "directly_visible=false is NOT satisfied",
        "An eye showing a pupil, iris or highlight is not",
        "a stem whose line breaks at a rim",
        "A hand hidden behind the prop",
    ):
        assert phrase in RUBRIC_V2
        assert phrase not in RUBRIC_V1


def test_v2_overrides_satisfied_when_not_directly_visible():
    data = judgment(["a", "b"])
    data["prompt_checks"][1]["directly_visible"] = False  # judge said satisfied but inferred
    r = astra_result(
        data, ["a", "b"], threshold=0.65, confidence_floor=0.6, rubric_version="bufo-astra-v2"
    )
    assert r["prompt_checks"][1]["satisfied"] is False
    assert r["visibility_overrides"] == ["b"]
    assert "prompt_mismatch" in r["failure_modes"] and r["quality_failure"]
    assert r["requirement_coverage"] == 0.5 and r["rubric_version"] == "bufo-astra-v2"
    assert data["prompt_checks"][1]["satisfied"] is True  # input not mutated


def test_v2_requires_the_flag_and_v1_ignores_it():
    data = judgment(["a"])
    del data["prompt_checks"][0]["directly_visible"]
    with pytest.raises(ValueError):
        astra_result(
            data, ["a"], threshold=0.65, confidence_floor=0.6, rubric_version="bufo-astra-v2"
        )
    r = astra_result(data, ["a"], threshold=0.65, confidence_floor=0.6)
    assert r["rubric_version"] == "bufo-astra-v1" and r["visibility_overrides"] == []
    with pytest.raises(ValueError):
        astra_result(data, ["a"], threshold=0.65, confidence_floor=0.6, rubric_version="nope")


def test_unknown_rubric_rejected_and_failures_carry_version():
    with pytest.raises(ValueError):
        AstraReferenceJudge(rubric_version="bufo-astra-v9")
    assert judge_failure("missing_image", rubric_version="bufo-astra-v2")["rubric_version"] == (
        "bufo-astra-v2"
    )
    assert judge_failure("missing_image")["rubric_version"] == "bufo-astra-v1"


def test_v3_keeps_visibility_rule_adds_eye_state_and_drops_vs_reference_sentence():
    assert RUBRIC_V3.startswith(RUBRIC_V1)
    assert "VISIBILITY RULE (binding)" in RUBRIC_V3
    assert "A hand hidden behind the prop" in RUBRIC_V3
    assert "Heavy-lidded, half-closed, squinting or drooping eyes are" in RUBRIC_V3
    assert "If one eye is closed and the other is not" in RUBRIC_V3
    assert "vs_reference: a candidate whose requirements are met less clearly" in RUBRIC_V2
    assert "vs_reference: a candidate whose requirements are met less clearly" not in RUBRIC_V3
    assert "\n\n" not in RUBRIC_V3.replace(RUBRIC_V1, "")  # no blank line left by the splice
    v3 = AstraReferenceJudge(rubric_version="bufo-astra-v3").configuration()
    v2 = AstraReferenceJudge(rubric_version="bufo-astra-v2").configuration()
    assert v3["schema_hash"] == v2["schema_hash"] and v3["rubric_hash"] != v2["rubric_hash"]
    assert schema_name("bufo-astra-v3") == "bufo_astra_v3"
    r = astra_result(
        judgment(["a"]), ["a"], threshold=0.65, confidence_floor=0.6, rubric_version="bufo-astra-v3"
    )
    assert r["rubric_version"] == "bufo-astra-v3"


def test_merge_judgments_is_conservative():
    from bufogen.evaluation.scorers.astra_judge import merge_judgments

    a = judgment(["x", "y"], similarity=1.0, judge_confidence=1.0, vs_reference="equal")
    b = judgment(["x", "y"], similarity=0.5, judge_confidence=0.5, vs_reference="worse")
    b["prompt_checks"][1]["satisfied"] = False
    b["prompt_checks"][1]["evidence"] = "no hand"
    merged = merge_judgments([a, b], ["x", "y"])
    assert merged["bufo_present"] is True
    assert merged["similarity"] == pytest.approx(0.75)
    assert merged["judge_confidence"] == pytest.approx(0.75)
    assert merged["vs_reference"] == "worse" and merged["vs_reference_reason"] == "same"
    x, y = merged["prompt_checks"]
    assert x["satisfied"] is True and x["votes"] == "2/2" and x["directly_visible"] is True
    assert y["satisfied"] is False and y["votes"] == "1/2" and "sample 2: no hand" in y["evidence"]
    assert merge_judgments([a], ["x", "y"]) is a
    with pytest.raises(ValueError):
        merge_judgments([a, judgment(["x"])], ["x", "y"])
    r = astra_result(
        merged, ["x", "y"], threshold=0.65, confidence_floor=0.6, rubric_version="bufo-astra-v3"
    )
    assert "prompt_mismatch" in r["failure_modes"] and r["requirement_coverage"] == 0.5


def test_samples_recorded_only_when_more_than_one():
    one = AstraReferenceJudge(rubric_version="bufo-astra-v3").configuration()
    three = AstraReferenceJudge(rubric_version="bufo-astra-v3", samples=3).configuration()
    assert "samples" not in one and three["samples"] == 3
    assert {k: v for k, v in three.items() if k != "samples"} == one
    with pytest.raises(ValueError):
        AstraReferenceJudge(samples=6)


def test_v4_adds_grip_definition_on_top_of_v3():
    assert RUBRIC_V4.startswith(RUBRIC_V3)
    assert "If you cannot describe the fingers, there is no grip." in RUBRIC_V4
    assert "a rounded bump at the\nobject's edge with no fingers" in RUBRIC_V4
    assert "there is no grip" not in RUBRIC_V3
    assert "\n\n" not in RUBRIC_V4.replace(RUBRIC_V1, "")
    v4 = AstraReferenceJudge(rubric_version="bufo-astra-v4").configuration()
    v3 = AstraReferenceJudge(rubric_version="bufo-astra-v3").configuration()
    assert v4["schema_hash"] == v3["schema_hash"] and v4["rubric_hash"] != v3["rubric_hash"]
    assert schema_name("bufo-astra-v4") == "bufo_astra_v4"


def test_v5_accepts_mitt_hands_but_keeps_hidden_hand_disqualifiers():
    assert RUBRIC_V5.startswith(RUBRIC_V3) and not RUBRIC_V5.startswith(RUBRIC_V4)
    assert "fingers or a rounded mitt" in RUBRIC_V5
    assert "finger detail is not required" in RUBRIC_V5
    assert "If you cannot describe the fingers" not in RUBRIC_V5
    assert "a hand mostly behind the object" in RUBRIC_V5
    assert "bump or sliver at the object's edge" in RUBRIC_V5
    assert "\n\n" not in RUBRIC_V5.replace(RUBRIC_V1, "")
    assert schema_name("bufo-astra-v5") == "bufo_astra_v5"
    assert sorted(RUBRICS) == [f"bufo-astra-v{i}" for i in range(1, 6)]
