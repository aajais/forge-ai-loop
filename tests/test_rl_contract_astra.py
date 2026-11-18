import copy

import pytest

from bufogen.training.rl_contract import build_astra_calibration_report, require_astra_calibration

CONFIG = {"rubric_version": "bufo-astra-v5", "samples": 3, "threshold": 0.65}


def scored(gid, strict, requirements=("a", "b"), split="validation"):
    return {
        "id": gid,
        "valid": True,
        "error": None,
        "benchmark_metadata": {
            "status": "completed",
            "split": split,
            "requirements": list(requirements),
        },
        "scores": {
            "astra_judge": {
                "status": "ok",
                "generation_id": gid,
                "strict_pass": strict,
                "judge_confidence": 0.9,
                "vs_reference": "equal",
                "scorer_configuration": CONFIG | {"calibration_status": "calibrated"},
                "prompt_checks": [{"requirement": r, "satisfied": strict} for r in requirements],
            }
        },
    }


def decision(gid, index, passed, missing=()):
    return {
        "index": index,
        "generation_id": gid,
        "pass": passed,
        "vs_reference": "equal" if passed else "worse",
        "missing": list(missing),
    }


def fixture():
    rows = [
        scored("g0", True),
        scored("g1", False),
        scored("g2", False),
        scored("g3", True),
        scored("g4", True),
    ]
    decisions = [
        decision("g0", 0, True),
        decision("g1", 1, False, [0, 1]),
        decision("g2", 2, True),  # judge fail, reviewer pass
        decision("g3", 3, True),
        decision("g4", 4, True),
    ]
    return rows, decisions


def test_report_shape_and_counts():
    rows, decisions = fixture()
    report = build_astra_calibration_report(rows, decisions, scorer_configuration=CONFIG)
    assert report["rl_ready"] is True
    assert report["agreement"] == 0.8
    assert report["positive_count"] == 4 and report["negative_count"] == 1
    assert report["confusion"] == {
        "judge_pass_reviewer_pass": 3,
        "judge_pass_reviewer_fail": 0,
        "judge_fail_reviewer_pass": 1,
        "judge_fail_reviewer_fail": 1,
    }
    assert report["judge_pass_reviewer_missing"] == 0
    assert len(report["observations"]) == 5


def test_judge_pass_with_reviewer_missing_blocks():
    rows, decisions = fixture()
    decisions[3] = decision("g3", 3, False, [1])
    report = build_astra_calibration_report(rows, decisions, scorer_configuration=CONFIG)
    assert report["judge_pass_reviewer_missing"] == 1
    assert report["rl_ready"] is False


def test_needs_positive_and_negative_and_agreement():
    rows, decisions = fixture()
    all_pass = [decision(d["generation_id"], d["index"], True) for d in decisions]
    assert not build_astra_calibration_report(rows, all_pass, scorer_configuration=CONFIG)[
        "rl_ready"
    ]
    low = [
        decision("g0", 0, False, [0]),
        decision("g1", 1, True),
        decision("g2", 2, True),
        decision("g3", 3, False, [0]),
        decision("g4", 4, True),
    ]
    report = build_astra_calibration_report(rows, low, scorer_configuration=CONFIG)
    assert report["agreement"] < 0.8 and not report["rl_ready"]


def test_unknown_generation_or_wrong_configuration_rejected():
    rows, decisions = fixture()
    with pytest.raises(ValueError):
        build_astra_calibration_report(rows[:4], decisions, scorer_configuration=CONFIG)
    with pytest.raises(ValueError):
        build_astra_calibration_report(
            rows, decisions, scorer_configuration=CONFIG | {"threshold": 0.1}
        )


def test_require_accepts_good_report_and_rejects_tamper():
    rows, decisions = fixture()
    report = build_astra_calibration_report(rows, decisions, scorer_configuration=CONFIG)
    assert require_astra_calibration(report, scorer_configuration=CONFIG)["rl_ready"]
    assert require_astra_calibration(
        report, scorer_configuration=CONFIG | {"calibration_status": "x"}
    )["rl_ready"]
    with pytest.raises(ValueError):
        require_astra_calibration(report, scorer_configuration=CONFIG | {"threshold": 0.1})

    bad_rows, bad_decisions = fixture()
    bad_decisions[3] = decision("g3", 3, False, [1])
    blocked = build_astra_calibration_report(bad_rows, bad_decisions, scorer_configuration=CONFIG)
    forged = copy.deepcopy(blocked)
    forged["rl_ready"] = True
    with pytest.raises(ValueError):
        require_astra_calibration(forged, scorer_configuration=CONFIG)

    edited = copy.deepcopy(report)
    edited["observations"][3]["decision"]["pass"] = False
    with pytest.raises(ValueError):
        require_astra_calibration(edited, scorer_configuration=CONFIG)


@pytest.mark.parametrize("split", ["heldout", "test", None])
def test_non_calibration_splits_are_refused(split):
    rows, decisions = fixture()
    rows[0] = scored("g0", True, split=split)
    with pytest.raises(ValueError, match="split"):
        build_astra_calibration_report(rows, decisions, scorer_configuration=CONFIG)


def test_require_refuses_a_report_with_a_heldout_observation():
    rows, decisions = fixture()
    report = build_astra_calibration_report(rows, decisions, scorer_configuration=CONFIG)
    forged = copy.deepcopy(report)
    forged["observations"][0]["generation"]["benchmark_metadata"]["split"] = "heldout"
    with pytest.raises(ValueError, match="split"):
        require_astra_calibration(forged, scorer_configuration=CONFIG)


def test_judge_errors_are_not_negatives_and_block_readiness():
    rows, decisions = fixture()
    rows[1]["scores"]["astra_judge"]["status"] = "judge_error"
    report = build_astra_calibration_report(rows, decisions, scorer_configuration=CONFIG)
    assert report["judge_errors"] == 1 and report["rl_ready"] is False
    assert report["negative_count"] == 0  # g1 (reviewer fail) is an error, not a judged negative
    assert sum(report["confusion"].values()) == 4
    assert report["readiness_policy"]["maximum_judge_errors"] == 0
    with pytest.raises(ValueError):
        require_astra_calibration(report, scorer_configuration=CONFIG)
