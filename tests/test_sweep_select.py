import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import sweep_select as ss  # noqa: E402


def fixture(tmp_path, arms):
    """arms: {label: (strict_pass, lower_bound, coverage)}"""
    checkpoints = []
    for arm, (passes, lower, coverage) in arms.items():
        d = tmp_path / arm
        (d / "astra-v5-k3").mkdir(parents=True)
        summary = {
            "arms": {
                arm: {
                    "strict_passes": passes,
                    "total": 48,
                    "requirement_coverage_mean": coverage,
                    "judge_errors": 0,
                    "other_failures": 0,
                    "output_tokens_mean": 1000.0,
                }
            }
        }
        (d / "astra-v5-k3/summary.json").write_text(json.dumps(summary))
        paired = {"paired": {"difference": lower, "ci95": [lower, lower + 0.1]}}
        (d / f"paired-{arm}-vs-open_base.json").write_text(json.dumps(paired))
        (d / f"paired-{arm}-vs-closed_quality_bar.json").write_text(json.dumps(paired))
        checkpoints.append({"arm_name": arm, "artifact": f"art-{arm}", "forge_job_id": f"j-{arm}"})
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps(checkpoints))
    return path


def test_no_winner_when_no_lower_bound_clears_zero(tmp_path):
    path = fixture(tmp_path, {"sft_j1": (0, 0.0, 0.41), "sft_j3": (0, 0.0, 0.45)})
    report = ss.build(tmp_path, ["sft_j1", "sft_j3"], path, tmp_path / "none.json")
    assert report["selected"] is None and report["schema_version"] == 1
    assert "no candidate has a paired lower bound" in report["reason"]
    assert report["j4"]["run"] is False
    assert [c["arm"] for c in report["candidates"]] == ["sft_j1", "sft_j3"]


def test_selects_best_eligible_candidate_by_strict_pass_then_coverage(tmp_path):
    path = fixture(
        tmp_path,
        {"sft_j1": (9, 0.05, 0.5), "sft_j3": (7, 0.1, 0.6), "sft_j4": (12, -0.01, 0.9)},
    )
    report = ss.build(tmp_path, ["sft_j1", "sft_j3", "sft_j4"], path, tmp_path / "none.json")
    assert report["selected"] == "sft_j1"  # sft_j4 has more passes but fails the lower bound
    assert report["artifact"] == "art-sft_j1" and report["forge_job_id"] == "j-sft_j1"


def test_each_candidate_is_compared_with_its_own_base_arm(tmp_path):
    path = fixture(tmp_path, {"sft_j1": (0, 0.0, 0.41), "sft_q3": (5, 0.04, 0.6)})
    # sft_q3's paired file is against qwen_base, not open_base
    d = tmp_path / "sft_q3"
    (d / "paired-sft_q3-vs-open_base.json").rename(d / "paired-sft_q3-vs-qwen_base.json")
    report = ss.build(
        tmp_path,
        ["sft_j1", "sft_q3"],
        path,
        tmp_path / "none.json",
        base_arms={"sft_q3": "qwen_base"},
    )
    by_arm = {c["arm"]: c for c in report["candidates"]}
    assert by_arm["sft_j1"]["base_arm"] == "open_base"
    assert by_arm["sft_q3"]["base_arm"] == "qwen_base"
    assert by_arm["sft_q3"]["paired_vs_base_arm"]["lower"] == 0.04
    assert report["selected"] == "sft_q3"
    assert "its base arm" in report["rule"]
