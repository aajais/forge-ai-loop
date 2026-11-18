"""Step 5 selection record for the Stage B SFT sweep (reads local sweep outputs only)."""

import argparse
import json
from pathlib import Path

RULE = (
    "highest validation strict pass; ties by coverage, fewer errors, fewer output tokens; "
    "selectable only if paired 95% lower bound over its base arm > 0"
)
NO_WINNER = "no candidate has a paired lower bound over its base arm > 0"
J4 = {
    "run": False,
    "reason": (
        "controller ruling T7-J4: selection rule cannot be met at full data; half-data row would "
        "be uninformative at the floor; reversible"
    ),
}


def candidate(sweep_dir, arm, checkpoints, distinctness, base_arm="open_base"):
    summary = json.loads((sweep_dir / arm / "astra-v5-k3/summary.json").read_text())["arms"][arm]
    paired = json.loads((sweep_dir / arm / f"paired-{arm}-vs-{base_arm}.json").read_text())
    gemma_file = sweep_dir / arm / f"paired-{arm}-vs-open_base.json"
    gemma = json.loads(gemma_file.read_text()) if gemma_file.exists() else None
    vs_sol = json.loads((sweep_dir / arm / f"paired-{arm}-vs-closed_quality_bar.json").read_text())
    entry = next(e for e in checkpoints if e["arm_name"] == arm)
    interval = lambda report: {  # noqa: E731
        "estimate": report["paired"]["difference"],
        "lower": report["paired"]["ci95"][0],
        "upper": report["paired"]["ci95"][1],
    }
    return {
        "arm": arm,
        "strict_pass": summary["strict_passes"],
        "n": summary["total"],
        "coverage": summary["requirement_coverage_mean"],
        "errors": summary["judge_errors"] + summary["other_failures"],
        "output_tokens_mean": summary["output_tokens_mean"],
        "base_arm": base_arm,
        "paired_vs_base_arm": interval(paired),
        "paired_vs_open_base": interval(gemma) if gemma else None,
        "paired_vs_sol": interval(vs_sol),
        "artifact": entry["artifact"],
        "forge_job_id": entry["forge_job_id"],
        "distinctness_verdict": (distinctness or {}).get("verdict", {}).get(arm),
    }


def select(candidates):
    eligible = [c for c in candidates if c["paired_vs_base_arm"]["lower"] > 0]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda c: (
            -c["strict_pass"],
            -c["coverage"],
            c["errors"],
            c["output_tokens_mean"],
        ),
    )


def build(sweep_dir, arms, checkpoints_path, distinctness_path, base_arms=None):
    sweep_dir = Path(sweep_dir)
    checkpoints = json.loads(Path(checkpoints_path).read_text())
    distinctness_path = Path(distinctness_path)
    distinctness = json.loads(distinctness_path.read_text()) if distinctness_path.exists() else None
    base_arms = base_arms or {}
    candidates = [
        candidate(sweep_dir, arm, checkpoints, distinctness, base_arms.get(arm, "open_base"))
        for arm in arms
    ]
    winner = select(candidates)
    return {
        "schema_version": 1,
        "rule": RULE,
        "candidates": candidates,
        "selected": winner["arm"] if winner else None,
        "artifact": winner["artifact"] if winner else None,
        "forge_job_id": winner["forge_job_id"] if winner else None,
        "reason": (
            f"selected {winner['arm']}: highest eligible validation strict pass"
            if winner
            else NO_WINNER
            + " ("
            + ", ".join(f"{c['arm']} {c['strict_pass']}/{c['n']}" for c in candidates)
            + " strict pass)"
        ),
        "j4": J4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("runs/distill/stage-b/sweep"))
    parser.add_argument("--arm", action="append", dest="arms")
    parser.add_argument("--base-arm", action="append", default=[], help="ARM=BASE_ARM")
    parser.add_argument(
        "--checkpoints", type=Path, default=Path("runs/distill/stage-b/checkpoints.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/distill/stage-b/sweep/selection.json")
    )
    args = parser.parse_args()
    report = build(
        args.sweep_dir,
        args.arms or ["sft_j1", "sft_j3"],
        args.checkpoints,
        args.sweep_dir / "distinctness.json",
        dict(item.split("=", 1) for item in args.base_arm),
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": report["selected"], "reason": report["reason"]}))


if __name__ == "__main__":
    main()
