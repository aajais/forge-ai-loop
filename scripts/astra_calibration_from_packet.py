"""Build the Astra RL calibration report from reviewed packet decisions (no API calls)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bufogen.evaluation.scorers.astra_judge import config_key  # noqa: E402
from bufogen.training.rl_contract import build_astra_calibration_report  # noqa: E402

CAL = ROOT / "runs/distill/calibration-v1"
ARMS = [
    ROOT / "runs/distill/arms-v1/astra-v5-k3",  # closed_quality_bar items
    ROOT / "runs/distill/arms-v1-students/astra-v5-k3",  # open_base and iter2_e3 items
]


def build(items: list[dict], key: list[dict], decisions: dict, rows: list[dict], config: dict):
    if decisions.get("packet_sha256") != key[0].get("packet_sha256"):
        raise ValueError("Decisions and unblinding key come from different packets")
    by_index = {k["index"]: k for k in key}
    joined = []
    for decision in decisions["decisions"]:
        index = decision["index"]
        if index not in by_index or index >= len(items):
            raise ValueError(f"Decision index {index} is not in the packet")
        joined.append(decision | {"generation_id": by_index[index]["generation_id"]})
    by_id = {row["id"]: row for row in rows}
    for decision in joined:  # the packet's requirement list must be the scored row's list
        row = by_id.get(decision["generation_id"])
        if row is None:
            raise ValueError(f"No scored row for {decision['generation_id']}")
        if row["benchmark_metadata"]["requirements"] != items[decision["index"]]["requirements"]:
            raise ValueError(f"Requirement lists differ for item {decision['index']}")
    return build_astra_calibration_report(rows, joined, scorer_configuration=config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=CAL / "packet-v4/items.json")
    parser.add_argument("--key", type=Path, default=CAL / "key-v4/unblinding-key.json")
    parser.add_argument("--decisions", type=Path, default=CAL / "decisions-user-v4.json")
    parser.add_argument(
        "--scored", type=Path, nargs="+", default=[a / "scored.jsonl" for a in ARMS]
    )
    parser.add_argument("--scorer", type=Path, nargs="+", default=[a / "scorer.json" for a in ARMS])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/distill/stage-b/rl/astra-calibration.json"
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for path in args.scored
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    configs = [json.loads(path.read_text())["judge"] for path in args.scorer]
    if any(config_key(c) != config_key(configs[0]) for c in configs):
        raise ValueError("Scored arms were judged with different scorer configurations")
    report = build(
        json.loads(args.packet.read_text()),
        json.loads(args.key.read_text()),
        json.loads(args.decisions.read_text()),
        rows,
        configs[0],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "rl_ready",
                    "agreement",
                    "confusion",
                    "positive_count",
                    "negative_count",
                    "judge_pass_reviewer_missing",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
