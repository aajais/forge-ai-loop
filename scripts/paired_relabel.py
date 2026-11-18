"""Pair an arm against the same-named arm of another run by relabelling the right side in memory.

``paired_across_runs.paired_across`` needs two distinct arm names; a replication or a before/after
comparison has the same name on both sides. This relabels the right arm's rows to ``--right-alias``
and then uses ``paired_across`` unchanged.
"""

import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "paired_across_runs", Path(__file__).with_name("paired_across_runs.py")
)
across = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(across)


def paired_relabel(left_dir: Path, left_arm: str, right_dir: Path, right_arm: str, alias: str):
    original = across.load_arm

    def load_arm(astra_dir, arm):
        if Path(astra_dir) == Path(right_dir) and arm == alias:
            scorer, rows = original(astra_dir, right_arm)
            for row in rows:
                row.backend = alias
            return scorer, rows
        return original(astra_dir, arm)

    across.load_arm = load_arm
    try:
        report = across.paired_across(Path(left_dir), left_arm, Path(right_dir), alias)
    finally:
        across.load_arm = original
    report["right"]["arm"] = f"{right_arm} (relabelled {alias})"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--left-arm", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--right-arm", required=True)
    parser.add_argument("--right-alias", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    alias = args.right_alias or f"{args.right_arm}_right"
    report = paired_relabel(args.left, args.left_arm, args.right, args.right_arm, alias)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["paired"], indent=2))


if __name__ == "__main__":
    main()
