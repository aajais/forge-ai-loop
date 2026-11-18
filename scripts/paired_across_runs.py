"""Paired family-cluster interval between arms scored in two different run directories.

Both ``astra/`` directories must record the same judge configuration and the same reference
manifest digest, the two arms must cover the same prompt ids, and each paired prompt id must
carry the same prompt text and requirements; otherwise the comparison is refused. Writes a
JSON report and prints it.
"""

import argparse
import hashlib
import json
from pathlib import Path

from bufogen.evaluation.scorers.astra_judge import config_key
from bufogen.evaluation.statistics import astra_pass, paired_family_interval
from bufogen.generation import BufoGeneration
from bufogen.settings import ROOT


def load_arm(astra_dir: Path, arm: str) -> tuple[dict, list[BufoGeneration]]:
    scorer = json.loads((astra_dir / "scorer.json").read_text())
    rows = [
        BufoGeneration(**json.loads(line))
        for line in (astra_dir / "scored.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if r.backend == arm]
    if not rows:
        raise SystemExit(f"No rows for arm {arm!r} in {astra_dir}")
    return scorer, rows


def _same_reference_drawings(left_refs, right_refs, prompt_ids) -> bool:
    """Different manifests (e.g. a superset) are comparable only if every paired prompt is judged
    against the very same reference drawing (PNG digest) in both."""
    try:
        rows = []
        for refs in (left_refs, right_refs):
            data = (ROOT / refs["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != refs["sha256"]:
                return False  # the manifest on disk is not the one the run was judged against
            rows.append(json.loads(data)["rows"])
        return bool(prompt_ids) and all(
            rows[0][pid]["png_sha256"] == rows[1][pid]["png_sha256"] for pid in prompt_ids
        )
    except (KeyError, TypeError, OSError, ValueError):
        return False


def paired_across(left_dir: Path, left_arm: str, right_dir: Path, right_arm: str) -> dict:
    left_scorer, left_rows = load_arm(left_dir, left_arm)
    right_scorer, right_rows = load_arm(right_dir, right_arm)
    if config_key(left_scorer["judge"]) != config_key(right_scorer["judge"]):
        raise SystemExit("Judge configuration differs between the two directories")
    prompt_ids = {r.prompt_id for r in left_rows} & {r.prompt_id for r in right_rows}
    per_prompt_rule = left_scorer["references"] != right_scorer["references"]
    if per_prompt_rule and not _same_reference_drawings(
        left_scorer["references"], right_scorer["references"], prompt_ids
    ):
        raise SystemExit("Reference manifest differs between the two directories")
    if left_scorer["self_reference"] or right_scorer["self_reference"]:
        raise SystemExit("Self-reference scoring cannot be paired against another arm")
    prompt_ids = {r.prompt_id for r in left_rows}
    right_rows = [r for r in right_rows if r.prompt_id in prompt_ids]
    # The same prompt id must mean the same request in both runs.
    left_by_id = {r.prompt_id: r for r in left_rows}
    for r in right_rows:
        other = left_by_id[r.prompt_id]
        if r.prompt != other.prompt or (r.benchmark_metadata or {}).get("requirements") != (
            other.benchmark_metadata or {}
        ).get("requirements"):
            raise SystemExit(
                f"Prompt {r.prompt_id!r} differs between the two directories "
                "(prompt text or requirements)"
            )
    interval = paired_family_interval(
        left_rows + right_rows, left_arm, right_arm, pass_fn=astra_pass
    )
    return {
        "left": {
            "dir": str(left_dir),
            "arm": left_arm,
            "strict_pass": sum(map(astra_pass, left_rows)),
        },
        "right": {
            "dir": str(right_dir),
            "arm": right_arm,
            "strict_pass": sum(map(astra_pass, right_rows)),
        },
        "judge": left_scorer["judge"],
        "references": left_scorer["references"],
        "references_left": left_scorer["references"],
        "references_right": right_scorer["references"],
        "per_prompt_reference_digest_rule": per_prompt_rule,
        "paired": interval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="astra/ directory holding the left arm")
    parser.add_argument("--left-arm", required=True)
    parser.add_argument("--right", required=True, help="astra/ directory holding the right arm")
    parser.add_argument("--right-arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = paired_across(Path(args.left), args.left_arm, Path(args.right), args.right_arm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["paired"], indent=2))


if __name__ == "__main__":
    main()
