"""Build the Stage A blinded review packet from validation rows of the arm runs.

Several --runs directories (one per confirmation arm) are mixed into one packet.
The unblinding key goes to --key-dir, which must not be inside --out: the reviewer
opens --out and must not see which arm drew which item.
"""

import argparse
import json
from pathlib import Path

from bufogen.curation.review_packet import (
    CONFIRMATION_SPLITS,
    create_packet,
    load_scored_runs,
    select_review_rows,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("runs/distill/arms-v1", "runs/distill/arms-v1-students")


def build(
    out: Path,
    key_dir: Path,
    seed: int,
    runs=RUNS,
    exclude_ids=frozenset(),
    scored_dir="astra",
    split="validation",
    confirmation=False,
    size=40,
) -> dict:
    if split in CONFIRMATION_SPLITS and not confirmation:
        raise ValueError(f"split {split!r} is reserved for the confirmation run (--confirmation)")
    rows = [
        r
        for r in load_scored_runs([ROOT / run for run in runs], scored_dir)
        if r.benchmark_metadata.get("split") == split
    ]
    chosen = select_review_rows(
        rows, size=size, seed=seed, confirmation=confirmation, exclude_ids=exclude_ids
    )
    return create_packet(chosen, out, seed=seed, key_dir=key_dir)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="packet directory to create")
    parser.add_argument(
        "--key-dir", type=Path, required=True, help="directory for unblinding-key.json"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--exclude-ids-file",
        type=Path,
        action="append",
        default=[],
        help="one generation id per line; repeatable; listed generations never enter the sample",
    )
    parser.add_argument(
        "--scored-dir",
        default="astra",
        help="judge run directory under each run (astra, astra-v4-k3, ...) whose verdicts go in the key",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        action="append",
        default=[],
        help="run directory holding <scored-dir>/scored.jsonl; repeatable, one per arm "
        f"(default for validation: {', '.join(RUNS)})",
    )
    parser.add_argument("--split", choices=("validation", "heldout"), default="validation")
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="required for --split heldout; the single confirmation run only",
    )
    parser.add_argument("--size", type=int, default=40)
    args = parser.parse_args(argv)
    # Refuse before any scored row is read.
    if args.split == "heldout" and not args.confirmation:
        parser.error("--split heldout requires --confirmation (reserved for the confirmation run)")
    if args.confirmation and args.split != "heldout":
        parser.error("--confirmation applies only to --split heldout")
    if args.split == "heldout" and not args.runs:
        parser.error("--split heldout requires at least one --runs directory")
    exclude = frozenset(
        line.strip()
        for path in args.exclude_ids_file
        for line in path.read_text().splitlines()
        if line.strip()
    )
    result = build(
        args.out.resolve(),
        args.key_dir.resolve(),
        args.seed,
        runs=tuple(args.runs) or RUNS,
        exclude_ids=exclude,
        scored_dir=args.scored_dir,
        split=args.split,
        confirmation=args.confirmation,
        size=args.size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
