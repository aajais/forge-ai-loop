"""Write Astra judge verdicts from a scored directory as training-review JSONL."""

import argparse
import json
from pathlib import Path

from bufogen.curation.judge_reviews import write_decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-dir", type=Path, required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = write_decisions(args.scored_dir, args.out, registry_key=(args.rubric, args.samples))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
