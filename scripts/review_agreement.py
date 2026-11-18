"""Agreement between one reviewer's decisions.json and the Astra judge on a review packet.

Writes agreement-<reviewer>.json beside the unblinding key and prints the gate and the
aggregate agreement only (per-item detail stays in the written report).
"""

import argparse
import json
from pathlib import Path

from bufogen.curation.review_packet import agreement_report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="packet directory")
    parser.add_argument("--key", type=Path, required=True, help="unblinding-key.json")
    parser.add_argument("--decisions", type=Path, required=True, help="downloaded decisions")
    parser.add_argument("--reviewer", required=True, help="label, [A-Za-z0-9_-]+")
    args = parser.parse_args(argv)
    report = agreement_report(
        args.packet, args.decisions, reviewer=args.reviewer, key_path=args.key
    )
    summary = {
        "reviewer": report["reviewer"],
        "n": report["n"],
        "strict_pass_agreement": report["strict_pass_agreement"],
        "gate": report["gate"],
        "report": str(args.key.parent / f"agreement-{args.reviewer}.json"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
