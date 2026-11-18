"""Weave annotation queue for the blinded review packet.

create: one neutral call per packet item (label, prompt, numbered requirements, A, B),
        three annotation fields (missing, pass, vs_reference), one queue. The record
        with the call-to-item map goes to --key-dir, never into the packet.
pull:   read the reviewer's annotations back into a bufo-review-v2 decisions file for
        scripts/review_agreement.py (step7).
"""

import argparse
import json
from pathlib import Path

from bufogen.curation.review_queue import create_queue, pull_decisions


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--packet", type=Path, required=True)
    c.add_argument("--key-dir", type=Path, required=True)
    c.add_argument("--name", required=True, help="queue name shown in Weave")
    p = sub.add_parser("pull")
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--key-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="decisions.json to write")
    args = parser.parse_args(argv)
    if args.command == "create":
        result = create_queue(args.packet.resolve(), args.key_dir.resolve(), name=args.name)
        print(json.dumps({k: v for k, v in result.items() if k != "calls"}, indent=2))
    else:
        print(json.dumps(pull_decisions(args.packet.resolve(), args.key_dir.resolve(), args.out)))


if __name__ == "__main__":
    main()
