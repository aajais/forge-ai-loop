"""Build original-emoji tasks for ART serverless RL; publishing is optional."""

import argparse
import asyncio
import json
from pathlib import Path

from bufogen.curation.rl_dataset import build_rl_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/emoji-rl-v1"))
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--link-registry",
        action="store_true",
        help="Also link the published artifact into WANDB_REGISTRY",
    )
    args = parser.parse_args()
    if args.link_registry and not args.publish:
        parser.error("--link-registry requires --publish")
    summary = asyncio.run(
        build_rl_dataset(args.output, limit=args.limit, concurrency=args.concurrency)
    )
    print(json.dumps(summary, indent=2))
    if args.publish:
        from bufogen.integrations.registry import RegistryClient

        print(
            RegistryClient().push_dataset(
                args.output,
                name="bufo-emoji-rl-dataset",
                metadata=summary,
                link_registry=args.link_registry,
            )
        )


if __name__ == "__main__":
    main()
