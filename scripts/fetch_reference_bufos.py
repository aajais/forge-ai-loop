import argparse
import asyncio
from pathlib import Path

from bufogen.curation.backup import fetch_reference_bufos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download static public bufos; optionally intersect a Slack emoji-name export."
    )
    parser.add_argument("--names", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    for path in asyncio.run(fetch_reference_bufos(limit=args.limit, names_file=args.names)):
        print(path)
