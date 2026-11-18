"""Live student generation, sandbox render, and target-image judging; no optimizer step."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from bufogen.agents.agent import BufoAgent
from bufogen.curation.rl_dataset import reference_path
from bufogen.evaluation.scorers.svg_validity import SvgValidity
from bufogen.evaluation.scorers.vlm_judge import VLMJudge
from bufogen.generation import Manifest
from bufogen.models import load_backends
from bufogen.rendering.renderer import Renderer


async def check(dataset: Path, output: Path, *, limit=1):
    splits = {
        split: [json.loads(line) for line in (dataset / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "heldout")
    }
    for key in ("prompt_id", "pixel_sha256", "split_group"):
        if {r[key] for r in splits["train"]} & {r[key] for r in splits["heldout"]}:
            raise ValueError(f"Train/held-out overlap in {key}.")
    for rows in splits.values():
        for row in rows:
            path = reference_path(dataset, row)
            if path is None or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Missing or changed original reference image.")
    renderer, judge = Renderer(), VLMJudge()
    manifest = Manifest(output / "manifest.jsonl")
    try:
        agent = BufoAgent(load_backends()["open_base"], renderer, output)
        for example in splits["heldout"][:limit]:
            row = await agent.generate(
                example["prompt"], prompt_id=example["prompt_id"], theme=example["theme"]
            )
            if not row.valid:
                manifest.append(row)
                raise RuntimeError(row.error or "Student produced an invalid SVG.")
            candidate = {**row.to_dict(), "reference_png": str(reference_path(dataset, example))}
            row.scores = {
                "svg_validity": SvgValidity().score(candidate),
                "vlm_judge": await judge.score(candidate),
            }
            manifest.append(row)
            print(
                json.dumps(
                    {
                        "prompt_id": row.prompt_id,
                        "valid": row.valid,
                        "png_path": row.png_path,
                        "scores": row.scores,
                    },
                    indent=2,
                )
            )
    finally:
        await renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/emoji-rl-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/emoji-rl-check"))
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    asyncio.run(check(args.dataset, args.output, limit=args.limit))


if __name__ == "__main__":
    main()
