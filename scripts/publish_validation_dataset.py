"""Publish the prompt-only validation set as a standalone Weave Dataset (no outputs, no judging).

Usage: uv run python scripts/publish_validation_dataset.py [--split validation] [--name bufo-validation-48]
The per-arm evaluation datasets carry each arm's rendered output; this one is the shared input set.
"""

import argparse
from pathlib import Path

from bufogen.evaluation.benchmark import load_benchmark
from bufogen.settings import ROOT


def rows_for(split: str, config: Path) -> list[dict]:
    prompts = [p for p in load_benchmark(config) if p["split"] == split]
    return [
        {
            "prompt_id": p["id"],
            "family": p["family"],
            "split": p["split"],
            "prompt": p["prompt"],
            "requirements": list(p.get("requirements", [])),
        }
        for p in sorted(prompts, key=lambda p: p["id"])
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--name", default=None)
    ap.add_argument("--config", type=Path, default=ROOT / "config/distill_prompts.yaml")
    args = ap.parse_args()
    rows = rows_for(args.split, args.config)
    if not rows:
        raise SystemExit(f"no prompts in split {args.split!r}")
    name = args.name or f"bufo-{args.split}-{len(rows)}"
    import weave

    from bufogen.integrations.telemetry import init_weave

    init_weave()
    ref = weave.publish(weave.Dataset(name=name, rows=rows))
    print(f"{name}: {len(rows)} rows, {len({r['family'] for r in rows})} families -> {ref.uri()}")


if __name__ == "__main__":
    main()
