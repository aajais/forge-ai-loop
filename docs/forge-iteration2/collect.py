"""Collect authentic updated-agent trajectories; never infer training acceptance."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from bufogen.evaluation.benchmark import load_benchmark
from bufogen.evaluation.benchmark import run_benchmark
from bufogen.models import load_backends
from bufogen.settings import ROOT
from bufogen.integrations.telemetry import init_weave

DIRECTORY = ROOT / "runs/forge-iteration2"


async def collect(stage):
    manifest = json.loads((DIRECTORY / "curriculum-manifest.json").read_text())
    source = ROOT / manifest["source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Frozen curriculum changed")
    prompts = load_benchmark(source)
    configs = load_backends()
    assert configs["closed_quality_bar"].model == "gpt-6.1-sol"
    assert configs["open_base"].model == "google/gemma-4-26B-A4B-it"
    if stage == "teacher":
        prompts = [p for p in prompts if p["split"] in {"train", "validation"}]
        assert len(prompts) == 104
        arms = [configs["closed_quality_bar"]]
    else:
        families = {"office_coffee", "bakery", "workshop_repair", "skate_park"}
        chosen = {}
        for p in prompts:
            if p["split"] == "train" and p["family"] in families:
                chosen.setdefault(p["family"], p)
        assert len(chosen) == 4
        prompts = list(chosen.values())
        arms = [configs["open_base"], configs["closed_quality_bar"]]
    client = init_weave()
    try:
        rows = await run_benchmark(
            prompts, arms, DIRECTORY / stage,
            concurrency=12, temperature=0.7, max_tokens=6000, max_turns=8,
        )
        plans = 0
        for row in rows:
            calls = [
                c["function"]["name"]
                for m in (row.trajectory or {}).get("messages", [])
                for c in m.get("tool_calls", [])
            ]
            plans += bool("plan_scene" in calls and "render_svg" in calls
                          and calls.index("plan_scene") < calls.index("render_svg"))
        print(json.dumps({
            "stage": stage, "attempts": len(rows), "valid": sum(r.valid for r in rows),
            "plan_before_render": plans, "training_acceptance": "requires_image_review",
        }), flush=True)
    finally:
        client.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "pilot-compatible", "teacher"], required=True)
    asyncio.run(collect(parser.parse_args().stage))
