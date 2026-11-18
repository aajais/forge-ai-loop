"""Generate/score real teacher trajectories; no automatic training acceptance."""
import asyncio
import hashlib
import json
from pathlib import Path

from bufogen.evaluation.benchmark import load_benchmark
from bufogen.evaluation.benchmark import run_benchmark
from bufogen.evaluation.benchmark import score_benchmark
from bufogen.models import load_backends
from bufogen.integrations.telemetry import init_weave

ROOT = Path(__file__).resolve().parents[2]

async def main():
    manifest = json.loads((ROOT / "runs/forge-gap/curriculum-manifest.json").read_text())
    source = ROOT / manifest["source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Curriculum changed after split freeze")
    prompts = [p for p in load_benchmark(source) if p["split"] in {"train", "validation"}]
    assert len(prompts) == 152
    configs = load_backends()
    teacher = configs["closed_quality_bar"]
    assert teacher.model == "gpt-6.1-sol"
    output = ROOT / "runs/forge-gap/teacher-curriculum"
    client = init_weave()
    try:
        rows = await run_benchmark(prompts, {teacher.name: teacher}, output, concurrency=16)
        print(json.dumps({"stage":"generated", "model_id":teacher.model,
                          "rows":len(rows), "valid":sum(r.valid for r in rows)}), flush=True)
        scored = await score_benchmark(output, concurrency=8, publish=True)
        print(json.dumps({"stage":"scored", "rows":len(scored),
                          "accepted_training_targets":0,
                          "review_status":"requires_explicit_image_reviews"}), flush=True)
    finally:
        client.finish()

if __name__ == "__main__":
    asyncio.run(main())
