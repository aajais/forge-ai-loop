"""Publish the eight-case serving pilot and explicit assistant image observations."""
import hashlib
import json
from pathlib import Path
import uuid

import wandb
from bufogen.settings import ROOT
from bufogen.settings import project_path

BASE = "google/gemma-4-26B-A4B-it"
TEACHER = "gpt-6.1-sol"
OBSERVATIONS = {
    "bakery": ("Rolling pin extends outside the right frame; no visible hand grips its handle.",
               "Visible green fingers wrap around the lower rolling-pin handle."),
    "office_coffee": ("No blue mug is visible; a blue torso garment occupies the foreground.",
                      "A connected arm reaches fingers into the visible blue mug handle."),
    "skate_park": ("Board lies behind the legs; neither visible foot rests on the deck.",
                   "One visible foot rests directly on the blue skateboard deck."),
    "workshop_repair": ("Several right-side arm shapes and a small blue shape; no recognizable gripped wrench.",
                       "A green hand wraps around a recognizable gray wrench attached to the arm."),
}


def main():
    directory = ROOT / "runs/forge-iteration2/pilot-compatible"
    receipt_path = directory / "visualization.json"
    if receipt_path.exists():
        raise RuntimeError("Pilot publication already started; inspect receipt before any retry")
    rows = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    lookup = {(row["benchmark_metadata"]["family"], row["model"]): row for row in rows}
    assert len(rows) == len(lookup) == 8
    assert set(lookup) == {(family, model) for family in OBSERVATIONS for model in (BASE, TEACHER)}
    table = wandb.Table(columns=["prompt", BASE, TEACHER, "assistant_base_observation",
                               "assistant_teacher_observation", "base_trace", "teacher_trace"])
    evidence = []
    for family, observations in OBSERVATIONS.items():
        base, teacher = [lookup[family, model] for model in (BASE, TEACHER)]
        assert base["prompt"] == teacher["prompt"]
        for row, observation in zip((base, teacher), observations):
            path = Path(row["png_path"])
            assert row["valid"] and path.is_file()
            evidence.append({"generation_id": row["id"], "prompt_id": row["prompt_id"],
                "model": row["model"], "png_path": str(path),
                "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "reviewer_kind": "assistant", "reviewer_id": "root-pilot-review",
                "observation": observation, "purpose": "diagnostic_only_not_training_acceptance"})
        table.add_data(base["prompt"], wandb.Image(base["png_path"]), wandb.Image(teacher["png_path"]),
                       *observations, base.get("conversation_url"), teacher.get("conversation_url"))
    (directory / "assistant-observations.json").write_text(json.dumps(evidence, indent=2) + "\n")
    receipt = {"status": "publishing", "run_id": uuid.uuid4().hex[:8], "attempts": 8,
               "purpose": "serving_compatibility_and_visual_diagnostic", "quality_claim": "none"}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    entity, project = project_path().split("/")
    with wandb.init(entity=entity, project=project, id=receipt["run_id"],
                    name=f"Planned-scene pilot: {BASE} / {TEACHER}", job_type="visual-diagnostic",
                    config=receipt) as run:
        run.log({"paired_images": table})
        artifact = wandb.Artifact("bufo-iteration2-pilot", type="evaluation")
        for name in ("benchmark.json", "manifest.jsonl", "assistant-observations.json", "sheet.png"):
            artifact.add_file(str(directory / name), name=name)
        saved = run.log_artifact(artifact)
        saved.wait()
        receipt.update(status="published", run_url=run.url, artifact_ref=f"{entity}/{project}/{saved.name}")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
