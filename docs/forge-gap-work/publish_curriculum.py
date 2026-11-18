"""Publish actual reviewed targets and all rejected images with explicit provenance."""

import hashlib
import json
import uuid
from pathlib import Path

import weave

import wandb
from bufogen.integrations.observability import loaded_image
from bufogen.settings import project_path
from bufogen.integrations.telemetry import init_weave

ROOT = Path("runs/forge-gap")


def main():
    training = ROOT / "training"
    source = training / "reviewed-examples.jsonl"
    examples = [json.loads(s) for s in source.read_text().splitlines()]
    accepted = {e["training_review"]["source_generation_id"]: e for e in examples}
    rows = [
        json.loads(s) for s in (ROOT / "teacher-curriculum/scored.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 152
    receipt_path = ROOT / "curriculum-publication.json"
    if receipt_path.exists():
        raise RuntimeError("Publication receipt exists; reconcile its run before retrying")
    run_id = uuid.uuid4().hex[:8]
    receipt = {
        "status": "publishing",
        "run_id": run_id,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "teacher_model_id": "gpt-6.1-sol",
        "reviewer_kind": "assistant",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    entity, project = project_path().split("/")
    client = init_weave()
    try:
        with wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            name="gpt-6.1-sol reviewed Bufo training targets",
            job_type="reviewed-distillation-dataset",
            config=receipt,
        ) as run:
            table = wandb.Table(
                columns=[
                    "model_id",
                    "prompt_id",
                    "family",
                    "partition",
                    "prompt",
                    "image",
                    "selected_for_dataset",
                    "judge_pass",
                    "judge_reason",
                    "trace",
                ]
            )
            for row in rows:
                score = row["scores"]["scene_quality"]
                table.add_data(
                    row["model"],
                    row["prompt_id"],
                    row["benchmark_metadata"]["family"],
                    row["benchmark_metadata"]["split"],
                    row["prompt"],
                    wandb.Image(row["png_path"]),
                    row["id"] in accepted,
                    score.get("quality_failure") is False,
                    score.get("reason"),
                    row.get("conversation_url"),
                )
            artifact = wandb.Artifact(
                "bufo-gap-reviewed-targets",
                type="dataset",
                metadata={
                    "teacher_model_id": "gpt-6.1-sol",
                    "reviewer_kind": "assistant",
                    "source_sha256": receipt["source_sha256"],
                    "accepted": len(examples),
                    "candidates": len(rows),
                    "confirmation_included": False,
                },
            )
            artifact.add_file(str(source), name=source.name)
            for path in sorted(training.glob("*.json*")):
                if path != source and "state" not in path.name:
                    artifact.add_file(str(path), name=path.name)
            artifact.add_file("config/forge_gap_prompts.yaml", name="curriculum.yaml")
            artifact.add_file(
                str(ROOT / "curriculum-manifest.json"), name="curriculum-manifest.json"
            )
            for path in sorted((ROOT / "review").glob("*/*.jsonl")):
                artifact.add_file(str(path), name="review/" + path.parent.name + "/" + path.name)
            artifact = run.log_artifact(artifact)
            artifact.wait()
            run.log({"reviewed_curriculum_images": table})
            receipt.update(run_url=run.url, artifact_ref=f"{entity}/{project}/{artifact.name}")
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            image_by_id = {r["id"]: r["png_path"] for r in rows}
            dataset_rows = [
                {
                    **e,
                    "image": loaded_image(
                        image_by_id[e["training_review"]["source_generation_id"]]
                    ),
                }
                for e in examples
            ]
            ref = weave.publish(weave.Dataset(name="bufo-gap-reviewed-targets", rows=dataset_rows))
            receipt.update(
                status="published",
                weave_dataset_ref=ref.uri(),
                accepted=len(examples),
                rejected=len(rows) - len(examples),
            )
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt), flush=True)
    finally:
        client.finish()


if __name__ == "__main__":
    main()
