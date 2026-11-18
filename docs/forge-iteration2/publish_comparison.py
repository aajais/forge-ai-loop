"""Publish matched PNG comparisons from completed, unchanged benchmark evidence."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wandb
from bufogen.models import frozen_model_presentations
from bufogen.models import generation_model_presentation
from bufogen.settings import project_path

BASE = "google/gemma-4-26B-A4B-it"
TEACHER = "gpt-6.1-sol"


def validated_evidence(root, stage):
    from evaluate import validated_evidence as validate
    return validate(root, stage)


def update_receipt(path, value):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validation", "confirmation"], required=True)
    args = parser.parse_args()
    root = Path("runs/forge-iteration2")
    directory = root / (
        "validation-candidates" if args.stage == "validation" else "confirmation-selected"
    )
    receipt_path = root / f"{args.stage}-visualization.json"
    if receipt_path.exists():
        raise RuntimeError("Publication already started; reconcile receipt before resuming")
    settings, generations, summary, _ = validated_evidence(root, args.stage)
    rows = [row.to_dict() for row in generations]
    arms = frozen_model_presentations(settings["configuration"]["arms"])
    prompts = settings["configuration"]["prompts"]
    lookup = {(r["prompt_id"], r["backend"]): r for r in rows}
    assert len(rows) == len(lookup) == len(prompts) * len(arms)
    assert set(lookup) == {(p["id"], arm) for p in prompts for arm in arms}
    assert settings["configuration_hash"] == summary["configuration_hash"]
    run_id = uuid.uuid4().hex[:8]
    receipt = {
        "run_id": run_id,
        "status": "publishing",
        "stage": args.stage,
        "comparison_fingerprint": settings["comparison_fingerprint"],
        "evidence_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("benchmark.json", "scored.jsonl", "summary.json")
        },
    }
    from evaluate import exclusive_json

    exclusive_json(receipt_path, receipt)
    entity, project = project_path().split("/")

    def passed(row):
        return bool(
            row["valid"]
            and row["benchmark_metadata"]["status"] == "completed"
            and row["scores"]["scene_quality"].get("quality_failure") is False
        )

    def png(row):
        path = row.get("png_path")
        return wandb.Image(path) if path and Path(path).is_file() else None

    with wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=f"Planned-scene distillation {args.stage}: {BASE} / {TEACHER} / ft-{BASE}",
        job_type="matched-model-comparison",
        config=receipt,
    ) as run:
        metrics = wandb.Table(
            columns=[
                "arm_id",
                "model_id",
                "checkpoint",
                "attempts",
                "strict_passes",
                "strict_pass_rate",
                "requirement_coverage_scored_only",
                "scored_images",
                "generation_errors",
                "judge_errors",
            ]
        )
        for name, arm in arms.items():
            subset = [r for r in rows if r["backend"] == name]
            identity = generation_model_presentation(subset[0], config=arm)
            score = summary["arms"][name]
            assert score["total"] == len(subset) and score["provisional_passes"] == sum(
                map(passed, subset)
            )
            metrics.add_data(
                name,
                identity["model_label"],
                identity["model_version"],
                len(subset),
                score["provisional_passes"],
                score["provisional_pass_rate_all_attempts"],
                score["metric_means_scored_only"]["constraint_coverage"],
                score["scored"],
                score["generation_errors"],
                score["judge_errors"],
            )
        payload = {"comparison_summary": metrics}
        for candidate in (a for a in arms if a.startswith("open_tuned_epoch")):
            chart = wandb.Table(columns=["model_id", "strict_pass_rate", "requirement_coverage"])
            for arm, label in (
                ("open_base", BASE),
                ("closed_quality_bar", TEACHER),
                (candidate, "ft-" + BASE),
            ):
                values = summary["arms"][arm]
                chart.add_data(
                    label,
                    values["provisional_pass_rate_all_attempts"],
                    values["metric_means_scored_only"]["constraint_coverage"],
                )
            payload[candidate + "_pass_rate"] = wandb.plot.bar(
                chart, "model_id", "strict_pass_rate", title="Strict scene passes — all attempts"
            )
            payload[candidate + "_coverage"] = wandb.plot.bar(
                chart,
                "model_id",
                "requirement_coverage",
                title="Visible requirement coverage — scored images",
            )
            table = wandb.Table(
                columns=[
                    "prompt_id",
                    "prompt",
                    "requirements",
                    BASE,
                    TEACHER,
                    "ft-" + BASE,
                    "base_pass",
                    "teacher_pass",
                    "ft_pass",
                    "ft_checkpoint",
                    "base_trace",
                    "teacher_trace",
                    "ft_trace",
                ]
            )
            for prompt in prompts:
                base, teacher, tuned = [
                    lookup[prompt["id"], arm]
                    for arm in ("open_base", "closed_quality_bar", candidate)
                ]
                table.add_data(
                    prompt["id"],
                    prompt["prompt"],
                    " | ".join(prompt["requirements"]),
                    png(base),
                    png(teacher),
                    png(tuned),
                    passed(base),
                    passed(teacher),
                    passed(tuned),
                    tuned["model"],
                    base.get("conversation_url"),
                    teacher.get("conversation_url"),
                    tuned.get("conversation_url"),
                )
            payload[candidate + "_paired_images"] = table
        run.log(payload)
        evidence = wandb.Artifact("bufo-iteration2-" + args.stage + "-evidence", type="evaluation")
        for name in (
            "benchmark.json",
            "manifest.jsonl",
            "scored.jsonl",
            "summary.json",
            "protocol.json",
            "weave-publication.json",
        ):
            path = directory / name
            if path.exists():
                evidence.add_file(str(path), name=name)
        for name in ("selection.json", "paired-results.json"):
            if (root / name).exists():
                evidence.add_file(str(root / name), name=name)
        artifact = run.log_artifact(evidence)
        artifact.wait()
        receipt.update(
            status="published",
            run_url=run.url,
            artifact_ref=f"{entity}/{project}/{artifact.name}",
            rows=len(rows),
        )
        update_receipt(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
