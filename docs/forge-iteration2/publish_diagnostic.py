"""Publish a completed teacher-plan diagnostic; default performs local validation only."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import uuid

from bufogen.evaluation import benchmark
from bufogen.generation import BufoGeneration
from bufogen.settings import ROOT
from bufogen.settings import project_path

BASE = "google/gemma-4-26B-A4B-it"
TEACHER = "gpt-6.1-sol"
ARMS = ("ordinary_plan", "teacher_plan_continuation")


def diagnostic_module():
    spec = importlib.util.spec_from_file_location(
        "plan_continuation_publication", ROOT / "docs/forge-iteration2/plan_continuation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(root, directory):
    module = diagnostic_module()
    prompts, arms, protocol = module.prepare(root)
    if module.read(directory / "diagnostic-protocol.json") != protocol:
        raise ValueError("Diagnostic protocol or source hashes changed")
    settings = module.read(directory / "benchmark.json")
    with module.diagnostic_hooks(protocol["plans"]):
        expected = benchmark._configuration(prompts, arms, **module.POLICY)
    config = settings["configuration"]
    if (
        config != expected
        or settings["configuration_hash"] != benchmark.digest(config)
        or settings["comparison_fingerprint"] != benchmark.comparison_fingerprint(config)
    ):
        raise ValueError("Frozen diagnostic generation configuration differs")
    manifest = module.jsonl(directory / "manifest.jsonl")
    scored = module.jsonl(directory / "scored.jsonl")
    pairs = {(p["id"], arm) for p in prompts for arm in ARMS}
    for rows in (manifest, scored):
        if len(rows) != 32 or {(r["prompt_id"], r["backend"]) for r in rows} != pairs:
            raise ValueError("Expected all 32 unique prompt/arm attempts, including failures")
        if len({r["id"] for r in rows}) != 32:
            raise ValueError("Duplicate generation IDs")
        benchmark._validate_rows([BufoGeneration(**r) for r in rows], settings)
    originals = {row["id"]: row for row in manifest}
    for row in scored:
        original = originals.get(row["id"])
        if original is None or any(
            row.get(key) != value for key, value in original.items() if key != "scores"
        ):
            raise ValueError("Scored attempt differs from generation evidence")
        score = row.get("scores", {}).get("scene_quality", {})
        if (
            score.get("scorer_configuration") != config["scorer"]
            or score.get("generation_id") != row["id"]
            or score.get("requirements") != row["benchmark_metadata"]["requirements"]
            or score.get("status") not in {"ok", "judge_error", "missing_image", "prior_turn_failure"}
        ):
            raise ValueError("Missing or mismatched terminal scoring record")
        if score["status"] == "ok":
            checks = score.get("prompt_checks", [])
            requirements = score["requirements"]
            if (
                sorted(c["requirement"] for c in checks) != sorted(requirements)
                or any(type(c.get("satisfied")) is not bool for c in checks)
                or score["constraint_coverage"] != sum(c["satisfied"] for c in checks) / len(requirements)
            ):
                raise ValueError("Score checklist or coverage differs")
        trajectory = row.get("trajectory")
        if row["benchmark_metadata"]["status"] == "completed" and trajectory is None:
            raise ValueError("Completed diagnostic has no provenance")
        if trajectory:
            provenance = trajectory["provenance"]
            diagnostic = provenance.get("diagnostic", {})
            if (
                provenance.get("training_eligible") is not False
                or diagnostic.get("arm") != row["backend"]
                or not any("not training eligible" in d for d in trajectory["diagnostics"])
            ):
                raise ValueError("Diagnostic training exclusion/provenance missing")
            if row["backend"] == ARMS[1] and (
                provenance.get("history_authentic") is not False
                or provenance.get("source_models") != [TEACHER, protocol["model"]]
                or diagnostic.get("plan") != protocol["plans"][row["prompt_id"]]
            ):
                raise ValueError("Teacher prefix lineage was misattributed")
    summary = module.read(directory / "summary.json")
    if summary != benchmark._summary([BufoGeneration(**r) for r in scored], settings):
        raise ValueError("Summary differs from all attempted generations")
    examples = {e["prompt_id"]: e for e in module.jsonl(root / "training/reviewed-examples.jsonl")}
    targets = {}
    for prompt in prompts:
        example = examples[prompt["id"]]
        review = example["training_review"]
        if (
            example["source_generation_id"] != protocol["plans"][prompt["id"]]["source_generation_id"]
            or example["source_model"] != TEACHER
            or example["partition"] != "train"
            or review["decision"] != "accept"
            or module.sha(review["source_ref"]) != review["png_sha256"]
        ):
            raise ValueError("Reviewed training target changed")
        targets[prompt["id"]] = example
    files = (
        "diagnostic-protocol.json", "benchmark.json", "manifest.jsonl",
        "scored.jsonl", "summary.json", "plan_continuation.py",
    )
    hashes = {name: module.sha(directory / name) for name in files}
    hashes["publisher_source"] = module.sha(__file__)
    return prompts, scored, targets, summary, protocol, hashes


def publish(root, directory, *, execute=False, resume_empty_run=False):
    prompts, rows, targets, summary, protocol, hashes = validate(root, directory)
    if not execute:
        return {
            "dry_validation": "passed", "network_calls": 0, "attempts": len(rows),
            "paired_rows": len(prompts), "teacher_training_targets": len(targets),
            "quality_claim": "none; training-only diagnostic", "arms": summary["arms"],
        }
    receipt_path = directory / "diagnostic-visualization.json"
    resuming = False
    with benchmark._locked(directory):
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("evidence_sha256") != hashes:
                previous = receipt.get("evidence_sha256", {})
                if not resume_empty_run or {
                    k: v for k, v in previous.items() if k != "publisher_source"
                } != {k: v for k, v in hashes.items() if k != "publisher_source"}:
                    raise ValueError("Publication receipt evidence differs")
            if receipt.get("status") == "published":
                return receipt
            if not resume_empty_run:
                raise RuntimeError(
                    "Publication already started; reconcile its recorded W&B run before retrying."
                )
            import wandb
            remote = wandb.Api(timeout=60).run(project_path() + "/" + receipt["run_id"])
            observation = {
                "run_id": remote.id, "state": remote.state,
                "history": remote.history(samples=1, pandas=False),
                "logged_artifacts": [a.name for a in remote.logged_artifacts()],
            }
            if remote.state not in {"failed", "crashed"} or observation["history"] or observation["logged_artifacts"]:
                raise ValueError("Existing run is not an empty failed initialization; refusing recovery")
            # Preserve the failed initialization receipt and GET evidence before reusing its ID.
            archive = directory / "diagnostic-publication-recovery.json"
            if archive.exists():
                raise ValueError("A publication recovery was already attempted; reconcile it first")
            with archive.open("x") as stream:
                json.dump({"original_receipt": receipt, "observation": observation,
                           "reason": "W&B auto-parsed the serving URI during config initialization"}, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            receipt["evidence_sha256"] = hashes
            receipt["recovery"] = "Reused verified empty failed run; no generation repeated"
            benchmark._write_json(receipt_path, receipt)
            resuming = True
        else:
            receipt = {
                "schema_version": 1, "status": "publishing", "run_id": uuid.uuid4().hex[:8],
                "purpose": "training-only teacher-plan continuation diagnostic",
                "quality_claim": "none", "promotion": False, "attempts": len(rows),
                "teacher_targets": len(targets), "evidence_sha256": hashes,
            }
            with receipt_path.open("x") as stream:
                json.dump(receipt, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
    import wandb

    def image(row):
        return wandb.Image(row["png_path"]) if row.get("png_path") else None

    lookup = {(r["prompt_id"], r["backend"]): r for r in rows}
    pairs = wandb.Table(columns=[
        "prompt_id", "prompt", "requirements", "student_model_id", "student_checkpoint",
        "ordinary_png", "teacher_prefix_png", "reviewed_training_target_png",
        "teacher_model_id", "teacher_target_role", "teacher_target_generation_id",
        "teacher_target_png_sha256", "teacher_target_assistant_review", "ordinary_checks", "teacher_prefix_checks",
        "ordinary_status", "teacher_prefix_status", "ordinary_error", "teacher_prefix_error",
        "ordinary_mixed_prefix", "teacher_prefix_mixed_prefix",
        "ordinary_trace", "teacher_prefix_trace",
    ])
    attempts = wandb.Table(columns=[
        "prompt_id", "arm", "model_id", "checkpoint", "generation_id", "png",
        "mixed_teacher_prefix", "training_eligible", "generation_status", "generation_error",
        "score_status", "requirement_checks", "constraint_coverage", "provisional_pass",
        "trace",
    ])
    for prompt in prompts:
        ordinary, supplied = [lookup[prompt["id"], arm] for arm in ARMS]
        target = targets[prompt["id"]]
        pairs.add_data(
            prompt["id"], prompt["prompt"], prompt["requirements"], "ft-" + BASE, protocol["model"],
            image(ordinary), image(supplied), wandb.Image(target["training_review"]["source_ref"]),
            TEACHER, "Previously reviewed training target; not a new diagnostic generation",
            target["source_generation_id"], target["training_review"]["png_sha256"],
            target["training_review"]["evidence"],
            ordinary["scores"]["scene_quality"]["prompt_checks"],
            supplied["scores"]["scene_quality"]["prompt_checks"],
            ordinary["benchmark_metadata"]["status"], supplied["benchmark_metadata"]["status"],
            ordinary.get("error"), supplied.get("error"), False, True,
            ordinary.get("conversation_url"), supplied.get("conversation_url"),
        )
        for row in (ordinary, supplied):
            score = row["scores"]["scene_quality"]
            attempts.add_data(
                row["prompt_id"], row["backend"], "ft-" + BASE, row["model"], row["id"], image(row),
                row["backend"] == ARMS[1], False, row["benchmark_metadata"]["status"], row.get("error"),
                score["status"], score["prompt_checks"], score.get("constraint_coverage"),
                bool(row["valid"] and row["benchmark_metadata"]["status"] == "completed"
                     and score.get("quality_failure") is False), row.get("conversation_url"),
            )
    entity, project = project_path().split("/")
    with wandb.init(
        entity=entity, project=project, id=receipt["run_id"],
        resume="must" if resuming else "never", allow_val_change=resuming,
        name=f"Teacher-plan diagnostic: ft-{BASE}", job_type="training-only-diagnostic",
        # W&B treats a top-level wandb-artifact URI as an artifact instruction.
        # Store the ordinary artifact path here; exact serving URI stays in table/evidence.
        config=receipt | {"model_id": "ft-" + BASE,
                          "student_checkpoint": protocol["model"].removeprefix("wandb-artifact:///"),
                          "temperature": 0.0,
                          "interpretation": protocol["interpretation"]},
    ) as run:
        run.log({"paired_images_and_training_targets": pairs, "all_attempts": attempts})
        for arm, values in summary["arms"].items():
            run.summary[f"{arm}/attempts"] = values["total"]
            run.summary[f"{arm}/provisional_passes"] = values["provisional_passes"]
            run.summary[f"{arm}/generation_errors"] = values["generation_errors"]
            run.summary[f"{arm}/requirement_coverage_scored_only"] = values["metric_means_scored_only"]["constraint_coverage"]
        artifact = wandb.Artifact("bufo-plan-continuation-diagnostic", type="evaluation")
        for name in hashes:
            if name != "publisher_source":
                artifact.add_file(str(directory / name), name=name)
        artifact.add_file(__file__, name="publish_diagnostic.py")
        saved = run.log_artifact(artifact)
        saved.wait()
        run_url, artifact_ref = run.url, f"{entity}/{project}/{saved.name}"
    # Mark successful only after the W&B run context finishes flushing.
    receipt.update(status="published", run_url=run_url, artifact_ref=artifact_ref)
    benchmark._write_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/forge-iteration2")
    parser.add_argument("--directory", type=Path, default=ROOT / "runs/forge-iteration2/plan-continuation")
    parser.add_argument("--publish", action="store_true", help="Upload one diagnostic W&B run")
    parser.add_argument("--resume-empty-run", action="store_true",
                        help="Reuse a verified empty failed run after initialization failure")
    args = parser.parse_args()
    print(json.dumps(publish(args.root.resolve(), args.directory.resolve(), execute=args.publish,
                             resume_empty_run=args.resume_empty_run), indent=2))


if __name__ == "__main__":
    main()
