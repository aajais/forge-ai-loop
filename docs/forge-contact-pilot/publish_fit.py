"""Publish a frozen in-sample fit comparison; default is offline validation only."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import uuid

from bufogen.evaluation import benchmark
from bufogen.settings import ROOT
from bufogen.settings import project_path

HERE = Path(__file__).resolve().parent
PURPOSE = "24 training-request fit comparison; not generalization, Sol parity or promotion"


def fit_module():
    spec = importlib.util.spec_from_file_location("contact_fit_publication_validator", HERE / "fit_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(root):
    fit = fit_module()
    root = Path(root).resolve()
    comparison_path = root / "fit-comparison/comparison.json"
    if not comparison_path.is_file():
        raise ValueError("Complete and freeze fit_compare --summary before publication")
    with ExitStack() as locks:
        for directory, _ in fit.ARMS.values():
            if not (root / directory).is_dir():
                raise ValueError("Missing completed fit source")
            locks.enter_context(benchmark._locked(root / directory))
        locks.enter_context(benchmark._locked(root / "fit-comparison"))
        data = fit.validate(root)
        frozen = fit.read(comparison_path)
        if fit.summarize(data) != frozen:
            raise ValueError("Frozen comparison differs from complete validated fit evidence")
        reviews, _ = fit.completed_reviews(data)
        scores, _ = fit.completed_scores(data)
        hashes = frozen["evidence_sha256"] | {
            str(comparison_path): fit.sha(comparison_path),
            str(Path(__file__).resolve()): fit.sha(Path(__file__).resolve()),
        }
        evidence = [
            {"source_path": path, "sha256": digest, "artifact_name": f"evidence/{index:04d}-{Path(path).name}"}
            for index, (path, digest) in enumerate(sorted(hashes.items()))
        ]
        result = {"fit": fit, "data": data, "comparison": frozen, "reviews": reviews,
                  "scores": scores, "hashes": hashes, "evidence_files": evidence}
        ensure_hashes(result)
        return result


def ensure_hashes(result):
    if any(result["fit"].sha(path) != digest for path, digest in result["hashes"].items()):
        raise ValueError("Validated fit publication evidence changed")


def verdicts(row, score, review):
    complete = bool(row["valid"] and row["benchmark_metadata"]["status"] == "completed")
    automated_image = score.get("quality_failure") is False
    assistant_image = review["passed"]
    return {
        "completed_and_valid": complete, "automated_image_pass": automated_image,
        "automated_completed_pass": complete and automated_image,
        "assistant_image_pass": assistant_image, "assistant_completed_pass": complete and assistant_image,
        "combined_pass": complete and automated_image and assistant_image,
    }


def tables(result, wandb):
    fit, data, frozen = result["fit"], result["data"], result["comparison"]
    checkpoint = frozen["candidate_identity"]["checkpoint"]

    def image(row):
        return wandb.Image(row["png_path"]) if row.get("png_path") else None

    def values(entry):
        row = entry["generation"]
        return verdicts(row, result["scores"][row["id"]], result["reviews"][row["id"]])

    verdict_names = [
        "completed_and_valid", "automated_image_pass", "automated_completed_pass",
        "assistant_image_pass", "assistant_completed_pass", "combined_pass",
    ]
    pairs = wandb.Table(columns=[
        "prompt_id", "prompt", "requirements", "base_model_id", "candidate_model_label", "candidate_checkpoint",
        *[f"{arm}_{field}" for arm in fit.ARMS for field in (
            "png", "generation_status", *verdict_names, "assistant_evidence", "trace",
        )],
    ])
    lookup = {(entry["generation"]["prompt_id"], entry["arm"]): entry for entry in data["rows"]}
    for prompt in data["prompts"]:
        row_values = [prompt["id"], prompt["prompt"], prompt["requirements"], fit.BASE, f"ft-{fit.BASE}", checkpoint]
        for arm in fit.ARMS:
            entry = lookup[prompt["id"], arm]
            row, evaluated = entry["generation"], values(entry)
            row_values.extend([
                image(row), row["benchmark_metadata"]["status"], *[evaluated[key] for key in verdict_names],
                result["reviews"][row["id"]]["evidence"], row.get("conversation_url"),
            ])
        pairs.add_data(*row_values)
    attempts = wandb.Table(columns=[
        "arm", "model_label", "actual_model_id", "immutable_checkpoint", "generation_id", "prompt_id",
        "prompt", "requirements", "png", "generation_status", "generation_valid", "generation_error",
        *verdict_names, "score_status", "requirement_checks", "automated_reason",
        "assistant_evidence", "assistant_failed_requirements", "blind_label", "reviewer_packet",
        "captured_png_sha256", "review_png_sha256", "capture_manifest_sha256", "review_file_sha256",
        "recorded_input_tokens", "recorded_output_tokens", "recorded_latency_s", "tool_calls",
        "prompt_version", "trace",
    ])
    for entry in data["rows"]:
        arm, row = entry["arm"], entry["generation"]
        score, review = result["scores"][row["id"]], result["reviews"][row["id"]]
        evaluated = values(entry)
        packet = f"reviewer-{review['index'] % 3}"
        review_path = data["root"] / "fit-comparison/visual-review" / packet / "decisions.jsonl"
        manifest_path = data["root"] / fit.ARMS[arm][0] / "manifest.jsonl"
        attempts.add_data(
            arm, fit.BASE if arm == "baseline" else f"ft-{fit.BASE}", row["model"],
            None if arm == "baseline" else checkpoint, row["id"], row["prompt_id"], row["prompt"],
            row["benchmark_metadata"]["requirements"], image(row), row["benchmark_metadata"]["status"],
            row["valid"], row.get("error"), *[evaluated[key] for key in verdict_names],
            score["status"], score["prompt_checks"], score.get("reason", score.get("error")),
            review["evidence"], review["failed_requirements"], review["label"], packet,
            row["benchmark_metadata"].get("png_sha256"), review["png_sha256"],
            result["hashes"][str(manifest_path)], result["hashes"][str(review_path)],
            row["input_tokens"], row["output_tokens"], row["latency_s"], row["tool_calls"],
            row["prompt_version"], row.get("conversation_url"),
        )
    summaries = wandb.Table(columns=["arm", *frozen["models"]["baseline"]])
    for arm, metrics in frozen["models"].items():
        summaries.add_data(arm, *[metrics[key] for key in frozen["models"]["baseline"]])
    return {"paired_training_request_images": pairs, "all_48_attempts": attempts, "fit_summary": summaries}


def publish(root, execute=False):
    root = Path(root).resolve()
    result = validate(root)
    frozen, fit = result["comparison"], result["fit"]
    if not execute:
        return {
            "offline_validation": "passed", "network_calls": 0, "attempts": 48, "paired_rows": 24,
            "purpose": PURPOSE, "in_sample": True, "promotion": False,
            "models": frozen["models"], "gate": frozen["gate"],
        }
    directory = root / "fit-comparison"
    receipt_path = directory / "visualization.json"
    destination = project_path()
    with benchmark._locked(directory):
        ensure_hashes(result)
        if receipt_path.exists():
            prior = fit.read(receipt_path)
            if prior.get("evidence_sha256") != result["hashes"] or prior.get("project") != destination:
                raise ValueError("Existing publication identity/evidence differs; no new upload attempted")
            if prior.get("status") == "published":
                return prior
            raise RuntimeError("Publication already started; reconcile its saved run ID before retrying")
        receipt = {
            "schema_version": 1, "status": "publishing", "run_id": uuid.uuid4().hex[:8],
            "project": destination, "purpose": PURPOSE, "in_sample": True, "promotion": False,
            "attempts": 48, "paired_rows": 24, "candidate_identity": frozen["candidate_identity"],
            "gate": frozen["gate"], "model_summary": frozen["models"], "limitations": frozen["limitations"],
            "evidence_sha256": result["hashes"], "evidence_files": result["evidence_files"],
        }
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    import wandb

    visualization = tables(result, wandb)
    ensure_hashes(result)
    entity, project = destination.split("/")
    # W&B interprets URI-valued config strings as artifact requests. The exact URI
    # stays in tables/receipt/evidence; config uses its unambiguous versioned path.
    config = {
        "purpose": PURPOSE, "in_sample": True, "promotion": False,
        "base_model_id": fit.BASE, "candidate_model_label": f"ft-{fit.BASE}",
        "candidate_artifact_version": frozen["candidate_identity"]["checkpoint"].removeprefix("wandb-artifact:///"),
        "finetune_id": frozen["candidate_identity"]["finetune_id"],
        "training_recipe": frozen["candidate_identity"]["recipe"],
        "attempts": 48, "paired_rows": 24, "fit_gate": frozen["gate"],
        "evidence_manifest_sha256": benchmark.digest(result["hashes"]),
    }
    config = json.loads(json.dumps(config, allow_nan=False))
    with wandb.init(
        entity=entity, project=project, id=receipt["run_id"], resume="never",
        name=f"In-sample contact fit: {fit.BASE} / ft-{fit.BASE}",
        job_type="in-sample-fit-evaluation", config=config,
    ) as run:
        run.log(visualization)
        for arm, metrics in frozen["models"].items():
            for key, value in metrics.items():
                if key not in {"model_id", "model_label"}:
                    run.summary[f"{arm}/{key}"] = value
        run.summary["in_sample_fit_gate_passed"] = frozen["gate"]["passed"]
        artifact = wandb.Artifact("bufo-contact-v2-in-sample-fit", type="evaluation")
        for item in result["evidence_files"]:
            artifact.add_file(item["source_path"], name=item["artifact_name"])
        artifact.add_file(str(receipt_path), name="publication-receipt-at-upload.json")
        saved = run.log_artifact(artifact)
        saved.wait()
        run_url, artifact_ref = str(run.url), f"{destination}/{saved.name}"
    ensure_hashes(result)
    receipt.update(status="published", run_url=run_url, artifact_ref=artifact_ref)
    with benchmark._locked(directory):
        current = fit.read(receipt_path)
        if current["run_id"] != receipt["run_id"] or current["status"] != "publishing":
            raise RuntimeError("Publication receipt changed; reconcile the saved run before retrying")
        benchmark._write_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/forge-contact-pilot/learnability")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args.root, execute=args.publish), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
