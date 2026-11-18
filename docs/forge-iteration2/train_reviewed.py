"""Freeze assistant-reviewed iteration-2 targets and run durable managed Forge SFT.

Default prepares local evidence only. --train explicitly imports the reviewed
bytes and submits independent e1/e3 jobs; --status only reads existing remote
jobs. Uncertain writes remain blocked until reconciled by the shared workflow.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import ssl
import tempfile
from uuid import uuid4

import certifi
import httpx

from bufogen.evaluation.benchmark import _validate_rows
from bufogen.evaluation.benchmark import comparison_fingerprint
from bufogen.evaluation.benchmark import digest
from bufogen.evaluation.benchmark import load_benchmark
from bufogen.curation.curate import _validate_review
from bufogen.integrations.forge import STUDENT_MODEL
from bufogen.integrations.forge import ForgeClient
from bufogen.integrations.forge import reviewed_entries
from bufogen.workflows.forge_loop import _clear_or_block
from bufogen.workflows.forge_loop import _locked
from bufogen.workflows.forge_loop import import_reviewed_dataset
from bufogen.workflows.forge_loop import initialize_workflow
from bufogen.workflows.forge_loop import resume_training
from bufogen.workflows.forge_loop import submit_training
from bufogen.generation import Manifest
from bufogen.agents.prompting import prompt_version
from bufogen.settings import ROOT
from bufogen.settings import required
from bufogen.curation.trajectories import export_training_example

DIRECTORY = ROOT / "runs/forge-iteration2"
TEACHER = "gpt-6.1-sol"
RECIPES = {"e1": 1, "e3": 3}


def transport_for_cli(tls12):
    """Opt into the verified TLS 1.2 upload compatibility workaround."""
    if not tls12:
        return None, {"mode": "default"}
    context = ssl.create_default_context(cafile=certifi.where())
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_2
    return httpx.HTTPTransport(verify=context), {
        "mode": "tls12", "certificate_verification": True, "hostname_check": True,
        "minimum_tls_version": "TLSv1.2", "maximum_tls_version": "TLSv1.2",
    }


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def jsonl(rows):
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing changed frozen evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def completed_plan(messages):
    """Require a successful recorded plan result before the first render call."""
    pending, latest = {}, None
    for message in messages:
        for call in message.get("tool_calls", []):
            name = call["function"]["name"]
            if name == "render_svg":
                if latest is None:
                    raise ValueError("Accepted trajectory requires successful plan_scene before render_svg")
                return latest
            if name == "plan_scene":
                pending[call["id"]] = call
        if message.get("role") == "tool" and message.get("tool_call_id") in pending:
            try:
                result = json.loads(message.get("content", ""))
            except (ValueError, TypeError):
                result = None
            if (isinstance(result, dict) and result.get("version") == "bufo-scene-plan-v1"
                    and result.get("verification") == "planned_geometry_only"
                    and not result.get("error")):
                latest = {"tool_call_id": message["tool_call_id"],
                          "warnings": result.get("warnings", []),
                          "verification": "planned_geometry_only"}
    raise ValueError("Accepted trajectory requires successful plan_scene and render_svg")


def prepare(root=DIRECTORY):
    root = Path(root).resolve()
    directory, teacher = root / "training", root / "teacher"
    with _locked(directory / "prepare"):
        receipt_path = directory / "source-receipt.json"
        expected_version = (read(receipt_path)["prompt_version"]
                            if receipt_path.exists() else prompt_version())
        curriculum_path = root / "curriculum-manifest.json"
        curriculum = read(curriculum_path)
        source_path = Path(curriculum["source"])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if sha(source_path.read_bytes()) != curriculum["sha256"]:
            raise ValueError("Frozen curriculum source changed")
        prompts = load_benchmark(source_path)
        counts = dict(Counter(p["split"] for p in prompts))
        if counts != curriculum["counts"] or counts.get("train") != 80 or counts.get("validation") != 24:
            raise ValueError("Require frozen 80 train and 24 validation prompts before rejection")
        expected = {p["id"]: p for p in prompts if p["split"] in {"train", "validation"}}
        heldout = [p for p in prompts if p["split"] in {"heldout", "test"}]
        heldout_ids = {p["id"] for p in heldout}
        heldout_families = {p["family"] for p in heldout}
        if not heldout_ids or not heldout_families:
            raise ValueError("Fresh held-out families must be declared before training")
        settings_path, manifest_path = teacher / "benchmark.json", teacher / "manifest.jsonl"
        settings = read(settings_path)
        config = settings["configuration"]
        if (digest(config) != settings["configuration_hash"]
                or comparison_fingerprint(config) != settings["comparison_fingerprint"]
                or config.get("prompt_version") != expected_version
                or {p["id"]: p for p in config["prompts"]} != expected
                or len(config["prompts"]) != len(expected)
                or len(config["arms"]) != 1
                or config["arms"][0]["model"] != TEACHER):
            raise ValueError("Teacher benchmark differs from the frozen iteration-2 curriculum/harness")
        rows = sorted(Manifest(manifest_path).read(), key=lambda r: r.prompt_id)
        if (len(rows) != 104 or len({r.id for r in rows}) != 104
                or {r.prompt_id for r in rows} != set(expected)):
            raise ValueError("Require exactly one teacher generation per 104 curriculum prompts")
        _validate_rows(rows, settings)
        by_id = {row.id: row for row in rows}
        paths = sorted((root / "reviews").glob("*/decisions.jsonl"))
        reviews = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
        if (len(reviews) != 104 or any(not isinstance(r, dict) for r in reviews)
                or {r.get("source_generation_id") for r in reviews} != set(by_id)):
            raise ValueError("Require exactly one explicit review per teacher generation")
        reviews.sort(key=lambda r: r["prompt_id"])
        bound, examples, excluded, images, plans = [], [], [], [], []
        for review in reviews:
            row = by_id[review["source_generation_id"]]
            _validate_review(review, row)
            prompt = expected[row.prompt_id]
            if (row.model != TEACHER or row.prompt_version != expected_version
                    or review["partition"] != prompt["split"]
                    or review["prompt_family"] != prompt["family"]):
                raise ValueError("Review/model partition or family differs from curriculum")
            if review.get("visual_quality_pass") is not (review["decision"] == "accept"):
                raise ValueError("Review visual_quality_pass and explicit decision disagree")
            if not row.png_path:
                if review["decision"] != "reject" or review.get("png_sha256") is not None:
                    raise ValueError("Missing image cannot qualify an accepted review")
                if Path(review["source_ref"]).resolve() != manifest_path.resolve():
                    raise ValueError("Missing-image rejection must reference the source manifest")
                image_receipt = {"generation_id": row.id, "prompt_id": row.prompt_id,
                                 "png_path": None, "png_sha256": None}
            else:
                image = Path(row.png_path).resolve()
                image_hash = sha(image.read_bytes())
                if image_hash != review.get("png_sha256") or Path(review["source_ref"]).resolve() != image:
                    raise ValueError("Review PNG source/hash differs from actual teacher image")
                image_receipt = {"generation_id": row.id, "prompt_id": row.prompt_id,
                                 "png_path": str(image), "png_sha256": image_hash}
            images.append(image_receipt)
            qualified = replace(row, training_review=deepcopy(review))
            bound.append(qualified.to_dict())
            if review["decision"] == "reject":
                excluded.append({"source_generation_id": row.id, "prompt_id": row.prompt_id,
                                 "reason": "explicit_review_rejection", "evidence": review["evidence"]})
                continue
            plan = completed_plan((row.trajectory or {}).get("messages", []))
            example = export_training_example(row)
            metadata = deepcopy(prompt.get("curriculum_metadata", {}))
            if metadata.get("version") != "bufo-curriculum-v2":
                raise ValueError("Accepted target lacks iteration-2 curriculum lineage")
            example.update(partition=review["partition"], prompt_family=review["prompt_family"],
                           training_review=deepcopy(review), origin="distillation",
                           metadata={"curriculum_metadata": metadata})
            example["provenance"]["curriculum_metadata"] = deepcopy(metadata)
            examples.append(example)
            plans.append({"source_generation_id": row.id, **plan})
        examples.sort(key=lambda r: r["prompt_id"])
        entries = reviewed_entries(examples, heldout_ids=heldout_ids, heldout_families=heldout_families)
        accepted_counts = dict(Counter(r["partition"] for r in examples))
        if accepted_counts.get("train", 0) < 1 or accepted_counts.get("validation", 0) < 1:
            raise ValueError("At least one reviewed training and validation target required")
        files = [curriculum_path, source_path, settings_path, manifest_path, *paths]
        sources = [{"path": str(path.resolve()), "sha256": sha(path.read_bytes())} for path in files]
        source_data, review_data = jsonl(examples), jsonl(reviews)
        receipt = {
            "schema_version": 1, "purpose": "reviewed_iteration2_planned_scene_distillation",
            "source_files": sources, "images": images, "completed_plans": plans,
            "source_sha256": sha(source_data), "review_sha256": sha(review_data),
            "upload_sha256": sha(json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
            "entry_counts": accepted_counts, "generated_counts": {"train": 80, "validation": 24},
            "review_counts": dict(Counter(r["decision"] for r in reviews)),
            "source_generation_ids": [r["source_generation_id"] for r in examples],
            "eligible_family_counts": dict(Counter(r["prompt_family"] for r in examples)),
            "excluded": excluded, "heldout_ids": sorted(heldout_ids), "heldout_families": sorted(heldout_families),
            "teacher_model": TEACHER, "student_model": STUDENT_MODEL, "prompt_version": expected_version,
            "comparison_fingerprint": settings["comparison_fingerprint"],
            "training_request_policy": {"epochs": [1, 3], "batch_size": 2,
                                        "learning_rate": 1e-4, "assistant_turns": "all"},
        }
        outputs = {directory / "reviewed-manifest.jsonl": jsonl(bound),
                   directory / "reviewed-examples.jsonl": source_data,
                   directory / "review-decisions.jsonl": review_data,
                   directory / "excluded.json": encoded(excluded),
                   directory / "source-receipt.json": encoded(receipt)}
        for index, path in enumerate(files):
            outputs[directory / "sources" / f"{index:02d}-{path.name}"] = path.read_bytes()
        # Validate every frozen output before writing any newly absent output.
        for path, data in outputs.items():
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"Refusing changed frozen evidence: {path}")
        for path, data in outputs.items():
            freeze(path, data)
        return receipt


def candidate_spec(name, root, receipt, project):
    return {"schema_version": 1, "candidate": name, "base_model": STUDENT_MODEL,
            "display_model_id": f"ft-{STUDENT_MODEL}", "initialization": "independent_base_model",
            "model_name": f"bufo-gemma-iter2-{name}-20261016",
            "dataset_name": f"bufo-sol-iter2-{name}-20261016",
            "source_path": str(root / "training/reviewed-examples.jsonl"),
            "source_sha256": receipt["source_sha256"], "upload_sha256": receipt["upload_sha256"],
            "source_receipt_sha256": sha((root / "training/source-receipt.json").read_bytes()),
            "training_config": {"epochs": RECIPES[name], "batch_size": 2,
                                "learning_rate": 1e-4, "assistant_turns": "all"},
            "project": {key: project[key] for key in
                        ("entity", "project", "project_id", "project_alias", "base_url", "training_base_url")}}


def train_or_status(mode, root, client, project, *, candidates=None, transport_metadata=None):
    if mode not in {"train", "status"}:
        raise ValueError("Expected train or status")
    root = Path(root).resolve()
    names = list(RECIPES) if candidates is None else list(candidates)
    if not names or len(set(names)) != len(names) or set(names) - set(RECIPES):
        raise ValueError("Choose unique e1/e3 candidates")
    receipt = prepare(root)
    for field in ("entity", "project_alias", "base_url", "training_base_url"):
        if project[field] != getattr(client, field):
            raise ValueError("Training client differs from declared project")
    specs = {name: candidate_spec(name, root, receipt, project) for name in names}
    with _locked(root / "training/operation"):
        # Reconcile all requested arms before allowing any new remote mutation.
        for name, spec in specs.items():
            folder = root / "training" / name
            spec_path, state_path = folder / "candidate.json", folder / "forge-state.json"
            if spec_path.exists() and read(spec_path) != spec:
                raise ValueError("Frozen candidate recipe changed")
            if mode == "status" and (not spec_path.exists() or not state_path.exists()):
                raise ValueError("Cannot poll candidate without its saved workflow")
            if state_path.exists():
                state = read(state_path)
                _clear_or_block(state)
                for key in ("source_path", "source_sha256", "upload_sha256", "model_name", "dataset_name", "base_model"):
                    if state.get(key) != spec[key]:
                        raise ValueError(f"Existing candidate differs at {key}")
                if state.get("training_config") is not None and state["training_config"] != spec["training_config"]:
                    raise ValueError("Existing candidate training configuration changed")
        # Persist transport evidence before remote writes, including failed attempts.
        # This is operational history, not part of the frozen model recipe.
        transport_receipt = {
            "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
            "operation": mode, "candidates": names,
            "transport": transport_metadata if transport_metadata is not None else {"mode": "client_supplied"},
        }
        freeze(root / "training/transport-history" /
               f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex}.json", encoded(transport_receipt))
        results = {}
        for name, spec in specs.items():
            folder, state_path = root / "training" / name, root / "training" / name / "forge-state.json"
            if mode == "train":
                freeze(folder / "candidate.json", encoded(spec))
                initialize_workflow(state_path, client, source_path=spec["source_path"],
                    dataset_name=spec["dataset_name"], model_name=spec["model_name"],
                    project_id=project["project_id"], project=project["project"],
                    heldout_ids=receipt["heldout_ids"], heldout_families=receipt["heldout_families"],
                    lineage={"source_receipt_sha256": spec["source_receipt_sha256"],
                             "candidate_spec": str(folder / "candidate.json"),
                             "initialization": "independent_base_model"})
                state = import_reviewed_dataset(state_path, client)
                state = submit_training(state_path, client, epochs=RECIPES[name], batch_size=2, learning_rate=1e-4)
            else:
                state = resume_training(state_path, client)
            observed = {"observed_at": datetime.now(timezone.utc).isoformat(), "candidate": name,
                        **{key: state.get(key) for key in ("status", "training_progress", "finetune_id",
                            "dataset_id", "dataset_revision", "model_artifact_ref", "training_config",
                            "source_sha256", "upload_sha256", "entry_counts")}}
            freeze(folder / "history" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex}.json", encoded(observed))
            results[name] = observed
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--root", type=Path, default=DIRECTORY)
    parser.add_argument("--candidate", action="append", choices=list(RECIPES))
    parser.add_argument("--tls12", action="store_true",
                        help="Use verified TLS 1.2 for Forge upload compatibility; default transport is unchanged")
    parser.add_argument("--project-state", type=Path, default=ROOT / "runs/forge-loop/forge-state.json")
    args = parser.parse_args()
    if not args.train and not args.status:
        print(json.dumps(prepare(args.root), indent=2))
        return
    project = read(args.project_state)
    transport, transport_metadata = transport_for_cli(args.tls12)
    with ForgeClient(api_key=required("WANDB_API_KEY"), entity=project["entity"],
            project_alias=project["project_alias"], base_url=project["base_url"],
            training_base_url=project["training_base_url"], timeout=120, transport=transport) as client:
        if args.train:
            client.preflight(STUDENT_MODEL, project=project["project"], project_id=project["project_id"])
        result = train_or_status("train" if args.train else "status", args.root, client, project,
                                 candidates=args.candidate, transport_metadata=transport_metadata)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
