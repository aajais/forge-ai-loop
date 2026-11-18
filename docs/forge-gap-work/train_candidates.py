"""Freeze reviewed real trajectories and run two independent managed Forge SFT jobs.

Default is local preparation only. --submit authorizes the two durable import/job
workflows; --poll performs reads for existing jobs only. Never replay an ambiguous
write or mutate a serving route. Each candidate imports identical source bytes.
"""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bufogen.evaluation.benchmark import _validate_rows
from bufogen.curation.curate import _validate_review
from bufogen.curation.curate import select_examples
from bufogen.integrations.forge import STUDENT_MODEL
from bufogen.integrations.forge import ForgeClient
from bufogen.integrations.forge import reviewed_entries
from bufogen.workflows.forge_loop import _locked
from bufogen.workflows.forge_loop import import_reviewed_dataset
from bufogen.workflows.forge_loop import initialize_workflow
from bufogen.workflows.forge_loop import resume_training
from bufogen.workflows.forge_loop import submit_training
from bufogen.generation import Manifest
from bufogen.settings import ROOT
from bufogen.settings import required

DIRECTORY = ROOT / "runs/forge-gap/training"
TEACHER = ROOT / "runs/forge-gap/teacher-curriculum"
REVIEW_PATHS = [
    ROOT / f"runs/forge-gap/review/{name}/decisions.jsonl"
    for name in ("reviewer-a", "reviewer-b", "reviewer-root")
]
ADJUDICATION_PATHS = [path.with_name("adjudications.jsonl") for path in REVIEW_PATHS]
PROMPT_PATHS = [ROOT / "config/forge_prompts.yaml", ROOT / "config/forge_gap_prompts.yaml"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def jsonl(rows):
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()


def atomic(path, data, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to replace frozen evidence: {path}")
        return
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


def prepare():
    """Validate full review/source coverage and preserve exact, eligible captures."""
    with _locked(DIRECTORY / "prepare.lock"):
        settings = json.loads((TEACHER / "benchmark.json").read_text())
        rows = sorted(Manifest(TEACHER / "manifest.jsonl").read(), key=lambda r: r.prompt_id)
        _validate_rows(rows, settings)
        expected = {p["id"]: p for p in settings["configuration"]["prompts"]}
        if (
            len(rows) != 152
            or len({r.id for r in rows}) != len(rows)
            or {r.prompt_id for r in rows} != set(expected)
        ):
            raise ValueError("Require complete unique 152-prompt teacher generation coverage")
        by_id = {row.id: row for row in rows}
        reviews = [
            json.loads(line)
            for path in REVIEW_PATHS
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if len(reviews) != len(rows) or {r["source_generation_id"] for r in reviews} != set(by_id):
            raise ValueError("Require exactly one explicit review per generation")
        original_reviews = reviews
        effective = {r["source_generation_id"]: r for r in reviews}
        adjudications, seen_adjudications = [], set()
        for path in ADJUDICATION_PATHS:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                review = json.loads(line)
                generation_id = review["source_generation_id"]
                if generation_id not in effective or generation_id in seen_adjudications:
                    raise ValueError("Unknown or repeated adjudication generation")
                prior = effective[generation_id]
                _validate_review(review, by_id[generation_id])
                if not (
                    review.get("adjudication") is True
                    or review.get("review_stage") == "judge_disagreement_adjudication"
                ):
                    raise ValueError("Adjudication requires an explicit review-stage marker")
                for field in (
                    "prompt_id",
                    "partition",
                    "prompt_family",
                    "png_sha256",
                    "source_ref",
                ):
                    if review[field] != prior[field]:
                        raise ValueError(f"Adjudication changed source identity: {field}")
                for field, prior_field in (
                    ("prior_decision", "decision"),
                    ("prior_reviewer_id", "reviewer_id"),
                ):
                    if field in review and review[field] != prior[prior_field]:
                        raise ValueError("Adjudication prior-review identity mismatch")
                adjudications.append(
                    {
                        "review": review,
                        "prior_review": prior,
                        "prior_review_sha256": sha(encoded(prior)),
                    }
                )
                effective[generation_id] = review
                seen_adjudications.add(generation_id)
        reviews = list(effective.values())
        bound, image_receipts = [], []
        for review in sorted(reviews, key=lambda r: r["prompt_id"]):
            row = by_id[review["source_generation_id"]]
            _validate_review(review, row)
            prompt = expected[row.prompt_id]
            if (
                row.model != "gpt-6.1-sol"
                or review["partition"] != prompt["split"]
                or review["prompt_family"] != prompt["family"]
            ):
                raise ValueError("Review/model identity differs from frozen prompt source")
            if (
                row.prompt != prompt["prompt"]
                or row.benchmark_metadata["requirements"] != prompt["requirements"]
            ):
                raise ValueError("Generation prompt/requirements differ from frozen curriculum")
            image_path = Path(row.png_path).resolve()
            image_hash = sha(image_path.read_bytes())
            if (
                image_hash != review["png_sha256"]
                or Path(review["source_ref"]).resolve() != image_path
            ):
                raise ValueError("Review PNG source or hash differs from actual generation")
            if review.get("visual_quality_pass") != (review["decision"] == "accept"):
                raise ValueError("Review visual pass and decision disagree")
            bound.append(replace(row, training_review=review))
            image_receipts.append(
                {
                    "generation_id": row.id,
                    "prompt_id": row.prompt_id,
                    "png_path": str(image_path),
                    "png_sha256": image_hash,
                }
            )
        heldout = [
            p
            for path in PROMPT_PATHS
            for p in yaml.safe_load(path.read_text())
            if p["split"] in {"heldout", "test"}
        ]
        heldout_ids, heldout_families = {p["id"] for p in heldout}, {p["family"] for p in heldout}
        diagnostics = []
        examples = select_examples(
            bound,
            heldout_ids=heldout_ids,
            heldout_families=heldout_families,
            allowed_partitions={"train", "validation"},
            diagnostics=diagnostics,
        )
        examples.sort(key=lambda r: r["prompt_id"])
        entries = reviewed_entries(
            examples, heldout_ids=heldout_ids, heldout_families=heldout_families
        )
        counts = dict(Counter(r["partition"] for r in examples))
        if counts.get("train", 0) < 80 or counts.get("validation", 0) < 1:
            raise ValueError("Insufficient eligible reviewed examples; inspect diagnostics")
        files = [
            TEACHER / "manifest.jsonl",
            TEACHER / "benchmark.json",
            *REVIEW_PATHS,
            *ADJUDICATION_PATHS,
            *PROMPT_PATHS,
            ROOT / "runs/forge-gap/curriculum-manifest.json",
        ]
        sources = []
        for path in files:
            relative = path.relative_to(ROOT)
            data = path.read_bytes()
            atomic(DIRECTORY / "sources" / relative, data, immutable=True)
            sources.append({"path": str(relative), "sha256": sha(data)})
        review_data = jsonl(sorted(reviews, key=lambda r: r["prompt_id"]))
        source_data = jsonl(examples)
        atomic(
            DIRECTORY / "original-review-decisions.jsonl",
            jsonl(sorted(original_reviews, key=lambda r: r["prompt_id"])),
            immutable=True,
        )
        atomic(DIRECTORY / "adjudications.jsonl", jsonl(adjudications), immutable=True)
        atomic(DIRECTORY / "review-decisions.jsonl", review_data, immutable=True)
        atomic(
            DIRECTORY / "reviewed-manifest.jsonl",
            jsonl([r.to_dict() for r in bound]),
            immutable=True,
        )
        atomic(DIRECTORY / "reviewed-examples.jsonl", source_data, immutable=True)
        atomic(DIRECTORY / "excluded.json", encoded(diagnostics), immutable=True)
        receipt = {
            "schema_version": 1,
            "source_files": sources,
            "images": image_receipts,
            "source_sha256": sha(source_data),
            "review_sha256": sha(review_data),
            "upload_sha256": sha(
                json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ),
            "source_generation_ids": [r["source_generation_id"] for r in examples],
            "entry_counts": counts,
            "review_counts": dict(Counter(r["decision"] for r in reviews)),
            "adjudication_count": len(adjudications),
            "adjudication_decision_changes": sum(
                a["review"]["decision"] != a["prior_review"]["decision"] for a in adjudications
            ),
            "eligible_family_counts": dict(Counter(r["prompt_family"] for r in examples)),
            "excluded": diagnostics,
            "heldout_ids": sorted(heldout_ids),
            "heldout_families": sorted(heldout_families),
            "assistant_turns": "all",
            "teacher_model": "gpt-6.1-sol",
            "student_model": STUDENT_MODEL,
            "comparison_fingerprint": settings["comparison_fingerprint"],
            "training_request_policy": {
                "epochs": [1, 3],
                "batch_size": 2,
                "assistant_turns": "all",
                "learning_rate": "service default",
            },
        }
        atomic(DIRECTORY / "source-receipt.json", encoded(receipt), immutable=True)
        return receipt


def run(mode):
    receipt = prepare()
    print(
        json.dumps(
            {
                "source": str(DIRECTORY / "reviewed-examples.jsonl"),
                "source_sha256": receipt["source_sha256"],
                "counts": receipt["entry_counts"],
            }
        ),
        flush=True,
    )
    if mode == "prepare":
        return
    project = json.loads((ROOT / "runs/forge-loop/forge-state.json").read_text())
    with ForgeClient(
        api_key=required("WANDB_API_KEY"),
        entity=project["entity"],
        project_alias=project["project_alias"],
        base_url=project["base_url"],
        training_base_url=project["training_base_url"],
        timeout=120,
    ) as client:
        if mode == "submit":
            preflight = client.preflight(
                STUDENT_MODEL, project=project["project"], project_id=project["project_id"]
            )
            atomic(DIRECTORY / "preflight.json", encoded(preflight))
        for epochs in (1, 3):
            candidate = DIRECTORY / f"e{epochs}"
            state_path = candidate / "forge-state.json"
            if mode == "submit":
                state = initialize_workflow(
                    state_path,
                    client,
                    source_path=DIRECTORY / "reviewed-examples.jsonl",
                    dataset_name=f"bufo-sol-gap-e{epochs}-20261015",
                    model_name=f"bufo-gemma-gap-e{epochs}-20261015",
                    project_id=project["project_id"],
                    project=project["project"],
                    heldout_ids=receipt["heldout_ids"],
                    heldout_families=receipt["heldout_families"],
                    lineage={
                        "source_receipt": str(DIRECTORY / "source-receipt.json"),
                        "source_receipt_sha256": sha(
                            (DIRECTORY / "source-receipt.json").read_bytes()
                        ),
                        "reviewer_kinds": ["assistant"],
                        "selection": "validation before confirmation",
                    },
                )
                state = import_reviewed_dataset(state_path, client)
                if (
                    state["source_sha256"] != receipt["source_sha256"]
                    or state["upload_sha256"] != receipt["upload_sha256"]
                ):
                    raise ValueError("Candidate source/upload differs from frozen shared dataset")
                state = submit_training(state_path, client, epochs=epochs, batch_size=2)
            else:
                state = resume_training(state_path, client)
            expected_config = {"epochs": epochs, "batch_size": 2, "assistant_turns": "all"}
            if state.get("training_config") != expected_config:
                raise ValueError("Saved training configuration differs from candidate")
            receipt_out = {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                **{
                    key: state.get(key)
                    for key in (
                        "entity",
                        "project",
                        "project_alias",
                        "dataset_id",
                        "dataset_revision",
                        "finetune_id",
                        "model_name",
                        "base_model",
                        "training_config",
                        "model_artifact_ref",
                        "dataset_artifact_ref",
                        "source_sha256",
                        "upload_sha256",
                        "entry_counts",
                        "status",
                        "training_progress",
                    )
                },
            }
            atomic(candidate / "latest.json", encoded(receipt_out))
            print(json.dumps(receipt_out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--submit", action="store_true")
    options.add_argument("--poll", action="store_true")
    args = parser.parse_args()
    run("submit" if args.submit else "poll" if args.poll else "prepare")
