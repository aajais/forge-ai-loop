import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from bufogen.curation.trajectories import export_training_example
from bufogen.generation import BufoGeneration, Manifest
from bufogen.integrations.registry import RegistryClient
from bufogen.settings import ROOT


def _validate_review(review: dict, row: BufoGeneration) -> None:
    if not isinstance(review, dict) or review.get("schema_version") != 1:
        raise ValueError("missing or unsupported training review")
    if review.get("decision") not in {"accept", "reject"}:
        raise ValueError("training review requires an explicit accept/reject decision")
    if review.get("reviewer_kind") not in {"human", "aria_reviewed", "assistant", "astra_judge"}:
        raise ValueError(
            "training review must identify a human, ARIA-reviewed, assistant, or Astra judge reviewer"
        )
    for key in ("reviewer_id", "source_ref", "evidence", "prompt_family"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise ValueError(f"training review is missing {key} provenance")
    if review.get("source_generation_id") != row.id or review.get("prompt_id") != row.prompt_id:
        raise ValueError("training review source identifiers do not match generation")
    if review.get("partition") not in {"train", "validation", "test", "heldout"}:
        raise ValueError("training review is missing an explicit dataset partition")
    try:
        reviewed_at = datetime.fromisoformat(review["reviewed_at"])
        if reviewed_at.tzinfo is None:
            raise ValueError("timezone required")
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("training review requires a timestamp with timezone") from exc


def import_review_decisions(rows: list[BufoGeneration], path: Path) -> list[BufoGeneration]:
    """Join explicit reviewer decisions by generation ID; never infer acceptance from scores.

    The JSONL decision schema is validated before any rows are returned. Unknown
    and duplicate IDs fail closed so a mismatched ARIA export cannot silently qualify data.
    Original rows are unchanged; append returned rows to a new manifest if needed.
    """
    by_id = {row.id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("review import requires unique generation IDs")
    reviews = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        review = json.loads(line)
        if not isinstance(review, dict):
            raise ValueError("training review must be an object")
        source_id = review.get("source_generation_id")
        if source_id not in by_id or source_id in reviews:
            raise ValueError("training review references an unknown or duplicate generation ID")
        _validate_review(review, by_id[source_id])
        reviews[source_id] = review
    return [replace(row, training_review=reviews.get(row.id, row.training_review)) for row in rows]


def select_examples(
    rows: list[BufoGeneration],
    *,
    heldout_ids: set[str],
    min_judge=None,
    min_clip=None,
    heldout_families: set[str] | None = None,
    diagnostics: list[dict] | None = None,
    allowed_partitions: set[str] | None = None,
) -> list[dict]:
    allowed_partitions = {"train"} if allowed_partitions is None else allowed_partitions
    if not allowed_partitions or not allowed_partitions <= {"train", "validation"}:
        raise ValueError("allowed_partitions must contain only train and/or validation")
    selected, seen = [], set()
    for row in sorted(
        rows, key=lambda r: r.scores.get("vlm_judge", {}).get("overall", 0), reverse=True
    ):
        try:
            _validate_review(row.training_review, row)
            review = row.training_review
            if review["decision"] != "accept":
                raise ValueError("training review rejected this generation")
            if (
                row.prompt_id in heldout_ids
                or review["partition"] not in allowed_partitions
                or review["prompt_family"] in (heldout_families or set())
            ):
                raise ValueError("held-out prompt, family, or non-training partition")
            if not row.valid:
                raise ValueError("invalid render")
            if (
                min_judge is not None
                and row.scores.get("vlm_judge", {}).get("overall", 0) < min_judge
            ) or (
                min_clip is not None
                and row.scores.get("clip_score", {}).get("similarity", 0) < min_clip
            ):
                raise ValueError("automatic quality thresholds were not met")
            example = export_training_example(row)
            key = hashlib.sha256((row.prompt.strip() + row.svg.strip()).encode()).hexdigest()
            if key in seen:
                raise ValueError("duplicate prompt and drawing")
            seen.add(key)
            example.update(
                {
                    "origin": row.origin
                    if row.origin != "generation"
                    else (
                        "distillation"
                        if row.backend == "closed_quality_bar"
                        else "self-improvement"
                    ),
                    "scores": deepcopy(row.scores),
                    "training_review": deepcopy(review),
                    "prompt_family": review["prompt_family"],
                    "partition": review["partition"],
                }
            )
            selected.append(example)
        except ValueError as exc:
            diagnostic = {"source_generation_id": row.id, "reason": str(exc)}
            if diagnostics is not None:
                diagnostics.append(diagnostic)
            logging.getLogger(__name__).info("Skipped generation %s: %s", row.id, exc)
    return selected


async def curate(
    *,
    prompts_ref: str,
    backup_ref: str | None,
    output_dir: Path,
    heldout_ids: set[str],
    review_decisions: Path | None = None,
    heldout_families: set[str] | None = None,
) -> str:
    from bufogen.evaluation.suite import evaluate_manifest
    from bufogen.workflows.pipeline import load_prompts, run_pipeline

    registry = RegistryClient()
    prompts_dir = registry.download(prompts_ref)
    prompts = load_prompts(prompts_dir / "prompts.yaml")
    if any(p["id"] in heldout_ids for p in prompts):
        raise ValueError("ARIA prompts overlap held-out IDs; regenerate disjoint training prompts.")
    rows = await run_pipeline(prompts, ["open_base", "closed_quality_bar"], output_dir)
    sources = [prompts_ref]
    if backup_ref:
        backup_dir = registry.download(backup_ref)
        rows.extend(Manifest(backup_dir / "scored.jsonl").read())
        sources.append(backup_ref)
    rows = await evaluate_manifest(rows, output_dir / "eval")
    if review_decisions is not None:
        rows = import_review_decisions(rows, review_decisions)
    diagnostics = []
    examples = select_examples(
        rows, heldout_ids=heldout_ids, heldout_families=heldout_families, diagnostics=diagnostics
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "curation-diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    if not examples:
        raise ValueError(
            "No reviewed examples passed curation. Import explicit review decisions and inspect "
            "curation-diagnostics.json; automatic scores alone cannot qualify SFT data."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in examples))
    registry.push_prompts(sorted((ROOT / "aria_prompts").glob("*.md")), sources=[prompts_ref])
    return registry.push_dataset(
        path,
        sources=sources,
        metadata={"examples": len(examples), "heldout_ids": sorted(heldout_ids)},
    )
