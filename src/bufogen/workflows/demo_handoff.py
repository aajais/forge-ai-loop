"""Prepare inspectable curation/release evidence. Never trains or changes traffic."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from bufogen.settings import ROOT


def validate_checkpoint(state: dict, checkpoint: dict) -> None:
    expected = {
        "artifact": state["model_artifact_ref"],
        "forge_job_id": state["finetune_id"],
        "forge_dataset_id": state["dataset_id"],
        "base_model": state["base_model"],
        "checkpoint_version": state["model_artifact_ref"].rsplit(":", 1)[1],
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()) or any(
        checkpoint.get("sft_config", {}).get(key) != value
        for key, value in state["training_config"].items()
    ):
        raise ValueError("Checkpoint lineage differs from the submitted Forge job")


def reviewed_rows(path: Path, project: str) -> list[dict]:
    """Export review provenance, not messages/tool targets or inferred human approval."""
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        source = json.loads(line)
        review = source.get("training_review") or {}
        generation_id = source["source_generation_id"]
        if (
            source.get("partition") != "train"
            or review.get("decision") != "accept"
            or review.get("source_generation_id") != generation_id
            or not review.get("reviewer_kind")
            or generation_id in seen
        ):
            raise ValueError("Handoff requires unique accepted train rows with identified reviews")
        seen.add(generation_id)
        group = source.get("source_group_id")
        rows.append(
            {
                "source_generation_id": generation_id,
                "source_conversation_url": (
                    f"https://forge.coreweave.com/wandb/{project}/weave/agents?dashConv="
                    + quote(group, safe="")
                    if group
                    else None
                ),
                "prompt_id": source["prompt_id"],
                "prompt": source["prompt"],
                "family": source.get("prompt_family"),
                "source_model": source.get("source_model"),
                "origin": source.get("origin"),
                "reviewer_kind": review["reviewer_kind"],
                "reviewer_id": review.get("reviewer_id"),
                "decision": review["decision"],
                "reviewed_at": review.get("reviewed_at"),
                "review_source": review.get("source_ref"),
                "review_evidence": review.get("evidence"),
            }
        )
    return rows


def prepare(root: Path = ROOT) -> dict:
    """Describe the evaluated one-epoch checkpoint; no automatic winner/promotion."""
    directory = root / "runs/forge-demo"
    source = root / "runs/distill/stage-b/export/accepted.jsonl"
    state = json.loads((root / "runs/distill/stage-b/sweep/forge-state-sft_q3.json").read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != state["source_sha256"]:
        raise ValueError("Training export no longer matches the submitted Forge source digest")
    project = f"{state['entity']}/{state['project']}"
    rows = reviewed_rows(source, project)
    if len(rows) != state["entry_counts"]["train"]:
        raise ValueError("Training row count differs from the submitted Forge job")
    checkpoints = json.loads((root / "runs/distill/stage-b/checkpoints.json").read_text())
    matches = [c for c in checkpoints if c["artifact"] == state["model_artifact_ref"]]
    if len(matches) != 1 or matches[0]["forge_job_id"] != state["finetune_id"]:
        raise ValueError("Checkpoint lineage differs from the Forge job")
    validate_checkpoint(state, matches[0])
    comparison = json.loads((directory / "comparison/comparison.json").read_text())
    candidates = [a for a in comparison["arms"] if a["model_id"] == state["model_artifact_ref"]]
    if len(candidates) != 1:
        raise ValueError("The candidate is not uniquely represented in the comparison")
    candidate = candidates[0]
    handoff = {
        "status": "prepared_not_promoted",
        "project": project,
        "candidate": state["model_artifact_ref"],
        "model_label": "ft-" + state["base_model"],
        "training_config": matches[0]["sft_config"],
        "training_rows": len(rows),
        "training_source_sha256": digest,
        "training_dataset": state["dataset_artifact_ref"],
        "reviewer_counts": dict(Counter(r["reviewer_kind"] for r in rows)),
        "origin_counts": dict(Counter(r["origin"] for r in rows)),
        "forge_job_url": (
            f"https://forge.coreweave.com/post-train/distillation/e/{state['entity']}"
            f"/p/{state['project']}/models/{state['finetune_id']}"
        ),
        "evaluation_url": candidate["evaluation_url"],
        "comparison_evidence_id": comparison["evidence_id"],
        "validation_summary": candidate["summary"],
        "rollback": {"model": state["base_model"], "mode": "direct studio backend qwen_base"},
        "routing": "unchanged; capture live routing before any routing deployment",
        "registry_status": "not_verified",
        "claims": {
            "production_trace_training": False,
            "human_reviewed_training": False,
            "sol_parity": False,
            "heldout_confirmation_for_candidate": False,
        },
        "notes": [
            "This is a validation-backed demo candidate, not an automatic production winner.",
            f"The {len(rows)} rows are judge-reviewed distillation examples, not human-reviewed production failures.",
            "Use an independent production failure review/export for the next iteration; do not relabel this dataset.",
            "Historical comparison has unrecorded client timeout overrides; see comparison caveats.",
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "curation-lineage.json").write_text(json.dumps(rows, indent=2) + "\n")
    (directory / "release-candidate.json").write_text(json.dumps(handoff, indent=2) + "\n")
    return handoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = prepare()
    print(json.dumps({k: result[k] for k in ("status", "candidate", "training_rows")}))


if __name__ == "__main__":
    main()
