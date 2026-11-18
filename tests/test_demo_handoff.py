import json

import pytest

from bufogen.workflows.demo_handoff import reviewed_rows, validate_checkpoint


def sample():
    return {
        "source_generation_id": "g1",
        "source_group_id": "conversation:1",
        "source_model": "gpt-6.1-sol",
        "prompt_id": "train-1",
        "prompt": "Draw Bufo",
        "partition": "train",
        "prompt_family": "concert",
        "origin": "distillation",
        "training_review": {
            "source_generation_id": "g1",
            "decision": "accept",
            "reviewer_kind": "astra_judge",
            "evidence": "Bufo is visible",
        },
    }


def test_lineage_retains_actual_reviewer_and_clickable_source(tmp_path):
    path = tmp_path / "accepted.jsonl"
    path.write_text(json.dumps(sample()) + "\n")
    rows = reviewed_rows(path, "team/project")
    assert rows[0]["reviewer_kind"] == "astra_judge"
    assert rows[0]["origin"] == "distillation"
    assert "conversation%3A1" in rows[0]["source_conversation_url"]
    assert "messages" not in rows[0]


@pytest.mark.parametrize(
    "change",
    [
        {"partition": "heldout"},
        {"training_review": {"decision": "reject", "source_generation_id": "g1"}},
        {"training_review": {"decision": "accept", "source_generation_id": "wrong"}},
    ],
)
def test_handoff_refuses_unqualified_rows(tmp_path, change):
    path = tmp_path / "accepted.jsonl"
    path.write_text(json.dumps(sample() | change) + "\n")
    with pytest.raises(ValueError):
        reviewed_rows(path, "team/project")


@pytest.mark.parametrize(
    "field,value",
    [
        ("forge_dataset_id", "other"),
        ("base_model", "other"),
        ("checkpoint_version", "v2"),
        ("sft_config", {"epochs": 3}),
    ],
)
def test_release_refuses_mixed_checkpoint_lineage(field, value):
    state = {
        "model_artifact_ref": "wandb-artifact:///team/project/model:v1",
        "finetune_id": "job",
        "dataset_id": "dataset",
        "base_model": "Qwen/model",
        "training_config": {"epochs": 1},
    }
    checkpoint = {
        "artifact": state["model_artifact_ref"],
        "forge_job_id": "job",
        "forge_dataset_id": "dataset",
        "base_model": "Qwen/model",
        "checkpoint_version": "v1",
        "sft_config": {"epochs": 1},
    }
    validate_checkpoint(state, checkpoint)
    with pytest.raises(ValueError):
        validate_checkpoint(state, checkpoint | {field: value})
