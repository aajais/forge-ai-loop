import importlib.util
from types import SimpleNamespace

from bufogen.settings import ROOT


def entrypoint():
    spec = importlib.util.spec_from_file_location(
        "sft_entrypoint", ROOT / "scripts/train_entrypoints/sft_entrypoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sft_targets_only_language_attention_not_vision_wrappers():
    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.vision_tower.layers.0.self_attn.q_proj",
        "model.vision_tower.layers.0.self_attn.q_proj.linear",
        "model.audio_tower.layers.0.self_attn.o_proj",
    ]
    model = SimpleNamespace(named_modules=lambda: [(name, object()) for name in names])
    assert entrypoint().lora_targets(model, multimodal=True) == names[:2]


def test_adapter_requires_real_weight_file(tmp_path):
    import json

    import pytest

    from bufogen.training.common import validate_adapter

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 8, "base_model_name_or_path": "base"})
    )
    with pytest.raises(ValueError, match="safetensors"):
        validate_adapter(tmp_path, "base")


def test_automatic_scores_cannot_qualify_training_examples():
    from bufogen.curation.curate import select_examples
    from test_trajectories import captured_row

    row = captured_row()
    row.scores = {
        "svg_validity": {"valid": True},
        "vlm_judge": {"overall": 1},
        "clip_score": {"similarity": 1},
    }
    diagnostics = []
    assert select_examples([row], heldout_ids=set(), diagnostics=diagnostics) == []
    assert "review" in diagnostics[0]["reason"]


def review(row, **overrides):
    return {
        "schema_version": 1,
        "source_generation_id": row.id,
        "prompt_id": row.prompt_id,
        "decision": "accept",
        "reviewer_kind": "human",
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-10-15T14:00:00+00:00",
        "source_ref": "aria://reviews/123",
        "evidence": "Inspected the rendered image; recognizable Bufo and correct theme.",
        "partition": "train",
        "prompt_family": "cloud-scenes",
        **overrides,
    }


def test_review_import_and_heldout_family_gating(tmp_path):
    import json

    from bufogen.curation.curate import import_review_decisions, select_examples
    from test_trajectories import captured_row

    row = captured_row()
    row.scores = {
        "svg_validity": {"valid": True},
        "vlm_judge": {"overall": 1},
        "clip_score": {"similarity": 1},
    }
    path = tmp_path / "reviews.jsonl"
    path.write_text(json.dumps(review(row)) + "\n")
    reviewed = import_review_decisions([row], path)
    assert row.training_review is None
    examples = select_examples(reviewed, heldout_ids=set())
    assert len(examples) == 1
    assert examples[0]["training_review"]["source_ref"] == "aria://reviews/123"
    assert examples[0]["messages"][-1]["content"] == "Here is Bufo!"
    assert select_examples(reviewed, heldout_ids={"p"}) == []
    assert select_examples(reviewed, heldout_ids=set(), heldout_families={"cloud-scenes"}) == []


def test_missing_negative_and_auto_only_reviews_are_denied():
    from bufogen.curation.curate import select_examples
    from test_trajectories import captured_row

    row = captured_row()
    row.scores = {
        "svg_validity": {"valid": True},
        "vlm_judge": {"overall": 1},
        "clip_score": {"similarity": 1},
    }
    for overrides in (
        {"decision": "reject"},
        {"reviewer_kind": "automatic"},
        {"partition": "test"},
        {"source_ref": ""},
        {"evidence": ""},
        {"source_generation_id": "someone-else"},
        {"reviewed_at": "bad"},
    ):
        row.training_review = review(row, **overrides)
        assert select_examples([row], heldout_ids=set()) == []
    row.training_review = review(row, reviewer_kind="assistant")
    assert len(select_examples([row], heldout_ids=set())) == 1


def test_reviewed_legacy_rows_are_skipped_with_diagnostics():
    from bufogen.curation.curate import select_examples
    from test_trajectories import captured_row

    row = captured_row()
    row.trajectory = None
    row.training_review = review(row)
    row.scores = {
        "svg_validity": {"valid": True},
        "vlm_judge": {"overall": 1},
        "clip_score": {"similarity": 1},
    }
    diagnostics = []
    assert select_examples([row], heldout_ids=set(), diagnostics=diagnostics) == []
    assert "trajectory" in diagnostics[0]["reason"]


def test_explicit_review_does_not_require_legacy_automatic_metrics():
    from bufogen.curation.curate import select_examples
    from test_trajectories import captured_row

    row = captured_row()
    row.training_review = review(row, reviewer_kind="assistant")
    row.scores = {}
    assert len(select_examples([row], heldout_ids=set())) == 1
    assert select_examples([row], heldout_ids=set(), min_judge=0.7) == []
    assert select_examples([row], heldout_ids=set(), min_clip=0.2) == []
