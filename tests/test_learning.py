import pytest

from bufogen.curation.curate import select_examples
from bufogen.evaluation.report import compare_themes
from bufogen.generation import BufoGeneration
from bufogen.training.common import reward, validate_adapter


def row(backend="open_base", valid=True, theme="cloud", score=0.8, **kwargs):
    return BufoGeneration(
        prompt_id="p",
        prompt="A frog on a cloud",
        backend=backend,
        model="m",
        group_id="g",
        svg="<svg/>",
        valid=valid,
        latency_s=1,
        theme=theme,
        scores={
            "svg_validity": {"valid": valid},
            "clip_score": {"similarity": 0.3},
            "vlm_judge": {
                "green_bufo": score,
                "blue_accent": score,
                "theme_match": score,
                "visual_quality": score,
                "overall": score,
            },
        },
        **kwargs,
    )


def test_curation_gates_and_deduplicates_without_crossing_eval_split():
    from test_training_contract import review
    from test_trajectories import captured_row

    def qualified(**kwargs):
        result = row(**kwargs)
        result.trajectory = captured_row().trajectory
        result.prompt_version = "v1"
        result.trajectory["provenance"]["prompt_id"] = result.prompt_id
        result.training_review = review(result)
        return result

    selected = select_examples(
        [qualified(), qualified(), qualified(valid=False), qualified(score=0.1)], heldout_ids=set()
    )
    assert len(selected) == 1
    assert selected[0]["messages"][-1]["content"] == "Here is Bufo!"
    assert selected[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "render_svg"
    assert select_examples([qualified()], heldout_ids={"p"}) == []


def test_legacy_judge_scores_do_not_earn_scene_reward():
    from bufogen.evaluation.scorers.scene_quality import SceneQuality

    assert reward(row(valid=False), scorer_configuration=SceneQuality().configuration()) == 0
    assert reward(row(), scorer_configuration=SceneQuality().configuration()) == 0


def test_adapter_contract_rejects_high_rank_and_wrong_base(tmp_path):
    import json

    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({"r": 32, "base_model_name_or_path": "base"}))
    with pytest.raises(ValueError, match="rank"):
        validate_adapter(tmp_path, "base")
    config.write_text(json.dumps({"r": 16, "base_model_name_or_path": "wrong"}))
    with pytest.raises(ValueError, match="base"):
        validate_adapter(tmp_path, "base")


def test_report_pairs_prompts_and_reports_missing_evidence():
    base, tuned = row(), row(backend="open_tuned", score=0.9)
    report = compare_themes([base, tuned], weak_themes={"cloud", "text"})
    assert "cloud" in report and "+0.100" in report
    assert "text" in report and "No paired evidence" in report


def test_validation_partition_requires_explicit_opt_in_and_never_allows_test():
    from test_training_contract import review
    from test_trajectories import captured_row

    result = captured_row()
    result.training_review = review(result, partition="validation")
    assert select_examples([result], heldout_ids=set()) == []
    selected = select_examples(
        [result], heldout_ids=set(), allowed_partitions={"train", "validation"}
    )
    assert selected[0]["partition"] == "validation"
    with pytest.raises(ValueError, match="partitions"):
        select_examples([result], heldout_ids=set(), allowed_partitions={"test"})
