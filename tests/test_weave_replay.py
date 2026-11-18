"""Shared prompt dataset + per-arm replay model for Weave evaluations."""

from types import SimpleNamespace

from bufogen.evaluation.weave_replay import (
    ReplayArm,
    shared_dataset_name,
    shared_dataset_rows,
    split_of,
)


def _row(prompt_id, backend, family="f", split="validation"):
    return SimpleNamespace(
        prompt_id=prompt_id,
        prompt=f"draw {prompt_id}",
        backend=backend,
        benchmark_metadata={"family": family, "split": split, "requirements": ["a", "b"]},
    )


def test_shared_rows_are_identical_across_arms_and_sorted():
    left = [_row("p2", "sft"), _row("p1", "sft")]
    right = [_row("p1", "base"), _row("p2", "base")]
    assert shared_dataset_rows(left) == shared_dataset_rows(right)
    assert [r["prompt_id"] for r in shared_dataset_rows(left)] == ["p1", "p2"]
    assert set(shared_dataset_rows(left)[0]) == {
        "prompt_id",
        "family",
        "split",
        "prompt",
        "requirements",
    }
    assert shared_dataset_name(split_of(left), len(left)) == "bufo-validation-2"
    assert split_of([_row("p1", "x"), _row("p2", "x", split="train")]) == "mixed"


def test_replay_arm_returns_saved_generation_and_hides_it_from_fields(monkeypatch):
    import bufogen.evaluation.weave_replay as module

    monkeypatch.setattr(
        module,
        "visual_generation",
        lambda prompt, generation: {"generation": generation, "prompt": prompt},
    )
    arm = ReplayArm(
        arm="sft_q3",
        model_label="ft-Qwen/Qwen3.8-27B",
        checkpoint="wandb-artifact:///e/p/m:v1",
        generations={"p1": {"id": "g1"}, "p2": {"id": "g2"}},
    )
    assert arm.predict(prompt="draw p2", prompt_id="p2") == {
        "generation": {"id": "g2"},
        "prompt": "draw p2",
    }
    assert "generations" not in arm.model_dump() and "_generations" not in arm.model_dump()
    assert arm.model_dump()["checkpoint"].endswith(":v1")
