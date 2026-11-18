import json

import pytest

from bufogen.cli import dispatch, parser
from bufogen.curation.judge_reviews import decision_from_row, write_decisions
from bufogen.evaluation.scorers.astra_judge import AstraReferenceJudge
from bufogen.generation import BufoGeneration, Manifest

KEY = ("bufo-astra-v5", 3)


def verdict(**overrides):
    config = AstraReferenceJudge(rubric_version=KEY[0], samples=KEY[1]).configuration()
    config["calibration_status"] = "uncalibrated"  # as stored before calibration existed
    data = {
        "status": "ok",
        "strict_pass": True,
        "judge_confidence": 0.9,
        "reason": "Recognizable Bufo satisfying every requirement.",
        "vs_reference": "equal",
        "rubric_version": KEY[0],
        "scorer_configuration": config,
    }
    return {**data, **overrides}


def scored_row(split="train", family="cloud-scenes", completed=True, **judge):
    row = BufoGeneration(
        prompt_id=f"{split}-{family}-01",
        prompt="Draw Bufo",
        backend="b",
        model="m",
        group_id=f"{family}-group",
        svg="<svg/>",
        valid=True,
        latency_s=1.0,
    )
    row.scores = {"astra_judge": verdict(**judge)}
    row.benchmark_metadata = {
        "split": split,
        "family": family,
        "status": "completed" if completed else "failed",
    }
    return row


def decide(row):
    return decision_from_row(
        row,
        registry_key=KEY,
        source_ref="runs/x/scored.jsonl",
        reviewed_at="2026-10-22T00:00:00+00:00",
    )


def test_decision_accepts_strict_pass_with_confidence():
    review = decide(scored_row())
    assert review["decision"] == "accept"
    assert review["reviewer_kind"] == "astra_judge"
    assert review["reviewer_id"] == "gpt-6-astra/bufo-astra-v5/k3"
    assert review["partition"] == "train"
    assert review["judge"]["samples"] == 3
    assert review["judge"]["vs_reference"] == "equal"


def test_decision_rejects_low_confidence_and_fail():
    assert decide(scored_row(judge_confidence=0.5))["decision"] == "reject"
    assert decide(scored_row(strict_pass=False))["decision"] == "reject"
    assert decide(scored_row(status="error"))["decision"] == "reject"
    assert decide(scored_row(completed=False))["decision"] == "reject"


def write_scored(tmp_path, rows):
    directory = tmp_path / "astra-v5-k3"
    directory.mkdir()
    (directory / "scored.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n" for r in rows))
    return directory


def test_write_decisions_counts_and_writes(tmp_path):
    rows = [scored_row(), scored_row(strict_pass=False)]
    counts = write_decisions(
        write_scored(tmp_path, rows), tmp_path / "out" / "d.jsonl", registry_key=KEY
    )
    assert counts == {"accepted": 1, "rejected": 1, "skipped": 0}
    assert len((tmp_path / "out" / "d.jsonl").read_text().splitlines()) == 2


def test_write_decisions_refuses_other_scorer_config(tmp_path):
    row = scored_row()
    row.scores["astra_judge"]["scorer_configuration"].pop("samples")
    directory = write_scored(tmp_path, [row])
    with pytest.raises(ValueError, match="not scored by"):
        write_decisions(directory, tmp_path / "d.jsonl", registry_key=KEY)


def test_write_decisions_refuses_uncalibrated_key(tmp_path):
    directory = write_scored(tmp_path, [scored_row()])
    with pytest.raises(ValueError, match="not calibrated"):
        write_decisions(directory, tmp_path / "d.jsonl", registry_key=("bufo-astra-v1", 1))


def reviewed_manifest(tmp_path, partitions):
    from test_trajectories import captured_row

    rows = []
    for partition in partitions:
        row = captured_row()
        family = f"{partition}-family"
        row.backend = "closed_quality_bar"
        row.group_id = family
        row.svg += f"<!-- {partition} -->"
        row.trajectory["provenance"]["group_id"] = family
        row.scores = {"astra_judge": verdict()}
        row.benchmark_metadata = {"split": partition, "family": family, "status": "completed"}
        rows.append(row)
    directory = write_scored(tmp_path, rows)
    decisions = tmp_path / "decisions.jsonl"
    write_decisions(directory, decisions, registry_key=KEY)
    manifest = tmp_path / "manifest.jsonl"
    for row in rows:
        Manifest(manifest).append(row)
    return rows, manifest, decisions


def export(manifest, decisions, output, *extra):
    return dispatch(
        parser().parse_args(
            ["--stage", "export-reviewed", "--manifest", str(manifest), "--reviews", str(decisions)]
            + ["--output", str(output), *extra]
        )
    )


async def test_export_reviewed_accepts_astra_judge_reviews(tmp_path):
    rows, manifest, decisions = reviewed_manifest(tmp_path, ["train"])
    await export(manifest, decisions, tmp_path / "export")
    data = json.loads((tmp_path / "export" / "reviewed-examples.jsonl").read_text())
    assert data["source_generation_id"] == rows[0].id
    assert data["training_review"]["reviewer_kind"] == "astra_judge"
    assert data["training_review"]["judge"]["strict_pass"] is True


async def test_validation_rows_export_only_as_val_split(tmp_path):
    from bufogen.integrations.forge import reviewed_entries

    _, manifest, decisions = reviewed_manifest(tmp_path, ["train", "validation"])
    await export(manifest, decisions, tmp_path / "without")
    train_only = (tmp_path / "without" / "reviewed-examples.jsonl").read_text().splitlines()
    assert [json.loads(line)["partition"] for line in train_only] == ["train"]
    await export(manifest, decisions, tmp_path / "with", "--include-validation")
    lines = (tmp_path / "with" / "reviewed-examples.jsonl").read_text().splitlines()
    examples = [json.loads(line) for line in lines]
    assert sorted(e["partition"] for e in examples) == ["train", "validation"]
    splits = {e["partition"]: x["split"] for e, x in zip(examples, reviewed_entries(examples))}
    assert splits == {"train": "train", "validation": "val"}


def rewrite(decisions, **changes):
    reviews = [json.loads(line) for line in decisions.read_text().splitlines()]
    for review in reviews:
        for key, value in changes.items():
            if key == "judge":
                review["judge"].update(value)
            else:
                review[key] = value
    decisions.write_text("".join(json.dumps(r) + "\n" for r in reviews))


async def test_export_refuses_uncalibrated_judge_decisions(tmp_path):
    _, manifest, decisions = reviewed_manifest(tmp_path, ["train"])
    rewrite(
        decisions,
        reviewer_id="gpt-6-astra/bufo-astra-v1/k1",
        judge={"rubric_version": "bufo-astra-v1", "samples": 1},
    )
    with pytest.raises(ValueError, match="not calibrated"):
        await export(manifest, decisions, tmp_path / "out")


async def test_export_refuses_decisions_whose_rows_disagree(tmp_path):
    rows, manifest, decisions = reviewed_manifest(tmp_path, ["train"])
    scored = tmp_path / "astra-v5-k3" / "scored.jsonl"
    row = json.loads(scored.read_text())
    row["scores"]["astra_judge"]["strict_pass"] = False
    scored.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="disagrees"):
        await export(manifest, decisions, tmp_path / "out")


async def test_export_refuses_decisions_from_other_scorer_config(tmp_path):
    _, manifest, decisions = reviewed_manifest(tmp_path, ["train"])
    scored = tmp_path / "astra-v5-k3" / "scored.jsonl"
    row = json.loads(scored.read_text())
    row["scores"]["astra_judge"]["scorer_configuration"].pop("samples")
    scored.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="not scored by"):
        await export(manifest, decisions, tmp_path / "out")
