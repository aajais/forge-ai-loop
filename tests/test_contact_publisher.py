"""Offline publication safeguards using synthetic identities and bytes only."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "docs/forge-contact-pilot/publish.py"
SPEC = importlib.util.spec_from_file_location("contact_publisher_tests", PATH)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def packets(tmp_path):
    folder = tmp_path / "blind-review"
    folder.mkdir()
    image = tmp_path / "synthetic.png"
    image.write_bytes(b"synthetic bytes; not a model output")
    evidence = {"svg": [], "contact": []}
    key, inputs, decisions = [], [[], [], []], [[], [], []]
    for index in range(64):
        interface = "svg" if index < 32 else "contact"
        row = {
            "id": f"attempt-{index}", "prompt_id": f"prompt-{index % 16}",
            "backend": "open_base" if index % 2 else "closed_quality_bar",
            "model": "synthetic-model", "prompt": "synthetic prompt",
            "png_path": str(image) if index else None,
            "benchmark_metadata": {"requirements": ["visible grip"], "status": "completed"},
            "valid": bool(index),
        }
        evidence[interface].append(row)
        key.append({k: row[k] for k in ("id", "prompt_id", "backend", "model")} | {
            "interface": interface, "index": index,
        })
        item = {
            "index": index, "label": f"Image {index+1:02d}",
            "prompt": row["prompt"], "requirements": ["visible grip"],
            "png_path": row["png_path"], "png_sha256": sha(image) if index else None,
        }
        inputs[index % 3].append(item)
        decisions[index % 3].append({
            k: item[k] for k in ("index", "label", "png_sha256")
        } | {
            "reviewer_kind": "assistant", "passed": bool(index),
            "evidence": "Synthetic evidence." if index else "No image generated.",
            "failed_requirements": [] if index else ["visible grip"],
        })
    (folder / "unblinding-key.json").write_text(json.dumps(key))
    for reviewer in range(3):
        directory = folder / f"reviewer-{reviewer}"
        directory.mkdir()
        (directory / "inputs.json").write_text(json.dumps(inputs[reviewer]))
        (directory / "decisions.jsonl").write_text(
            "\n".join(json.dumps(item) for item in decisions[reviewer])
        )
    return tmp_path, evidence


def validate(packets):
    directory, evidence = packets
    return publisher.validate_reviews(directory, evidence, sha=sha, read=read)


def test_complete_review_includes_failed_attempt_and_freezes_evidence(packets):
    reviews, hashes, status = validate(packets)
    assert status["status"] == "complete"
    assert len(reviews) == 64
    assert reviews["attempt-0"]["passed"] is False
    assert len(hashes) == 7  # key + three input packets + three decision files


def test_partial_review_is_waiting(packets):
    path = packets[0] / "blind-review/reviewer-1/decisions.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    reviews, _, status = validate(packets)
    assert len(reviews) == 63
    assert status["status"] == "waiting"


@pytest.mark.parametrize("mutation", ["duplicate", "label", "hash", "kind", "requirements", "missing_png_pass"])
def test_invalid_or_misattributed_review_rejected(packets, mutation):
    path = packets[0] / "blind-review/reviewer-0/decisions.jsonl"
    rows = publisher.jsonl(path)
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "label":
        rows[1]["label"] = "Image 99"
    elif mutation == "hash":
        rows[1]["png_sha256"] = "changed"
    elif mutation == "kind":
        rows[1]["reviewer_kind"] = "human"
    elif mutation == "requirements":
        rows[1]["failed_requirements"] = ["invented requirement"]
    else:
        rows[0]["passed"] = True
        rows[0]["failed_requirements"] = []
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        validate(packets)


def test_key_model_identity_must_match_actual_attempt(packets):
    path = packets[0] / "blind-review/unblinding-key.json"
    key = read(path)
    key[0]["model"] = "different-model"
    path.write_text(json.dumps(key))
    with pytest.raises(ValueError, match="identity"):
        validate(packets)


def test_publish_refuses_incomplete_reviews_before_network(monkeypatch, tmp_path):
    directory = tmp_path / "comparison"
    for interface in ("svg", "contact"):
        (directory / interface).mkdir(parents=True)
    monkeypatch.setattr(publisher, "validate", lambda _: (
        SimpleNamespace(BASE="base", TEACHER="teacher"), [], {}, {}, {"purpose": "test"},
        {}, {}, {"status": "waiting", "reviewed_attempts": 63},
    ))
    assert publisher.publish(tmp_path)["offline_validation"] == "waiting_for_blind_review"
    with pytest.raises(ValueError, match="all 64"):
        publisher.publish(tmp_path, execute=True)
    assert not (directory / "visualization.json").exists()


def test_visual_success_does_not_hide_generation_failure():
    row = {"valid": True, "benchmark_metadata": {"status": "generation_error"}}
    assert not publisher.assistant_passed(row, {"passed": True})


def test_publication_tables_align_reviews_and_provisional_scores(monkeypatch, packets):
    root, evidence = packets
    reviews, _, status = validate(packets)
    summaries = {}
    prompts = [{"id": f"prompt-{i}", "prompt": "synthetic prompt", "requirements": ["visible grip"]}
               for i in range(16)]
    for interface, rows in evidence.items():
        (root / "comparison" / interface).mkdir(parents=True)
        for i, row in enumerate(rows):
            row.update(
                prompt_id=f"prompt-{i // 2}", input_tokens=1, output_tokens=2,
                latency_s=1.0, tool_calls=1, prompt_version="synthetic",
                scores={"scene_quality": {
                    "status": "ok", "quality_failure": False,
                    "constraint_coverage": 1.0, "prompt_checks": [True],
                }},
            )
        summaries[interface] = {"arms": {
            arm: {
                "total": 16, "provisional_passes": 16, "provisional_pass_rate_all_attempts": 1.0,
                "metric_means_scored_only": {"constraint_coverage": 1.0},
                "generation_errors": 0, "judge_errors": 0,
            } for arm in publisher.ARMS
        }}
    monkeypatch.setattr(publisher, "validate", lambda _: (
        SimpleNamespace(BASE="base", TEACHER="teacher", INTERFACES=("svg", "contact"), read=read),
        prompts, evidence, summaries, {"purpose": "synthetic"}, {}, reviews, status,
    ))
    tables, logged = [], {}

    class Table:
        def __init__(self, columns):
            self.columns, self.rows = columns, []
            tables.append(self)

        def add_data(self, *values):
            assert len(values) == len(self.columns)
            self.rows.append(dict(zip(self.columns, values, strict=True)))

    class Run:
        url = "synthetic://run"
        summary = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def log(self, items):
            logged.update(items)

        def log_artifact(self, _):
            return SimpleNamespace(name="evidence:v0", wait=lambda: None)

    monkeypatch.setattr(publisher, "project_path", lambda: "synthetic/project")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        Table=Table, Image=lambda path: path, init=lambda **_: Run(),
        Artifact=lambda *_, **__: SimpleNamespace(add_file=lambda *_, **__: None),
    ))
    result = publisher.publish(root, execute=True)
    assert result["status"] == "published"
    assert [len(table.rows) for table in tables] == [16, 64, 4]
    attempts = logged["all_64_attempts"].rows
    failed = next(row for row in attempts if row["generation_id"] == "attempt-0")
    assert failed["assistant_visual_pass"] is False
    assert failed["assistant_evidence"] == "No image generated."
    assert failed["requirement_checks"] == [True]
    assert failed["recorded_input_tokens"] == 1
    assert all(row["attempts"] == 16 for row in logged["interface_summary"].rows)
