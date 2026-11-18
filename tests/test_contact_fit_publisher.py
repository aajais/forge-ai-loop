"""Offline publication checks over the real synthetic comparison validator."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_contact_fit_compare import compare, comparison, write_reviews

PATH = Path(__file__).resolve().parents[1] / "docs/forge-contact-pilot/publish_fit.py"
SPEC = importlib.util.spec_from_file_location("contact_fit_publisher_tests", PATH)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


@pytest.fixture
def ready(comparison, monkeypatch):
    root = comparison[0]
    data = compare.validate(root)
    write_reviews(data)
    compare.summarize(data)
    monkeypatch.setattr(publisher, "fit_module", lambda: compare)
    return root


def fake_wandb(monkeypatch, *, fail_upload=False):
    calls, logged, files = [], {}, []

    class Table:
        def __init__(self, columns):
            self.columns, self.rows = columns, []

        def add_data(self, *values):
            assert len(values) == len(self.columns)
            self.rows.append(dict(zip(self.columns, values, strict=True)))

    class Run:
        url = "synthetic://fit-run"

        def __init__(self):
            self.summary = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def log(self, tables):
            logged.update(tables)

        def log_artifact(self, artifact):
            if fail_upload:
                raise RuntimeError("Synthetic uncertain upload")
            return SimpleNamespace(name="fit-evidence:v0", wait=lambda: None)

    def init(**kwargs):
        calls.append(kwargs)
        assert "wandb-artifact:///" not in json.dumps(kwargs["config"])
        assert kwargs["config"]["in_sample"] is True
        assert kwargs["config"]["promotion"] is False
        kwargs["config"]["fit_gate"]["synthetic_sdk_metadata"] = True
        return Run()

    monkeypatch.setattr(publisher, "project_path", lambda: "synthetic/project")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Table=Table,
            Image=lambda path: path,
            init=init,
            Artifact=lambda *_, **__: SimpleNamespace(
                add_file=lambda path, name: files.append((path, name))
            ),
        ),
    )
    return calls, logged, files


def test_default_reuses_frozen_comparison_without_publication_or_rewrite(ready, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **_: pytest.fail("network"))
    )
    summary_path = ready / "fit-comparison/comparison.json"
    stamp = summary_path.stat().st_mtime_ns
    result = publisher.publish(ready)
    assert result["network_calls"] == 0 and result["attempts"] == 48 and result["paired_rows"] == 24
    assert result["in_sample"] is True and result["promotion"] is False
    assert summary_path.stat().st_mtime_ns == stamp
    assert not (ready / "fit-comparison/visualization.json").exists()


def test_missing_summary_does_not_create_comparison_or_publish(comparison, monkeypatch):
    monkeypatch.setattr(publisher, "fit_module", lambda: compare)
    with pytest.raises(ValueError, match="freeze fit_compare"):
        publisher.publish(comparison[0])
    assert not (comparison[0] / "fit-comparison").exists()


def test_all_attempts_and_partial_png_verdicts_are_preserved_and_upload_is_idempotent(
    ready, monkeypatch
):
    calls, logged, files = fake_wandb(monkeypatch)
    receipt = publisher.publish(ready, execute=True)
    assert receipt["status"] == "published"
    assert "synthetic_sdk_metadata" not in receipt["gate"]
    assert receipt["candidate_identity"]["checkpoint"].endswith(":v0")
    assert len(logged["paired_training_request_images"].rows) == 24
    attempts = logged["all_48_attempts"].rows
    assert len(attempts) == 48
    partial = next(row for row in attempts if row["generation_id"] == "baseline-0")
    assert partial["png"] and partial["assistant_image_pass"] and partial["automated_image_pass"]
    assert partial["generation_status"] == "generation_error"
    assert not partial["completed_and_valid"] and not partial["combined_pass"]
    assert not partial["automated_completed_pass"] and not partial["assistant_completed_pass"]
    missing = next(row for row in attempts if row["generation_id"] == "baseline-1")
    assert missing["png"] is None and not missing["combined_pass"]
    tuned = next(row for row in attempts if row["arm"] == "candidate")
    assert tuned["model_label"] == f"ft-{compare.BASE}"
    assert tuned["actual_model_id"] == tuned["immutable_checkpoint"]
    assert tuned["captured_png_sha256"] == tuned["review_png_sha256"]
    assert tuned["capture_manifest_sha256"] and tuned["review_file_sha256"]
    assert len({name for _, name in files}) == len(files)
    assert any("comparison.json" in name for _, name in files)
    assert publisher.publish(ready, execute=True) == receipt
    assert len(calls) == 1


def test_uncertain_upload_blocks_retry_with_the_saved_run_id(ready, monkeypatch):
    calls, _, _ = fake_wandb(monkeypatch, fail_upload=True)
    with pytest.raises(RuntimeError, match="uncertain upload"):
        publisher.publish(ready, execute=True)
    receipt = compare.read(ready / "fit-comparison/visualization.json")
    assert receipt["status"] == "publishing" and receipt["run_id"] == calls[0]["id"]
    with pytest.raises(RuntimeError, match="reconcile"):
        publisher.publish(ready, execute=True)
    assert len(calls) == 1


def test_modified_review_cannot_republish_existing_run(ready, monkeypatch):
    calls, _, _ = fake_wandb(monkeypatch)
    publisher.publish(ready, execute=True)
    path = ready / "fit-comparison/visual-review/reviewer-0/decisions.jsonl"
    rows = compare.jsonl(path)
    rows[0]["evidence"] = "Changed after frozen comparison"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        publisher.publish(ready, execute=True)
    assert len(calls) == 1


def test_validate_does_not_repair_an_altered_frozen_summary(ready):
    path = ready / "fit-comparison/comparison.json"
    summary = compare.read(path)
    summary["gate"]["passed"] = not summary["gate"]["passed"]
    path.write_bytes(compare.encoded(summary))
    changed = path.read_bytes()
    with pytest.raises(ValueError):
        publisher.validate(ready)
    assert path.read_bytes() == changed
