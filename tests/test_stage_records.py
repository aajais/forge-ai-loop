import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bufogen.workflows import stage_records

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/publish_stage_b_records.py"
spec = importlib.util.spec_from_file_location("publish_stage_b_records", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


@pytest.fixture
def fakes(monkeypatch):
    calls = SimpleNamespace(weave=[], inits=[], artifacts=[], datasets=[])

    class FakeDataset:
        def __init__(self, name, rows):
            self.name, self.rows = name, rows
            calls.datasets.append(self)

    class FakeArtifact:
        def __init__(self, name, type, metadata):
            self.name, self.type, self.metadata, self.files = name + ":v0", type, metadata, []
            calls.artifacts.append(self)

        def add_file(self, path, name=None):
            self.files.append((path, name))

        def wait(self):
            return self

    class FakeRun:
        id = "run123"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def log_artifact(self, artifact, aliases=None):
            return artifact

    import wandb
    import weave

    import bufogen.integrations.telemetry as telemetry

    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(weave, "Dataset", FakeDataset)

    def publish(obj, name=None):
        calls.weave.append((obj, name))
        return SimpleNamespace(uri=lambda: f"weave:///e/p/object/{name}:digest{len(calls.weave)}")

    monkeypatch.setattr(weave, "publish", publish)
    monkeypatch.setattr(wandb, "init", lambda **kw: calls.inits.append(kw) or FakeRun())
    monkeypatch.setattr(wandb, "Artifact", FakeArtifact)
    return calls


def test_publish_once_then_skip_then_republish(tmp_path, fakes):
    ledger = tmp_path / "publications.json"
    f = tmp_path / "funnel.json"
    f.write_text('{"a": 1}')
    first = stage_records.publish_record("funnel", f, metadata={"n": 1}, ledger=ledger)
    assert first["published"] is True
    assert len(fakes.weave) == 1 and fakes.weave[0][1] == "bufo-stage-b-funnel"
    assert fakes.weave[0][0]["content"] == {"a": 1}
    assert fakes.inits[0]["job_type"] == "stage-b-record"
    assert fakes.inits[0]["group"] == "stage-b" and fakes.inits[0]["name"] == "stage-b-funnel"
    assert fakes.inits[0]["reinit"] is True
    art = fakes.artifacts[0]
    assert art.type == "stage-b-record" and art.metadata["n"] == 1
    assert art.files[0][1] == "funnel.json"
    entries = json.loads(ledger.read_text())
    assert len(entries) == 1
    assert set(entries[0]) == {
        "kind",
        "sha256",
        "weave_ref",
        "wandb_artifact",
        "run_id",
        "published_at",
    }
    assert entries[0]["run_id"] == "run123" and entries[0]["weave_ref"].startswith("weave:///")

    again = stage_records.publish_record("funnel", f, metadata={"n": 1}, ledger=ledger)
    assert again["published"] is False and again["sha256"] == first["sha256"]
    assert len(fakes.weave) == 1 and len(json.loads(ledger.read_text())) == 1

    f.write_text('{"a": 2}')
    third = stage_records.publish_record("funnel", f, metadata={}, ledger=ledger)
    assert third["published"] is True and len(fakes.weave) == 2
    assert len(json.loads(ledger.read_text())) == 2


def test_jsonl_becomes_dataset_rows_and_markdown_text(tmp_path, fakes):
    ledger = tmp_path / "l.json"
    j = tmp_path / "d.jsonl"
    j.write_text('{"x": 1}\n{"x": 2}\n')
    stage_records.publish_record("decisions-attempt-1", j, metadata={}, ledger=ledger)
    assert fakes.datasets[0].rows == [{"x": 1}, {"x": 2}]
    assert fakes.datasets[0].name == "bufo-stage-b-decisions-attempt-1"
    m = tmp_path / "p.md"
    m.write_text("# hi")
    stage_records.publish_record("progress-doc", m, metadata={}, ledger=ledger)
    assert fakes.weave[-1][0]["content"] == "# hi"


def test_env_values_never_in_metadata_or_content(tmp_path, fakes, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "SECRET-VALUE-12345")
    f = tmp_path / "a.json"
    f.write_text("{}")
    ledger = tmp_path / "l.json"
    with pytest.raises(ValueError, match="environment"):
        stage_records.publish_record("x", f, metadata={"k": "SECRET-VALUE-12345"}, ledger=ledger)
    f.write_text('{"k": "SECRET-VALUE-12345"}')
    with pytest.raises(ValueError, match="environment"):
        stage_records.publish_record("x", f, metadata={}, ledger=ledger)
    assert not fakes.weave and not ledger.exists()
    f.write_text("{}")
    stage_records.publish_record("x", f, metadata={"k": 1}, ledger=ledger)
    assert "SECRET-VALUE-12345" not in ledger.read_text()


def _tree(root: Path):
    sb = root / "runs/distill/stage-b"
    files = {
        "export/funnel.json": "{}",
        "decisions/attempt-1.jsonl": '{"decision": "accept"}\n{"decision": "reject"}\n',
        "decisions/attempt-2.jsonl": '{"decision": "accept"}\n',
        "decisions/arms-v1-v5-k3.jsonl": '{"decision": "accept"}\n',
        "rl/astra-calibration.json": "{}",
        "anchoring/report.json": "{}",
        "anchoring/manifest.json": "{}",
        "student-probe.json": "{}",
        "rl/tasks-train.metadata.json": "{}",
        "checkpoints.json": "[]",
        "lowconf/NONE.txt": "none",
        "lowconf/summary.json": '{"count": 0}',
        "sweep/paired-foo.json": "{}",
        "rl-eval/paired-foo.json": "{}",
        "sweep/key/paired-k.json": "{}",
        "other/paired-bar.json": "{}",
        "heldout-sol-arm/paired-secret.json": "{}",
        "sol-data/attempt-1/paired-x.json": "{}",
        "stage-b-log.md": "log",
    }
    for rel, text in files.items():
        p = sb / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / "docs/distill").mkdir(parents=True)
    (root / "docs/distill/augmented-prompts.md").write_text("aug")
    (root / "docs/distill/stage-b-progress.md").write_text("prog")
    (root / "config").mkdir()
    (root / "config/distill_prompts.yaml").write_text("prompts: []")


def test_script_enumerates_kinds_dry_run(tmp_path, monkeypatch, capsys, fakes):
    _tree(tmp_path)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    assert script.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    for kind in (
        "funnel",
        "decisions-attempt-1",
        "decisions-attempt-2",
        "astra-calibration",
        "anchoring-probe",
        "student-probe",
        "rl-tasks",
        "checkpoints",
        "lowconf",
        "sweep-paired-foo",
        "rl-eval-paired-foo",
        "prompt-set",
        "prompt-set-doc",
        "progress-doc",
        "stage-log",
    ):
        assert f"WOULD PUBLISH {kind} " in out, kind
    assert "paired-bar" not in out and "paired-k" not in out
    assert "arms-v1" not in out and "secret" not in out and "sol-data" not in out
    assert not fakes.weave and not (tmp_path / "runs/distill/stage-b/publications.json").exists()


def test_script_metadata_counts_and_missing_skipped(tmp_path, monkeypatch, capsys):
    (tmp_path / "runs/distill/stage-b/decisions").mkdir(parents=True)
    (tmp_path / "runs/distill/stage-b/decisions/attempt-1.jsonl").write_text(
        '{"decision": "accept"}\n{"decision": "reject"}\n{"decision": "accept"}\n'
    )
    (tmp_path / "runs/distill/stage-b/lowconf").mkdir()
    (tmp_path / "runs/distill/stage-b/lowconf/NONE.txt").write_text("n")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    records = script.collect(tmp_path)
    assert [(k, m["accept"], m["reject"]) for k, _, m in records if k.startswith("decisions")] == [
        ("decisions-attempt-1", 2, 1)
    ]
    script.main(["--dry-run"])
    assert "SKIP missing" in capsys.readouterr().out


def test_lowconf_prefers_summary_and_errors_when_absent(tmp_path):
    _tree(tmp_path)
    low = tmp_path / "runs/distill/stage-b/lowconf"
    kinds = {k: p.name for k, p, _ in script.collect(tmp_path)}
    assert kinds["lowconf"] == "summary.json"
    (low / "summary.json").unlink()
    assert {k: p.name for k, p, _ in script.collect(tmp_path)}["lowconf"] == "NONE.txt"
    (low / "NONE.txt").unlink()
    with pytest.raises(FileNotFoundError, match="lowconf"):
        script.collect(tmp_path)


def test_skip_compares_latest_sha_so_a_b_a_republishes(tmp_path, fakes):
    ledger = tmp_path / "l.json"
    f = tmp_path / "a.json"
    for text in ('{"v": "A"}', '{"v": "B"}', '{"v": "A"}'):
        f.write_text(text)
        assert stage_records.publish_record("k", f, metadata={}, ledger=ledger)["published"]
    assert len(json.loads(ledger.read_text())) == 3
    assert not stage_records.publish_record("k", f, metadata={}, ledger=ledger)["published"]


def test_confirmation_and_sweep_kinds_present_absent_and_blinded(tmp_path):
    _tree(tmp_path)
    sb = tmp_path / "runs/distill/stage-b"
    new = {
        "sweep-selection": "sweep/selection.json",
        "sweep-distinctness": "sweep/distinctness.json",
        "confirmation-summary": "confirmation/astra-v5-k3/summary.json",
        "heldout-sol-summary": "heldout-sol-arm/astra-v5-k3/summary.json",
    }
    # Absent files are listed with None metadata (logged as SKIP), never an error.
    first = {k: (p, m) for k, p, m in script.collect(tmp_path)}
    for kind in new:
        assert first[kind][1] is None, kind
    for rel in new.values():
        (sb / rel).parent.mkdir(parents=True, exist_ok=True)
        (sb / rel).write_text("{}")
    # Blinded material must never be collected.
    for rel in (
        "confirmation/packet/items.json",
        "confirmation/key/unblinding-key.json",
        "confirmation/key/paired-x.json",
    ):
        (sb / rel).parent.mkdir(parents=True, exist_ok=True)
        (sb / rel).write_text("{}")
    records = script.collect(tmp_path)
    found = {k: (p, m) for k, p, m in records}
    for kind, rel in new.items():
        assert found[kind][0] == sb / rel and found[kind][1]["source_path"].endswith(rel), kind
    assert not any(
        "packet" in str(p) or "key" in p.relative_to(tmp_path).parts for _, p, _ in records
    )
