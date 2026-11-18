"""Offline checks for immutable experiment identity and fixture provenance."""

import asyncio
import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "docs/forge-contact-pilot/experiment.py"
SPEC = importlib.util.spec_from_file_location("contact_experiment_tests", PATH)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "wandb-artifact:///e/p/a:latest",
        "wandb-artifact:///e/p/a:step3",
        "wandb-artifact:///e/p/a:candidate",
        "wandb-artifact:///e/p/a:v-1",
        "wandb-artifact:///e/p/a:v1/extra",
        "wandb-artifact:///e/p:v1",
        "wandb-artifact:///e//a:v1",
        "wandb-artifact:///e/p/a:v1\n",
    ],
)
def test_reject_mutable_or_malformed_checkpoints(value):
    with pytest.raises(ValueError, match="immutable"):
        experiment.require_checkpoint(value)


@pytest.mark.parametrize("version", ["v0", "v1", "v103"])
def test_accept_exact_artifact_version(version):
    value = f"wandb-artifact:///my-entity/my_project/artifact-name:{version}"
    assert experiment.require_checkpoint(value) == value


@pytest.fixture
def protocol_inputs(tmp_path, monkeypatch):
    here, fixtures = tmp_path / "code", tmp_path / "fixtures"
    here.mkdir()
    fixtures.mkdir()
    (here / "evaluate.py").write_text("# synthetic verifier\n")
    (fixtures / "manifest.json").write_text('{"synthetic":"manifest"}')
    (fixtures / "assistant-review.json").write_text('{"synthetic":"review"}')
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "HERE", here)
    monkeypatch.setattr(experiment, "FIXTURES", fixtures)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config = {"source_hashes": {"code/evaluate.py": sha(here / "evaluate.py")}}
    evidence = {
        "manifest_sha256": sha(fixtures / "manifest.json"),
        "review_sha256": sha(fixtures / "assistant-review.json"),
    }
    return tmp_path / "output", config, evidence


def test_freeze_retains_evidence_and_resumes_without_rewriting(protocol_inputs):
    output, config, fixtures = protocol_inputs
    first = experiment.freeze_protocol(output, config, fixtures)
    path = output / "experiment-protocol.json"
    stamp = path.stat().st_mtime_ns
    assert first == experiment.freeze_protocol(output, config, fixtures)
    assert path.stat().st_mtime_ns == stamp
    assert json.loads(path.read_text())["fixture_evidence"] == fixtures
    snapshots = output / "protocol-evidence"
    assert {p.name for p in snapshots.iterdir()} == {
        "fixture-manifest.json",
        "fixture-assistant-review.json",
        "fixture-verifier.py",
    }


def test_refuse_adopting_existing_outputs(protocol_inputs):
    output, config, fixtures = protocol_inputs
    output.mkdir()
    (output / "manifest.jsonl").write_text("")
    with pytest.raises(ValueError, match="nonempty"):
        experiment.freeze_protocol(output, config, fixtures)
    assert not (output / "experiment-protocol.json").exists()


@pytest.mark.parametrize("filename", ["manifest.json", "assistant-review.json", "evaluate.py"])
def test_preparation_race_fails_before_protocol_created(protocol_inputs, filename):
    output, config, fixtures = protocol_inputs
    folder = experiment.HERE if filename == "evaluate.py" else experiment.FIXTURES
    (folder / filename).write_text("changed after validation")
    with pytest.raises(ValueError, match="changed after preparation"):
        experiment.freeze_protocol(output, config, fixtures)
    assert not output.exists()


def test_frozen_evidence_tampering_rejected(protocol_inputs):
    output, config, fixtures = protocol_inputs
    experiment.freeze_protocol(output, config, fixtures)
    (output / "protocol-evidence/fixture-verifier.py").write_text("changed")
    with pytest.raises(ValueError, match="Frozen fixture evidence"):
        experiment.freeze_protocol(output, config, fixtures)


def test_configuration_change_cannot_resume(protocol_inputs):
    output, config, fixtures = protocol_inputs
    experiment.freeze_protocol(output, config, fixtures)
    with pytest.raises(ValueError, match="protocol changed"):
        experiment.freeze_protocol(output, config | {"model": "another"}, fixtures)


def test_protocol_rejection_happens_before_weave_or_paid_work(monkeypatch, tmp_path):
    import bufogen.integrations.telemetry

    called = []
    monkeypatch.setattr(experiment, "prepare", lambda *_: ([], [], {}, {}))
    monkeypatch.setattr(
        bufogen.integrations.telemetry, "init_weave", lambda: called.append("network")
    )

    def reject(*_):
        raise ValueError("protocol rejected")

    monkeypatch.setattr(experiment, "freeze_protocol", reject)
    args = SimpleNamespace(prompts=None, arm=[], checkpoint=None, output=tmp_path)
    with pytest.raises(ValueError, match="protocol rejected"):
        asyncio.run(experiment.run(args))
    assert called == []


def test_relative_runpy_cli_offline_hashes_sources(monkeypatch, tmp_path, capsys):
    """Exercise the actual CLI/harness with runpy's relative __file__."""
    import bufogen.integrations.telemetry

    repository = PATH.parents[2]
    monkeypatch.chdir(repository)
    monkeypatch.syspath_prepend(str(PATH.parent))
    import evaluate

    monkeypatch.setattr(
        evaluate,
        "validate_fixture_review",
        lambda *_: {
            "manifest_sha256": "synthetic",
            "review_sha256": "synthetic",
        },
    )

    def forbid_network():
        raise AssertionError("Offline CLI must not initialize Weave")

    monkeypatch.setattr(bufogen.integrations.telemetry, "init_weave", forbid_network)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment.py",
            "--output",
            str(tmp_path / "offline-output"),
        ],
    )
    names = ("BufoAgent", "instructions", "prompt_version", "_source_hashes")
    originals = {name: getattr(experiment.benchmark, name) for name in names}
    runpy.run_path("docs/forge-contact-pilot/experiment.py", run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["offline_validation"] == "passed"
    assert result["network_calls"] == 0
    assert result["attempts"] == 32
    assert len(result["configuration_hash"]) == 64
    assert all(getattr(experiment.benchmark, name) is value for name, value in originals.items())
    assert not (tmp_path / "offline-output").exists()
