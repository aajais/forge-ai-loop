from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from bufogen.app.ui import create_app
from bufogen.generation import BufoGeneration, Manifest
from bufogen.models import load_backends


def test_status_uses_serving_ids_and_separate_adapter_versions(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_MODEL", "google/current-model")
    monkeypatch.setenv("SOL_MODEL", "gpt-current")
    monkeypatch.setenv("OPEN_TUNED_MODEL", "wandb-artifact:///team/project/adapter:v2")
    monkeypatch.setenv("OPEN_TUNED_BASE_MODEL", "google/trained-model")
    monkeypatch.setenv("OPEN_RL_MODEL", "wandb-artifact:///team/project/adapter:v3")
    monkeypatch.setenv("OPEN_RL_BASE_MODEL", "google/trained-model")
    with TestClient(create_app(output_dir=tmp_path)) as client:
        models = {row["id"]: row for row in client.get("/api/status").json()["models"]}
    assert models["open_base"]["label"] == "google/current-model"
    assert models["closed_quality_bar"]["label"] == "gpt-current"
    assert models["open_tuned"]["label"] == "ft-google/trained-model"
    assert models["open_rl"]["label"] == "ft-google/trained-model"
    assert models["open_tuned"]["model_version"].endswith("adapter:v2")
    assert models["open_rl"]["model_version"].endswith("adapter:v3")
    assert models["open_tuned"]["model"].startswith("wandb-artifact:///")
    assert "alias" in models["forge_proxy"]["label"]


def test_history_uses_recorded_models_not_current_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_MODEL", "google/new-model")
    monkeypatch.setenv("OPEN_TUNED_BASE_MODEL", "google/new-model")
    records = [
        ("open_base", "google/old-model", {}),
        ("open_tuned", "wandb-artifact:///t/p/a:v1", {"base_model": "google/trained-model"}),
        (
            "forge_proxy",
            "bufo-gen",
            {"requested_alias": "bufo-gen", "resolved_serving_models": ["google/resolved-model"]},
        ),
    ]
    for backend, model, provenance in records:
        Manifest(tmp_path / "manifest.jsonl").append(
            BufoGeneration(
                prompt_id="p",
                prompt="Bufo",
                backend=backend,
                model=model,
                group_id="g",
                svg="",
                valid=False,
                latency_s=0,
                trajectory={"provenance": provenance},
            )
        )
    with TestClient(create_app(output_dir=tmp_path)) as client:
        rows = client.get("/api/history").json()
    assert [r["model_label"] for r in rows] == [
        "google/old-model",
        "ft-google/trained-model",
        "google/resolved-model",
    ]
    assert rows[1]["model_version"] == "wandb-artifact:///t/p/a:v1"
    assert rows[2]["model"] == "bufo-gen"


@pytest.mark.asyncio
async def test_agent_trace_name_uses_model_id(tmp_path, monkeypatch):
    from bufogen.agents.agent import BufoAgent
    from bufogen.integrations import telemetry

    monkeypatch.setenv("OPEN_MODEL", "google/current-model")
    seen = []

    @contextmanager
    def generation_turn(name, model, *args, **kwargs):
        seen.append((name, model))
        yield SimpleNamespace()

    monkeypatch.setattr(telemetry, "generation_turn", generation_turn)
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    agent = BufoAgent(load_backends()["open_base"], None, tmp_path)

    async def generate(*args, **kwargs):
        return SimpleNamespace(response_text="Here you go!", png_content_ref=None)

    monkeypatch.setattr(agent, "_generate", generate)
    await agent.generate("Bufo")
    assert seen == [("Bufo google/current-model", "google/current-model")]


def test_reports_use_frozen_ids_and_leave_source_evidence_unchanged(tmp_path, monkeypatch):
    import json

    from test_forge_cli import benchmark_fixture

    from bufogen.workflows.loop_report import write_loop_report

    monkeypatch.setenv("OPEN_MODEL", "google/new-model")
    directory = tmp_path / "benchmark"
    benchmark_fixture(directory, fingerprint="frozen")
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    page = write_loop_report(tmp_path, tmp_path / "report", benchmark_dirs=[directory])
    report = json.loads(page.with_name("loop-report.json").read_text())
    run = report["benchmarks"][0]
    model = run["models"][0]["model"]
    assert run["model_presentations"]["student"]["model_label"] == model
    assert f"<td>{model}</td>" in page.read_text()
    assert "<td>student</td>" not in page.read_text()
    assert before == {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def test_legacy_frozen_artifact_labels_preserve_uri_separately():
    from bufogen.models import frozen_model_presentations, generation_model_presentation

    arms = [
        {"name": "open_base", "model": "google/recorded-base", "label": "Old alias"},
        {"name": "open_tuned", "model": "wandb-artifact:///t/p/m:v1", "label": "Old tuned alias"},
    ]
    config = frozen_model_presentations(arms)["open_tuned"]
    display = generation_model_presentation(
        {"backend": "open_tuned", "model": config["model"]}, config=config
    )
    assert display == {
        "model_label": "ft-google/recorded-base",
        "model_version": "wandb-artifact:///t/p/m:v1",
    }
    assert "base_model" not in arms[1]


@pytest.mark.parametrize("base", ["google/trained-base", "", "bufo-gen"])
def test_forge_resolved_adapter_uses_recorded_base_and_separate_version(base):
    from bufogen.models import generation_model_presentation

    artifact = "wandb-artifact:///t/p/adapter:v4"
    row = {
        "backend": "forge_proxy",
        "model": "bufo-gen",
        "trajectory": {
            "provenance": {
                "requested_alias": "bufo-gen",
                "base_model": base,
                "resolved_serving_models": [artifact],
            }
        },
    }
    display = generation_model_presentation(row)
    assert display["model_label"] == (
        "ft-google/trained-base"
        if base == "google/trained-base"
        else "ft-model (base ID unrecorded)"
    )
    assert display["model_version"] == artifact
    assert display["routing_alias"] == "bufo-gen"
