"""Offline tests for restoring a frozen experiment runtime."""

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/restore_experiment_runtime.py"
SPEC = importlib.util.spec_from_file_location("restore_experiment_runtime_tests", PATH)
restore_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore_mod)
restore = restore_mod.restore
RestoreError = restore_mod.RestoreError


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def fake_project(tmp_path):
    root = tmp_path / "proj"
    write(root / "pyproject.toml", "[project]\n")
    write(root / "uv.lock", "lock\n")
    write(root / ".python-version", "3.12\n")
    write(root / "src/bufogen/agent.py", "NEW")
    write(root / "src/bufogen/settings.py", "SETTINGS")
    write(root / "tests/test_x.py", "def test_x(): pass\n")
    write(root / "docs/forge-x/a.py", "A")
    write(root / ".env", "SECRET=1")
    write(root / ".env.example", "EXAMPLE=1")
    write(root / "src/bufogen/__pycache__/agent.pyc", "x")
    write(root / "runs/old/benchmark.json", "{}")
    for junk in (".venv", "wandb", ".pytest_cache", ".ruff_cache", ".superpowers"):
        write(root / junk / "junk.txt", "x")
    run = root / "runs/exp/run1"
    write(run / "frozen/src/bufogen/agent.py", "OLD")
    benchmark = {
        "run_id": "run1",
        "configuration_hash": "cfg123",
        "configuration": {
            "source_hashes": {"src/bufogen/agent.py": sha("OLD")},
            "scorer": {
                "judge_backend": "closed_quality_bar",
                "judge_model": "gpt-6.1-sol",
            },
            "arms": [
                {
                    "name": "open_base",
                    "model": "google/gemma-4-26B-A4B-it",
                    "model_env": "OPEN_MODEL",
                },
                {"name": "sol", "model": "gpt-6.1-sol", "model_env": "SOL_MODEL"},
            ],
            "runtime": {"packages": {"weave": "0.53.11"}},
        },
    }
    write(run / "benchmark.json", json.dumps(benchmark))
    return SimpleNamespace(root=root, run=run)


def test_frozen_file_overrides_current_and_hashes_verify(fake_project, tmp_path):
    runtime = restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)
    assert (tmp_path / "t/src/bufogen/agent.py").read_text() == "OLD"
    assert runtime["restored"] == {"src/bufogen/agent.py": sha("OLD")}
    assert "src/bufogen/settings.py" in runtime["from_current_tree"]
    assert "src/bufogen/agent.py" not in runtime["from_current_tree"]
    assert (tmp_path / "t/tests/test_x.py").exists()
    assert (tmp_path / "t/docs/forge-x/a.py").exists()
    on_disk = json.loads((tmp_path / "t/RUNTIME.json").read_text())
    assert on_disk == runtime
    assert runtime["run_id"] == "run1" and runtime["configuration_hash"] == "cfg123"
    assert runtime["recorded_packages"] == {"weave": "0.53.11"}
    assert runtime["installed_packages"] is None and runtime["package_differences"] is None
    assert (tmp_path / "t/README.md").exists()


def test_scaffolding_excludes_junk_and_secrets(fake_project, tmp_path):
    restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)
    t = tmp_path / "t"
    assert not (t / ".env").exists()
    assert (t / ".env.example").exists()
    for name in ("runs", ".venv", "wandb", ".pytest_cache", ".ruff_cache", ".superpowers"):
        assert not (t / name).exists()
    assert not (t / "src/bufogen/__pycache__").exists()


def test_frozen_docs_overlay_current_docs(fake_project, tmp_path):
    write(fake_project.run / "frozen/docs/forge-x/a.py", "FROZEN-A")
    restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)
    assert (tmp_path / "t/docs/forge-x/a.py").read_text() == "FROZEN-A"


def test_missing_frozen_file_fails_naming_path(fake_project, tmp_path):
    (fake_project.run / "frozen/src/bufogen/agent.py").unlink()
    with pytest.raises(RestoreError, match="src/bufogen/agent.py"):
        restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)


def test_tampered_frozen_file_fails(fake_project, tmp_path):
    (fake_project.run / "frozen/src/bufogen/agent.py").write_text("TAMPERED")
    with pytest.raises(RestoreError, match="sha256"):
        restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)


def test_refuses_nonempty_target(fake_project, tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t/x").write_text("x")
    with pytest.raises(RestoreError, match="empty"):
        restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)


def test_env_frozen_has_judge_and_arm_models_but_no_secrets(fake_project, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)
    env = (tmp_path / "t/env.frozen").read_text()
    assert "SCENE_JUDGE_BACKEND=closed_quality_bar" in env
    assert "SCENE_JUDGE_MODEL=gpt-6.1-sol" in env
    assert "BUFO_WEB_SEARCH=false" in env
    assert "OPEN_MODEL=google/gemma-4-26B-A4B-it" in env
    assert "SOL_MODEL=gpt-6.1-sol" in env
    assert "sk-secret" not in env and "API_KEY" not in env


def test_frozen_files_under_runs_and_wandb_dirs_are_restored(fake_project, tmp_path):
    for rel, text in (("src/runs/x.py", "RUNS"), ("src/wandb/y.py", "WB")):
        write(fake_project.run / "frozen" / rel, text)
    bench = json.loads((fake_project.run / "benchmark.json").read_text())
    bench["configuration"]["source_hashes"].update(
        {"src/runs/x.py": sha("RUNS"), "src/wandb/y.py": sha("WB")}
    )
    (fake_project.run / "benchmark.json").write_text(json.dumps(bench))
    restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)
    assert (tmp_path / "t/src/runs/x.py").read_text() == "RUNS"
    assert (tmp_path / "t/src/wandb/y.py").read_text() == "WB"


def test_post_overlay_hash_mismatch_raises(fake_project, tmp_path, monkeypatch):
    real = restore_mod._copy_all

    def corrupt(src, dst):
        real(src, dst)
        if dst.name == "t":
            (dst / "src/bufogen/agent.py").write_text("CLOBBERED")

    monkeypatch.setattr(restore_mod, "_copy_all", corrupt)
    with pytest.raises(RestoreError, match="after overlay"):
        restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)


def test_readme_has_real_commands(fake_project, tmp_path):
    restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)
    readme = (tmp_path / "t/README.md").read_text()
    assert "set -a; . ./env.frozen; set +a" in readme
    assert "BUFO_ROOT=$PWD .venv/bin/python -m pytest -q" in readme
    assert (
        "BUFO_ROOT=$PWD .venv/bin/bufogen --stage benchmark-score --output ./rescore-copy" in readme
    )
    assert "uv sync --locked --extra rl" in readme


def test_parse_freeze_and_package_differences():
    installed = restore_mod.parse_freeze(
        "weave==0.53.11\nOpenAI_Agents==0.21.0\nfoo @ git+https://x/y\n# c\n\n-e /p\n"
    )
    assert installed["weave"] == "0.53.11"
    assert installed["openai-agents"] == "0.21.0"
    assert installed["foo"] == "@ git+https://x/y"
    diff = restore_mod.package_differences(
        {"weave": "0.53.11", "openai-agents": "0.20.0", "cwsandbox": "1.18.0"}, installed
    )
    assert diff == {
        "openai-agents": {"recorded": "0.20.0", "installed": "0.21.0"},
        "cwsandbox": {"recorded": "1.18.0", "installed": None},
    }
    assert restore_mod.package_differences(None, installed) == {}


def _fake_run_factory(calls, freeze_out):
    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout=freeze_out if "freeze" in cmd else "")

    return fake_run


def test_sync_runs_uv_with_env_and_cwd_and_records_packages(fake_project, tmp_path, monkeypatch):
    target = tmp_path / "t"
    restore(fake_project.run, target, project_root=fake_project.root)
    calls = []
    monkeypatch.setattr(
        restore_mod.subprocess, "run", _fake_run_factory(calls, "weave==0.99\nopenai==3.26.1\n")
    )
    runtime = restore_mod.sync_runtime(target, project_root=fake_project.root)
    sync_cmd, sync_kw = calls[0]
    assert sync_cmd == ["uv", "sync", "--locked", "--extra", "rl"]
    assert sync_kw["env"]["UV_PROJECT_ENVIRONMENT"] == str(target.resolve() / ".venv")
    assert Path(sync_kw["cwd"]).resolve() == target.resolve()
    freeze_cmd = calls[1][0]
    assert freeze_cmd[:3] == ["uv", "pip", "freeze"]
    assert str(target.resolve() / ".venv/bin/python") in freeze_cmd
    assert runtime["installed_packages"] == {"weave": "0.99", "openai": "3.26.1"}
    assert runtime["package_differences"] == {"weave": {"recorded": "0.53.11", "installed": "0.99"}}
    assert json.loads((target / "RUNTIME.json").read_text()) == runtime


@pytest.fixture
def local_tz(monkeypatch):
    """Pin the process time zone; wandb directory names are read as local time."""

    def pin(name):
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield pin
    monkeypatch.undo()
    time.tzset()


def test_wandb_requirements_best_effort(fake_project, tmp_path, monkeypatch, local_tz):
    local_tz("UTC")  # directory names below then read as UTC
    target = tmp_path / "t"
    restore(fake_project.run, target, project_root=fake_project.root)
    # first manifest row created 2026-10-15T16:22Z
    write(
        fake_project.run / "manifest.jsonl",
        json.dumps({"created_at": "2026-10-15T16:22:39.254416+00:00"}) + "\n",
    )
    monkeypatch.setattr(
        restore_mod.subprocess, "run", _fake_run_factory([], "weave==0.99\nopenai==3.26.1\n")
    )
    # no wandb dir at all -> null with reason, no failure
    rt = restore_mod.sync_runtime(target, project_root=fake_project.root, run_dir=fake_project.run)
    assert rt["wandb_requirements"]["packages"] is None
    assert rt["wandb_requirements"]["reason"]
    # earlier and later runs exist; pick latest that predates the first row
    write(
        fake_project.root / "wandb/run-20261015_100000-aaa/files/requirements.txt", "weave==0.1\n"
    )
    write(
        fake_project.root / "wandb/run-20261015_120000-bbb/files/requirements.txt",
        "weave==0.53.11\nopenai==3.26.1\n",
    )
    write(fake_project.root / "wandb/run-20261015_170000-ccc/files/requirements.txt", "weave==9\n")
    rt = restore_mod.sync_runtime(target, project_root=fake_project.root, run_dir=fake_project.run)
    w = rt["wandb_requirements"]
    assert w["source"].endswith("run-20261015_120000-bbb/files/requirements.txt")
    assert w["packages"]["weave"] == {
        "wandb": "0.53.11",
        "recorded": "0.53.11",
        "installed": "0.99",
    }
    assert "openai" not in w["packages"]  # wandb == installed


def test_wandb_run_time_comes_from_metadata_not_local_dir_name(fake_project, tmp_path, monkeypatch):
    # Directory names are local time; 12:30 EDT is 16:30Z, after the 16:22Z first row.
    target = tmp_path / "t"
    restore(fake_project.run, target, project_root=fake_project.root)
    write(
        fake_project.run / "manifest.jsonl",
        json.dumps({"created_at": "2026-10-15T16:22:39+00:00"}) + "\n",
    )
    monkeypatch.setattr(restore_mod.subprocess, "run", _fake_run_factory([], "weave==0.99\n"))
    early = fake_project.root / "wandb/run-20261015_120000-aaa"
    late = fake_project.root / "wandb/run-20261015_123000-bbb"
    write(early / "files/requirements.txt", "weave==0.53.11\n")
    write(early / "files/wandb-metadata.json", json.dumps({"startedAt": "2026-10-15T16:00:00Z"}))
    write(late / "files/requirements.txt", "weave==9\n")
    write(late / "files/wandb-metadata.json", json.dumps({"startedAt": "2026-10-15T16:30:00Z"}))
    rt = restore_mod.sync_runtime(target, project_root=fake_project.root, run_dir=fake_project.run)
    w = rt["wandb_requirements"]
    assert w["source"].endswith("run-20261015_120000-aaa/files/requirements.txt")
    assert w["run_started_utc"] == "2026-10-15T16:00:00+00:00"


def test_wandb_dir_name_fallback_is_local_time(tmp_path, local_tz):
    local_tz("America/New_York")
    started = restore_mod._wandb_started_at(tmp_path / "run-20261015_123000-bbb")
    assert started.isoformat() == "2026-10-15T16:30:00+00:00"


SCAFFOLD_FROM_CURRENT = [
    ".env.example",
    ".python-version",
    "docs/forge-x/a.py",
    "pyproject.toml",
    "src/bufogen/settings.py",
    "tests/test_x.py",
    "uv.lock",
]


def test_from_current_tree_lists_every_non_frozen_copied_file(fake_project, tmp_path):
    write(fake_project.root / "config/models.yaml", "models: {}\n")
    write(fake_project.root / "scripts/tool.py", "T")
    write(fake_project.run / "frozen/config/brand.yaml", "FROZEN-BRAND")
    write(fake_project.root / "config/brand.yaml", "CURRENT-BRAND")
    runtime = restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)
    expected = sorted([*SCAFFOLD_FROM_CURRENT, "config/models.yaml", "scripts/tool.py"])
    assert sorted(runtime["from_current_tree"]) == expected
    # each entry is the sha256 of the file as it sits in the target
    for rel, digest in runtime["from_current_tree"].items():
        assert digest == hashlib.sha256((tmp_path / "t" / rel).read_bytes()).hexdigest()
    assert runtime["from_current_tree"]["scripts/tool.py"] == sha("T")
    # frozen overlay files (hashed or not) and excluded junk/secrets are never listed
    assert "src/bufogen/agent.py" not in runtime["from_current_tree"]
    assert "config/brand.yaml" not in runtime["from_current_tree"]
    assert not any(".env" == p or "__pycache__" in p for p in runtime["from_current_tree"])


def test_refresh_manifest_recomputes_from_current_tree_in_place(fake_project, tmp_path):
    target = tmp_path / "t"
    restore(fake_project.run, target, project_root=fake_project.root)
    runtime = json.loads((target / "RUNTIME.json").read_text())
    runtime["from_current_tree"] = ["src/bufogen/settings.py"]  # old list-only manifest
    write(target / "src/bufogen/settings.py", "EDITED-IN-TARGET")  # hash must follow the target
    runtime["installed_packages"] = {"weave": "0.53.11"}
    runtime["reconstructed"] = {"pyproject.toml": {"derived_from": "x", "sha256": "y"}}
    (target / "RUNTIME.json").write_text(json.dumps(runtime))
    write(target / "VERIFICATION.md", "not scaffold")
    write(target / "rescore-copy/summary.json", "{}")

    code = restore_mod.main(
        [
            str(fake_project.run),
            str(target),
            "--refresh-manifest",
            "--project-root",
            str(fake_project.root),
        ]
    )
    assert code == 0
    refreshed = json.loads((target / "RUNTIME.json").read_text())
    assert sorted(refreshed["from_current_tree"]) == [
        p for p in SCAFFOLD_FROM_CURRENT if p != "pyproject.toml"
    ]
    assert refreshed["from_current_tree"]["src/bufogen/settings.py"] == sha("EDITED-IN-TARGET")
    assert refreshed["installed_packages"] == {"weave": "0.53.11"}  # untouched
    assert refreshed["reconstructed"] == runtime["reconstructed"]
    assert not (target / ".venv").exists()  # no sync


@pytest.mark.parametrize("legacy_run", [True, False])
def test_restore_rejects_mixed_source_layouts_before_writing(fake_project, tmp_path, legacy_run):
    old = "src/bufogen/agent.py"
    new = "src/bufogen/agents/agent.py"
    root, run = fake_project.root, fake_project.run
    if legacy_run:
        (root / old).unlink()
        write(root / new, "NEW")
    else:
        (run / f"frozen/{old}").unlink()
        write(run / f"frozen/{new}", "OLD")
        config_path = run / "benchmark.json"
        record = json.loads(config_path.read_text())
        record["configuration"]["source_hashes"] = {new: sha("OLD")}
        config_path.write_text(json.dumps(record))
    target = tmp_path / "mixed-layout"
    with pytest.raises(RestoreError, match="--project-root"):
        restore(run, target, project_root=root)
    assert not target.exists()


def test_restore_packaged_sources_preserves_hashes(fake_project, tmp_path):
    old = "src/bufogen/agent.py"
    new = "src/bufogen/agents/agent.py"
    root, run = fake_project.root, fake_project.run
    (root / old).unlink()
    (run / f"frozen/{old}").unlink()
    write(root / new, "NEW")
    write(run / f"frozen/{new}", "OLD")
    record_path = run / "benchmark.json"
    record = json.loads(record_path.read_text())
    record["configuration"]["source_hashes"] = {new: sha("OLD")}
    record_path.write_text(json.dumps(record))
    target = tmp_path / "packaged"
    result = restore(run, target, project_root=root)
    assert (target / new).read_text() == "OLD"
    assert result["restored"] == {new: sha("OLD")}
