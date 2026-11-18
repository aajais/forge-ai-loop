"""Offline safeguards for the standalone controlled evaluation runner."""

import copy
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from bufogen.generation import BufoGeneration

SCRIPTS = Path(__file__).resolve().parents[1] / "docs/forge-gap-work"
spec = importlib.util.spec_from_file_location("paired_statistics", SCRIPTS / "paired_statistics.py")
statistics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = statistics
spec.loader.exec_module(statistics)
spec = importlib.util.spec_from_file_location("gap_evaluation", SCRIPTS / "evaluate_candidates.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("OPEN_MODEL", runner.BASE)
    monkeypatch.setenv("SOL_MODEL", runner.TEACHER)
    monkeypatch.setenv("SCENE_JUDGE_BACKEND", "closed_quality_bar")
    monkeypatch.delenv("SCENE_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_PROJECT", "project")
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setattr(runner, "environment", lambda: {"packages": {"fixture": "1"}})


def prompts(split, families, variants):
    return [
        {
            "id": f"{split}-{f}-{v}",
            "family": f"{split}-{f}",
            "split": split,
            "prompt": f"Bufo scene {f} variant {v}",
            "requirements": ["Bufo holds a mug"],
        }
        for f in range(families)
        for v in range(variants)
    ]


def make_run(root, split, candidates, *, winner="open_tuned_epoch3", selection_hash=None):
    rows_in = prompts(split, 4 if split == "validation" else 16, 6 if split == "validation" else 5)
    arms = runner.arms_for(candidates)
    config = runner._configuration(rows_in, arms, **runner.POLICY)
    settings = {
        "run_id": split,
        "configuration": config,
        "configuration_hash": runner.digest(config),
        "comparison_fingerprint": runner.comparison_fingerprint(config),
    }
    directory = root / (
        "validation-candidates" if split == "validation" else "confirmation-selected"
    )
    dump(directory / "benchmark.json", settings)
    manifest, scored = [], []
    for prompt in rows_in:
        for arm in arms:
            row = BufoGeneration(
                prompt_id=prompt["id"],
                prompt=prompt["prompt"],
                backend=arm.name,
                model=arm.model,
                group_id=f"{split}:{arm.name}:{prompt['id']}",
                svg="",
                valid=True,
                latency_s=1,
                benchmark_metadata={
                    "run_id": split,
                    "configuration_hash": settings["configuration_hash"],
                    "requirements": prompt["requirements"],
                    "family": prompt["family"],
                    "split": split,
                    "turns": [],
                    "status": "completed",
                },
            )
            manifest.append(row.to_dict())
            row.scores["scene_quality"] = {
                "status": "ok",
                "quality_failure": arm.name not in {winner, "closed_quality_bar"},
                "generation_id": row.id,
                "requirements": prompt["requirements"],
                "scorer_configuration": config["scorer"],
                **{
                    k: 0.8
                    for k in (
                        "similarity",
                        "anatomy",
                        "composition",
                        "readability",
                        "constraint_coverage",
                    )
                },
            }
            scored.append(row)
    jsonl(directory / "manifest.jsonl", manifest)
    jsonl(directory / "scored.jsonl", [r.to_dict() for r in scored])
    dump(directory / "summary.json", runner._summary(scored, settings))
    protocol = {
        "schema_version": 1,
        "configuration": config,
        "candidates": candidates,
        "curriculum_sha256": "curriculum",
        "environment": runner.environment(),
    }
    if selection_hash:
        protocol["selection_sha256"] = selection_hash
    dump(directory / "protocol.json", protocol)
    return rows_in, scored


@pytest.fixture
def experiment(tmp_path, configured, monkeypatch):
    candidates = {
        name: {
            "artifact": f"wandb-artifact:///team/project/epoch{epoch}:v1",
            "epochs": epoch,
            "state_path": str(tmp_path / f"state{epoch}.json"),
        }
        for name, epoch in runner.CANDIDATES.items()
    }
    val, _ = make_run(tmp_path, "validation", candidates)
    heldout = prompts("heldout", 16, 5)
    monkeypatch.setattr(runner, "curriculum", lambda _: (val + heldout, "curriculum"))
    return tmp_path, candidates


def test_selection_is_exclusive_and_confirmation_bound_to_exact_coverage(experiment):
    root, candidates = experiment
    selection = runner.select(root, "unused")
    assert selection["selected_arm"] == "open_tuned_epoch3"
    assert selection["selected_artifact"] == candidates["open_tuned_epoch3"]["artifact"]
    with pytest.raises(FileExistsError):
        runner.select(root, "unused")
    assert runner.selection_for_confirmation(root, "unused")[0] == selection
    path = root / "validation-candidates/scored.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after selection"):
        runner.selection_for_confirmation(root, "unused")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "score_binding", "summary"])
def test_selection_rejects_incomplete_or_relabelled_results(experiment, mutation):
    root, _ = experiment
    directory = root / "validation-candidates"
    if mutation == "summary":
        path = directory / "summary.json"
        value = runner.read(path)
        value["arms"]["open_tuned_epoch1"]["provisional_passes"] = 999
        dump(path, value)
    else:
        path = directory / "scored.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "missing":
            rows.pop()
        elif mutation == "duplicate":
            rows[-1] = rows[0]
        else:
            rows[0]["scores"]["scene_quality"]["generation_id"] = "another-image"
        jsonl(path, rows)
    with pytest.raises(ValueError):
        runner.select(root, "unused")
    assert not (root / "selection.json").exists()


async def test_confirmation_requires_selection_before_network_or_generation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Must not contact deployment or generate before selection")

    monkeypatch.setattr(runner, "verify_candidate", forbidden)
    monkeypatch.setattr(runner, "run_benchmark", forbidden)
    with pytest.raises(FileNotFoundError):
        await runner.execute("confirmation", tmp_path, "unused", {})


async def test_confirmation_rejects_runtime_drift_before_generation(experiment, monkeypatch):
    root, candidates = experiment
    runner.select(root, "unused")
    monkeypatch.setattr(runner, "verify_candidate", lambda *a, **k: candidates["open_tuned_epoch3"])
    monkeypatch.setattr(runner, "environment", lambda: {"packages": {"fixture": "2"}})
    original_read = runner.read
    monkeypatch.setattr(
        runner, "read", lambda p: {"sha256": "curriculum"} if p == "unused" else original_read(p)
    )
    with pytest.raises(ValueError, match="changed after validation"):
        await runner.execute("confirmation", root, "unused", {})
    assert not (root / "confirmation-selected").exists()


def test_report_uses_family_clusters_and_preserves_model_and_checkpoint_identity(experiment):
    root, candidates = experiment
    runner.select(root, "unused")
    chosen = "open_tuned_epoch3"
    make_run(
        root,
        "heldout",
        {chosen: candidates[chosen]},
        selection_hash=runner.sha(root / "selection.json"),
    )
    result = runner.report(root, "unused")
    assert result["improvement_over_base"]["difference"] == 1
    assert result["improvement_over_base"]["families"] == 16
    assert result["remaining_teacher_gap"]["difference"] == 0
    assert result["models"][chosen] == "ft-" + runner.BASE
    assert result["selected_artifact"] == candidates[chosen]["artifact"]
    assert result["remaining_teacher_gap"]["degenerate_interval"] is True


def test_bootstrap_resamples_families_not_individual_rows():
    rows = []
    for family in range(4):
        for variant in range(5):
            for arm in ("a", "b"):
                rows.append(
                    SimpleNamespace(
                        prompt_id=f"{family}-{variant}",
                        backend=arm,
                        valid=True,
                        benchmark_metadata={"family": str(family), "status": "completed"},
                        scores={
                            "scene_quality": {"quality_failure": not (arm == "a" and family < 2)}
                        },
                    )
                )
    result = statistics.paired_family_interval(rows, "a", "b", families=4)
    assert result["difference"] == 0.5
    assert result["ci95"] == [0, 1]
    assert result == statistics.paired_family_interval(rows, "a", "b", families=4)
    with pytest.raises(ValueError, match="coverage"):
        statistics.paired_family_interval(rows[:-1], "a", "b", families=4)


def test_selection_tiebreaks_are_declared_and_deterministic():
    def values(passes, coverage, errors):
        return {
            "provisional_passes": passes,
            "generation_errors": errors,
            "metric_means_scored_only": {"constraint_coverage": coverage},
        }

    summary = {
        "arms": {"open_tuned_epoch1": values(8, 0.7, 0), "open_tuned_epoch3": values(9, 0.6, 1)}
    }
    assert runner.rank(summary) == "open_tuned_epoch3"
    summary["arms"]["open_tuned_epoch1"] = values(9, 0.7, 2)
    assert runner.rank(summary) == "open_tuned_epoch1"
    summary["arms"]["open_tuned_epoch3"] = values(9, 0.7, 1)
    assert runner.rank(summary) == "open_tuned_epoch3"
    summary["arms"]["open_tuned_epoch1"] = values(9, 0.7, 1)
    assert runner.rank(summary) == "open_tuned_epoch1"


def test_selection_cannot_follow_confirmation_evidence(experiment):
    root, _ = experiment
    dump(root / "confirmation-selected/started.jsonl", {"already": "opened"})
    with pytest.raises(ValueError, match="must precede"):
        runner.select(root, "unused")
    assert not (root / "selection.json").exists()


def test_generation_failure_never_counts_as_pass():
    row = SimpleNamespace(
        valid=True,
        benchmark_metadata={"status": "generation_error"},
        scores={"scene_quality": {"quality_failure": False}},
    )
    assert statistics.strict_pass(row) == 0


def test_candidate_identity_requires_deployed_job_and_matching_artifact(tmp_path, configured):
    source = tmp_path / "reviewed.jsonl"
    source.write_text("actual reviewed source bytes")
    state = {
        "status": "deployed",
        "pending_operation": None,
        "base_model": runner.BASE,
        "training_config": {"epochs": 1, "batch_size": 2, "assistant_turns": "all"},
        "source_path": str(source),
        "source_sha256": runner.sha(source),
        "upload_sha256": "upload",
        "entity": "team",
        "project": "project",
        "project_alias": "bufo-gen",
        "finetune_id": "job",
        "dataset_id": "data",
        "model_artifact_ref": "wandb-artifact:///team/project/adapter:v1",
    }
    path = tmp_path / "forge-state.json"
    dump(path, state)
    job = {
        "id": "job",
        "status": "deployed",
        "dataset_id": "data",
        "base_model": runner.BASE,
        "model_config": {"config": state["training_config"]},
        "wandb_run_id": "run",
        "model_artifact_ref": state["model_artifact_ref"],
    }
    logged = SimpleNamespace(
        id="run", entity="team", project="project", config={"base_model": runner.BASE}
    )
    artifact = SimpleNamespace(
        id="actual-id",
        digest="actual-digest",
        name="adapter:v1",
        entity="team",
        project="project",
        metadata={},
        logged_by=lambda: logged,
    )
    api = SimpleNamespace(artifact=lambda *a, **k: artifact)

    @contextmanager
    def forge_factory(**kwargs):
        yield SimpleNamespace(get_finetune=lambda _: job)

    result = runner.verify_candidate(path, 1, api=api, forge_factory=forge_factory)
    assert result["identity"]["artifact_id"] == "actual-id"
    assert result["artifact"].endswith(":v1")
    job["status"] = "training"
    with pytest.raises(ValueError, match="deployed"):
        runner.verify_candidate(path, 1, api=api, forge_factory=forge_factory)
    job["status"] = "deployed"
    wrong = copy.copy(artifact)
    wrong.id = "other-id"
    count = iter([artifact, wrong])
    api.artifact = lambda *a, **k: next(count)
    with pytest.raises(ValueError, match="differs"):
        runner.verify_candidate(path, 1, api=api, forge_factory=forge_factory)


@pytest.fixture
def publisher(experiment, monkeypatch):
    root, _ = experiment
    spec = importlib.util.spec_from_file_location(
        "gap_publisher", SCRIPTS / "publish_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "evaluate_candidates", runner)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setattr(
        module, "Path", lambda path: root if path == "runs/forge-gap" else Path(path)
    )

    def forbidden(**kwargs):
        pytest.fail("Invalid publication evidence reached wandb.init")

    monkeypatch.setattr(module.wandb, "init", forbidden)
    return module


@pytest.mark.parametrize("mutation", ["stale_coverage", "changed_png"])
def test_publisher_rejects_changed_evidence_before_remote_publication(
    experiment, publisher, monkeypatch, mutation
):
    root, _ = experiment
    directory = root / "validation-candidates"
    if mutation == "stale_coverage":
        summary = runner.read(directory / "summary.json")
        summary["arms"]["open_tuned_epoch3"]["metric_means_scored_only"]["constraint_coverage"] = (
            0.99
        )
        dump(directory / "summary.json", summary)
    else:
        png = root / "original.png"
        png.write_bytes(b"original image evidence")
        original_hash = runner.sha(png)
        for filename in ("manifest.jsonl", "scored.jsonl"):
            path = directory / filename
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["png_path"] = str(png)
            rows[0]["benchmark_metadata"]["png_sha256"] = original_hash
            jsonl(path, rows)
        png.write_bytes(b"changed image evidence")
    monkeypatch.setattr(sys, "argv", ["publish_comparison.py", "--stage", "validation"])
    with pytest.raises(ValueError, match="summary|image changed"):
        publisher.main()
    assert not (root / "validation-visualization.json").exists()


def test_publisher_rejects_unselected_confirmation_candidate_before_remote_publication(
    experiment, publisher, monkeypatch
):
    root, candidates = experiment
    selection = runner.select(root, "unused")
    assert selection["selected_arm"] == "open_tuned_epoch3"
    make_run(
        root,
        "heldout",
        {"open_tuned_epoch1": candidates["open_tuned_epoch1"]},
        selection_hash=runner.sha(root / "selection.json"),
    )
    monkeypatch.setattr(sys, "argv", ["publish_comparison.py", "--stage", "confirmation"])
    with pytest.raises(ValueError, match="arm/configuration mismatch"):
        publisher.main()
    assert not (root / "confirmation-visualization.json").exists()


def test_publisher_accepts_matched_selected_evidence_without_network(experiment, publisher):
    root, candidates = experiment
    selection = runner.select(root, "unused")
    chosen = selection["selected_arm"]
    make_run(
        root,
        "heldout",
        {chosen: candidates[chosen]},
        selection_hash=runner.sha(root / "selection.json"),
    )
    _, rows, summary, protocol = publisher.validated_evidence(root, "confirmation")
    assert len(rows) == 240
    assert set(summary["arms"]) == {"open_base", "closed_quality_bar", chosen}
    assert protocol["candidates"][chosen]["artifact"] == selection["selected_artifact"]
