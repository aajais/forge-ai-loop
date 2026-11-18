"""Offline evidence and selection safeguards for the second experiment."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from bufogen.evaluation.benchmark import _configuration, _summary, comparison_fingerprint, digest
from bufogen.generation import BufoGeneration

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "docs/forge-iteration2/evaluate.py"
    assert path.exists(), "Iteration-two evaluator has not been implemented"
    spec = importlib.util.spec_from_file_location("iteration2_evaluation", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, values):
    path.write_text("".join(json.dumps(v) + "\n" for v in values))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setenv("OPEN_MODEL", m.BASE)
    monkeypatch.setenv("SOL_MODEL", m.TEACHER)
    monkeypatch.setenv("SCENE_JUDGE_BACKEND", "closed_quality_bar")
    monkeypatch.delenv("SCENE_JUDGE_MODEL", raising=False)
    candidates = {
        name: {
            "artifact": f"wandb-artifact:///team/project/{name}:v1",
            "epochs": epoch,
            "state_path": str(tmp_path / name / "forge-state.json"),
        }
        for name, epoch in {m.PRIOR: 1, **m.CANDIDATES}.items()
    }
    val = [
        {
            "id": f"v-{i}",
            "family": f"val-{i // 6}",
            "split": "validation",
            "prompt": f"Draw Bufo holding object {i}",
            "requirements": ["Visible contact"],
        }
        for i in range(24)
    ]
    heldout = [
        {
            "id": f"h-{i}",
            "family": f"heldout-{i // 4}",
            "split": "heldout",
            "prompt": f"Draw a new Bufo action {i}",
            "requirements": ["Visible contact"],
        }
        for i in range(48)
    ]
    monkeypatch.setattr(m, "curriculum", lambda path: (val + heldout, "curriculum"))
    arms = m.arms_for(candidates)
    config = _configuration(val, arms, **m.POLICY)
    settings = {
        "run_id": "validation",
        "configuration": config,
        "configuration_hash": digest(config),
        "comparison_fingerprint": comparison_fingerprint(config),
    }
    directory = tmp_path / "validation-candidates"
    dump(directory / "benchmark.json", settings)
    manifest, scored = [], []
    for prompt in val:
        for arm in arms:
            row = BufoGeneration(
                prompt_id=prompt["id"],
                prompt=prompt["prompt"],
                backend=arm.name,
                model=arm.model,
                group_id=f"{arm.name}:{prompt['id']}",
                svg="",
                valid=True,
                latency_s=1,
                benchmark_metadata={
                    "run_id": "validation",
                    "configuration_hash": settings["configuration_hash"],
                    "requirements": prompt["requirements"],
                    "family": prompt["family"],
                    "split": "validation",
                    "turns": [],
                    "status": "completed",
                },
            )
            manifest.append(row.to_dict())
            row.scores["scene_quality"] = {
                "status": "ok",
                "quality_failure": arm.name == "open_base",
                "generation_id": row.id,
                "requirements": prompt["requirements"],
                "scorer_configuration": config["scorer"],
                "constraint_coverage": 1 if arm.name == m.PRIOR else 0.8,
                "similarity": 0.8,
                "anatomy": 0.8,
                "composition": 0.8,
                "readability": 0.8,
            }
            scored.append(row)
    jsonl(directory / "manifest.jsonl", manifest)
    jsonl(directory / "scored.jsonl", [r.to_dict() for r in scored])
    dump(directory / "summary.json", _summary(scored, settings))
    dump(
        directory / "protocol.json",
        {
            "schema_version": 1,
            "stage": "validation",
            "configuration": config,
            "candidates": candidates,
            "curriculum_sha256": "curriculum",
            "environment": {"test": True},
        },
    )
    return m, tmp_path, candidates


def test_selection_excludes_better_prior_control_and_breaks_ties_by_one_epoch(experiment):
    m, root, candidates = experiment
    selected = m.select(root, "unused")
    assert selected["selected_arm"] == "open_tuned_epoch1"
    assert selected["selected_artifact"] == candidates["open_tuned_epoch1"]["artifact"]
    assert m.selected_context(root, "unused")[0] == selected
    with pytest.raises(FileExistsError):
        m.select(root, "unused")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "score_binding", "summary"])
def test_selection_requires_every_attempt_and_bound_scores(experiment, mutation):
    m, root, _ = experiment
    folder = root / "validation-candidates"
    if mutation == "summary":
        value = m.read(folder / "summary.json")
        value["arms"]["open_tuned_epoch1"]["provisional_passes"] = 999
        dump(folder / "summary.json", value)
    else:
        values = [json.loads(line) for line in (folder / "scored.jsonl").read_text().splitlines()]
        if mutation == "missing":
            values.pop()
        elif mutation == "duplicate":
            values[-1] = values[0]
        else:
            values[0]["scores"]["scene_quality"]["generation_id"] = "wrong-image"
        jsonl(folder / "scored.jsonl", values)
    with pytest.raises(ValueError):
        m.select(root, "unused")


def test_confirmation_must_follow_selection_and_bind_unchanged_evidence(experiment):
    m, root, _ = experiment
    with pytest.raises(FileNotFoundError):
        m.selected_context(root, "unused")
    m.select(root, "unused")
    path = root / "validation-candidates/scored.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        m.selected_context(root, "unused")


def test_cannot_select_after_confirmation_has_started(experiment):
    m, root, _ = experiment
    dump(root / "confirmation-selected/protocol.json", {})
    with pytest.raises(ValueError, match="precede"):
        m.select(root, "unused")


def test_rank_prefers_passes_then_coverage_then_fewer_errors():
    m = module()
    summary = {
        "arms": {
            name: {
                "provisional_passes": 1,
                "generation_errors": 0,
                "metric_means_scored_only": {"constraint_coverage": 0.8},
            }
            for name in m.CANDIDATES
        }
    }
    assert m.rank(summary) == "open_tuned_epoch1"
    summary["arms"]["open_tuned_epoch3"]["metric_means_scored_only"]["constraint_coverage"] = 0.9
    assert m.rank(summary) == "open_tuned_epoch3"
    summary["arms"]["open_tuned_epoch1"]["provisional_passes"] = 2
    assert m.rank(summary) == "open_tuned_epoch1"
    summary["arms"]["open_tuned_epoch3"] = copy.deepcopy(summary["arms"]["open_tuned_epoch1"])
    summary["arms"]["open_tuned_epoch1"]["generation_errors"] = 1
    assert m.rank(summary) == "open_tuned_epoch3"


def test_curriculum_checks_file_hash_and_family_disjointness(tmp_path, monkeypatch):
    import yaml

    m = module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    original = ROOT / "config/forge_iteration2_prompts.yaml"
    rows = yaml.safe_load(original.read_text())
    source = tmp_path / "curriculum.yaml"
    source.write_text(yaml.safe_dump(rows))
    receipt = tmp_path / "manifest.json"
    dump(
        receipt,
        {
            "source": "curriculum.yaml",
            "sha256": m.sha(source),
            "counts": {"train": 80, "validation": 24, "heldout": 48},
        },
    )
    assert len(m.curriculum(receipt)[0]) == 152
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        m.curriculum(receipt)
    rows[-1]["family"] = rows[0]["family"]
    source.write_text(yaml.safe_dump(rows))
    dump(
        receipt,
        {
            "source": "curriculum.yaml",
            "sha256": m.sha(source),
            "counts": {"train": 80, "validation": 24, "heldout": 48},
        },
    )
    with pytest.raises(ValueError, match="famil"):
        m.curriculum(receipt)


def test_protocol_resume_refuses_unbound_or_changed_outputs(tmp_path):
    m = module()
    m.freeze_protocol(tmp_path / "run", {"stage": "validation"})
    m.freeze_protocol(tmp_path / "run", {"stage": "validation"})
    with pytest.raises(ValueError):
        m.freeze_protocol(tmp_path / "run", {"stage": "confirmation"})
    dump(tmp_path / "orphan/manifest.json", {})
    with pytest.raises(ValueError):
        m.freeze_protocol(tmp_path / "orphan", {"stage": "validation"})


def test_fit_checks_fixed_training_prompts_and_reviewed_source(tmp_path):
    m = module()
    prompts, targets = [], []
    for i in range(16):
        prompt = {
            "id": f"train-{i}-contact",
            "family": f"train-{i}",
            "split": "train",
            "prompt": f"Contact {i}",
            "curriculum_metadata": {"difficulty": "contact"},
        }
        prompts.append(prompt)
        targets.append(
            {
                "prompt_id": prompt["id"],
                "prompt_family": prompt["family"],
                "partition": "train",
                "prompt": prompt["prompt"],
                "source_generation_id": f"image-{i}",
                "training_review": {
                    "decision": "accept",
                    "visual_quality_pass": True,
                    "partition": "train",
                    "prompt_id": prompt["id"],
                    "source_generation_id": f"image-{i}",
                    "png_sha256": "png",
                },
            }
        )
    source = tmp_path / "training/reviewed-examples.jsonl"
    source.parent.mkdir()
    jsonl(source, targets)
    candidate = {"source_sha256": m.sha(source)}
    chosen, provenance = m.fixed_fit_prompts(prompts, tmp_path, candidate)
    assert len(chosen) == len(provenance) == 16
    targets[0]["training_review"]["decision"] = "reject"
    jsonl(source, targets)
    with pytest.raises(ValueError, match="source changed"):
        m.fixed_fit_prompts(prompts, tmp_path, candidate)
    candidate["source_sha256"] = m.sha(source)
    with pytest.raises(ValueError, match="provenance"):
        m.fixed_fit_prompts(prompts, tmp_path, candidate)


def test_validation_protocol_cannot_mislabel_stage(experiment):
    m, root, _ = experiment
    path = root / "validation-candidates/protocol.json"
    protocol = m.read(path)
    protocol["stage"] = "confirmation"
    dump(path, protocol)
    with pytest.raises(ValueError, match="stage"):
        m.select(root, "unused")


def test_fit_evidence_returns_actual_scored_generation_objects(experiment, monkeypatch):
    m, root, candidates = experiment
    m.select(root, "unused")
    source = root / "validation-candidates"
    original_settings, protocol = (
        m.read(source / "benchmark.json"),
        m.read(source / "protocol.json"),
    )
    chosen = "open_tuned_epoch1"
    prompts = original_settings["configuration"]["prompts"][:16]
    monkeypatch.setattr(m, "fixed_fit_prompts", lambda *args: (prompts, []))
    config = copy.deepcopy(original_settings["configuration"])
    config["prompts"] = prompts
    config["prompt_hash"] = digest(prompts)
    config["arms"] = [a for a in config["arms"] if a["name"] == chosen]
    settings = {
        "run_id": "fit",
        "configuration": config,
        "configuration_hash": digest(config),
        "comparison_fingerprint": comparison_fingerprint(config),
    }
    directory = root / "training-fit-selected"
    dump(directory / "benchmark.json", settings)
    ids = {p["id"] for p in prompts}
    values = [json.loads(line) for line in (source / "manifest.jsonl").read_text().splitlines()]
    values = [r for r in values if r["prompt_id"] in ids and r["backend"] == chosen]
    scored_source = {
        r["id"]: r
        for r in [json.loads(line) for line in (source / "scored.jsonl").read_text().splitlines()]
    }
    scored = []
    for row in values:
        row["benchmark_metadata"].update(
            run_id="fit", configuration_hash=settings["configuration_hash"]
        )
        item = copy.deepcopy(row)
        item["scores"] = scored_source[row["id"]]["scores"]
        scored.append(BufoGeneration(**item))
    jsonl(directory / "manifest.jsonl", values)
    jsonl(directory / "scored.jsonl", [r.to_dict() for r in scored])
    dump(directory / "summary.json", _summary(scored, settings))
    dump(
        directory / "protocol.json",
        {
            **protocol,
            "stage": "fit",
            "configuration": config,
            "candidates": {chosen: candidates[chosen]},
            "selection_sha256": m.sha(root / "selection.json"),
        },
    )
    _, result, summary, _ = m.validated_evidence(root, "fit", "unused")
    assert len(result) == 16 and all(isinstance(row, BufoGeneration) for row in result)
    assert summary["arms"][chosen]["total"] == 16


@pytest.fixture
def candidate_files(tmp_path, monkeypatch):
    m = module()
    folder = tmp_path / "training/e1"
    source = tmp_path / "training/reviewed-examples.jsonl"
    source.parent.mkdir()
    source.write_text("{}\n")
    curriculum = tmp_path / "curriculum-manifest.json"
    dump(curriculum, {"sha256": "curriculum"})
    receipt = {
        "source_sha256": m.sha(source),
        "upload_sha256": "upload",
        "entry_counts": {"train": 70, "validation": 20},
        "source_files": [{"path": str(curriculum), "sha256": m.sha(curriculum)}],
    }
    receipt_path = tmp_path / "training/source-receipt.json"
    dump(receipt_path, receipt)
    recipe = {"epochs": 1, "batch_size": 2, "learning_rate": 1e-4, "assistant_turns": "all"}
    spec = {
        "candidate": "e1",
        "initialization": "independent_base_model",
        "base_model": m.BASE,
        "source_receipt_sha256": m.sha(receipt_path),
        "training_config": recipe,
        "source_sha256": m.sha(source),
        "upload_sha256": "upload",
    }
    dump(folder / "candidate.json", spec)
    state = {
        "training_config": recipe,
        "base_model": m.BASE,
        "source_path": str(source),
        "source_sha256": m.sha(source),
        "upload_sha256": "upload",
        "entry_counts": {"train": 70, "val": 20},
        "lineage": {
            "source_receipt_sha256": m.sha(receipt_path),
            "initialization": "independent_base_model",
        },
    }
    dump(folder / "forge-state.json", state)
    calls = []

    def verify(path, epoch):
        calls.append((path, epoch))
        return {"artifact": "wandb-artifact:///team/project/new:v1", "epochs": 1}

    monkeypatch.setattr(m.shared, "verify_candidate", verify)
    return m, tmp_path, folder, state, calls


def test_candidate_checks_reviewed_identity_before_live_lookup(candidate_files):
    m, root, folder, state, calls = candidate_files
    value = m.verify_new_candidate(root, "open_tuned_epoch1")
    assert value["artifact"].endswith(":v1")
    assert calls == [(folder / "forge-state.json", 1)]
    calls.clear()
    state["training_config"]["learning_rate"] = 2e-4
    dump(folder / "forge-state.json", state)
    with pytest.raises(ValueError, match="recipe"):
        m.verify_new_candidate(root, "open_tuned_epoch1")
    assert calls == []


def test_candidate_review_source_must_bind_current_curriculum(candidate_files):
    m, root, _, _, calls = candidate_files
    dump(root / "curriculum-manifest.json", {"sha256": "another-curriculum"})
    with pytest.raises(ValueError, match="source"):
        m.verify_new_candidate(root, "open_tuned_epoch1")
    assert calls == []
