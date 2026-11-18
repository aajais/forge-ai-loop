"""Synthetic targets test local qualification and durable managed lifecycle; no network."""
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_contact_training_contract import captured

from bufogen.evaluation import benchmark
from bufogen.generation import Manifest
from bufogen.models import load_backends
from bufogen.workflows.forge_loop import AmbiguousOperation

HERE = Path(__file__).resolve().parents[1] / "docs/forge-contact-pilot"
sys.path.insert(0, str(HERE))
import contact_agent_v2
import experiment

SPEC = importlib.util.spec_from_file_location("contact_trainer_tests", HERE / "train_contact.py")
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path / "learnability"
    root.mkdir()
    prompts = [{
        "id": f"synthetic-{index}", "family": f"synthetic_family_{index}",
        "split": "train" if index < 24 else "heldout",
        "relation_class": f"relation_{index // 6 if index < 24 else (index - 24) // 2}",
        "prompt": f"Draw synthetic standing Bufo number {index}.",
        "requirements": ["Synthetic visible contact"],
    } for index in range(32)]
    for name, rows in (("curriculum.json", prompts), ("train-prompts.json", prompts[:24]),
                       ("diagnostic-prompts.json", prompts[24:])):
        (root / name).write_bytes(trainer.encoded(rows))
    reserved = root / "reserved-identity-receipt.json"
    reserved.write_bytes(trainer.encoded({
        "heldout_ids": [f"reserved-{i}" for i in range(48)],
        "heldout_families": [f"reserved_family_{i}" for i in range(12)],
    }))
    monkeypatch.setattr(trainer, "RESERVED_RECEIPT", reserved)
    with experiment.harness():
        config = benchmark._configuration(
            prompts[:24], [load_backends()["closed_quality_bar"]], **experiment.POLICY,
        )
    # Configuration hashes and loaded code are real; only fixture acceptance is synthetic.
    teacher = root / "teacher"
    teacher.mkdir()
    settings = {
        "run_id": "synthetic-run", "configuration": config,
        "configuration_hash": benchmark.digest(config),
        "comparison_fingerprint": benchmark.comparison_fingerprint(config),
    }
    (teacher / "benchmark.json").write_bytes(trainer.encoded(settings))
    evidence = teacher / "protocol-evidence"
    evidence.mkdir()
    (evidence / "fixture-manifest.json").write_text("synthetic fixture manifest")
    (evidence / "fixture-assistant-review.json").write_text("synthetic explicit fixture review")
    (evidence / "fixture-verifier.py").write_bytes((HERE / "evaluate.py").read_bytes())
    hashes = {path.name: trainer.sha(path.read_bytes()) for path in evidence.iterdir()}
    fixtures = {
        "manifest_sha256": hashes["fixture-manifest.json"],
        "review_sha256": hashes["fixture-assistant-review.json"],
    }
    (teacher / "experiment-protocol.json").write_bytes(trainer.encoded({
        "schema_version": 1, "configuration": config,
        "configuration_hash": benchmark.digest(config),
        "fixture_evidence": fixtures, "evidence_sha256": hashes,
    }))
    for relative in config["source_hashes"]:
        target = teacher / "frozen" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((trainer.ROOT / relative).read_bytes())
    monkeypatch.setattr(trainer, "current_configuration", lambda _: (config, fixtures))
    rows, reviews = [], []
    for index, prompt in enumerate(prompts[:24]):
        folder = root / "synthetic-captures" / str(index)
        folder.mkdir(parents=True)
        row, review, _ = captured(folder)
        row.id, row.prompt_id, row.group_id = f"generation-{index}", prompt["id"], f"group-{index}"
        row.prompt, row.prompt_version = prompt["prompt"], config["prompt_version"]
        trajectory = row.trajectory
        trajectory["system_instructions"] = contact_agent_v2.instructions()
        trajectory["messages"][0]["content"] = contact_agent_v2.instructions()
        trajectory["messages"][1]["content"] = row.prompt
        trajectory["provenance"].update(
            prompt_id=row.prompt_id, group_id=row.group_id, prompt_version=row.prompt_version,
            tool_contract=contact_agent_v2.TOOL_CONTRACT,
        )
        row.benchmark_metadata = benchmark._metadata(prompt, settings["run_id"], config) | {
            "status": "completed", "png_sha256": review["png_sha256"],
        }
        review.update(
            source_generation_id=row.id, prompt_id=row.prompt_id, prompt_family=prompt["family"],
        )
        Manifest(teacher / "manifest.jsonl").append(row)
        rows.append(row)
        reviews.append(review)
    (root / "reviews").mkdir()
    (root / "reviews/decisions.jsonl").write_bytes(trainer.jsonl(reviews))
    return root, rows, reviews, config


def test_prepare_binds_exact_clean_contact_upload_and_is_idempotent(corpus):
    root, rows, _, _ = corpus
    receipt = trainer.prepare(root)
    assert receipt["entry_counts"] == {"train": 24, "val": 0}
    assert len(receipt["heldout_ids"]) == 56
    assert len(receipt["heldout_families"]) == 20
    assert receipt["nominal_optimizer_updates"] == 72
    assert receipt["teacher_model"] == "gpt-6.1-sol"
    assert trainer.prepare(root) == receipt
    entries = json.loads((root / "training/upload-entries.json").read_text())
    assert trainer.sha((root / "training/upload-entries.json").read_bytes()) == receipt["upload_sha256"]
    assert len(entries) == 24 and {entry["split"] for entry in entries} == {"train"}
    by_prompt = {entry["metadata"]["prompt_id"]: entry for entry in entries}
    for row in rows:
        assert by_prompt[row.prompt_id]["messages"] == row.trajectory["messages"]
        assert by_prompt[row.prompt_id]["tools"] == row.trajectory["tools"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reject", "png", "family"])
def test_review_failures_stop_before_creating_training_source(corpus, mutation):
    root, _, reviews, _ = corpus
    if mutation == "missing":
        reviews.pop()
    elif mutation == "duplicate":
        reviews[-1] = deepcopy(reviews[0])
    elif mutation == "reject":
        reviews[0].update(decision="reject", visual_quality_pass=False)
    elif mutation == "png":
        reviews[0]["png_sha256"] = "0" * 64
    else:
        reviews[0]["prompt_family"] = "reserved_family_0"
    (root / "reviews/decisions.jsonl").write_bytes(trainer.jsonl(reviews))
    with pytest.raises(ValueError):
        trainer.prepare(root)
    assert not (root / "training/reviewed-examples.jsonl").exists()


@pytest.mark.parametrize("mutation", ["repair", "incomplete", "wrong_teacher", "frozen_source"])
def test_collection_or_capture_mismatch_stops_preparation(corpus, mutation):
    root, rows, _, config = corpus
    if mutation == "frozen_source":
        relative = next(iter(config["source_hashes"]))
        (root / "teacher/frozen" / relative).write_text("changed")
    else:
        if mutation == "repair":
            rows[0].tool_calls = 2
        elif mutation == "incomplete":
            rows.pop()
        else:
            rows[0].model = trainer.STUDENT_MODEL
        (root / "teacher/manifest.jsonl").write_bytes(trainer.jsonl([row.to_dict() for row in rows]))
    with pytest.raises(ValueError):
        trainer.prepare(root)
    assert not (root / "training/reviewed-examples.jsonl").exists()


def test_diagnostic_or_reserved_overlap_is_rejected(corpus):
    root, _, _, _ = corpus
    prompts = trainer.read(root / "curriculum.json")
    prompts[0]["family"] = "reserved_family_0"
    (root / "curriculum.json").write_bytes(trainer.encoded(prompts))
    with pytest.raises(ValueError, match="disjoint"):
        trainer.prepare(root)


class FakeForge:
    entity = "synthetic"
    project_alias = "project"
    base_url = "https://synthetic.invalid"
    training_base_url = "https://training.synthetic.invalid"

    def __init__(self):
        self.calls = []

    def preflight(self, *_args, **_kwargs):
        self.calls.append("preflight")

    def create_dataset(self, name):
        self.calls.append("create_dataset")
        return {"id": "dataset-synthetic", "name": name}

    def upload_entries(self, dataset, entries, dry_run):
        self.calls.append("validate_entries" if dry_run else "upload_entries")
        assert len(entries) == 24 and {e["split"] for e in entries} == {"train"}
        return {"dry_run": dry_run, "group_conflicts": [], "revision": "synthetic-revision",
                "entry_counts": {"total": 24, "train": 24, "val": 0}}

    def create_finetune(self, **kwargs):
        self.calls.append("create_finetune")
        assert kwargs == {
            "dataset_id": "dataset-synthetic", "model_name": trainer.MODEL_NAME,
            "base_model": trainer.STUDENT_MODEL, "epochs": 6, "batch_size": 2, "learning_rate": 1e-4,
        }
        return {"id": "job-synthetic", "status": "queued"}

    def get_finetune(self, job_id):
        assert job_id == "job-synthetic"
        self.calls.append("get_finetune")
        return {"id": job_id, "status": "deployed", "base_model": trainer.STUDENT_MODEL,
                "model_artifact_ref": "wandb-artifact:///synthetic/project/contact:v0"}


def project():
    return {key: getattr(FakeForge, key) for key in (
        "entity", "project_alias", "base_url", "training_base_url",
    )} | {"project": "project", "project_id": "project-synthetic"}


def test_real_durable_lifecycle_creates_once_and_status_only_polls(corpus):
    root = corpus[0]
    client = FakeForge()
    first = trainer.train_or_status("train", root, client, project())
    assert first["status"] == "queued"
    assert first["training_config"] == trainer.RECIPE
    second = trainer.train_or_status("train", root, client, project())
    assert second["model_artifact_ref"].endswith(":v0")
    assert client.calls.count("create_dataset") == client.calls.count("upload_entries") == 1
    assert client.calls.count("create_finetune") == 1
    client.calls.clear()
    trainer.train_or_status("status", root, client, project())
    assert client.calls == ["get_finetune"]


def test_pending_write_blocks_preflight_and_every_remote_call(corpus):
    root = corpus[0]
    client = FakeForge()
    trainer.train_or_status("train", root, client, project())
    path = root / "training/candidate/forge-state.json"
    state = trainer.read(path)
    state["pending_operation"] = {"name": "create_finetune"}
    path.write_bytes(trainer.encoded(state))
    client.calls.clear()
    with pytest.raises(AmbiguousOperation):
        trainer.train_or_status("train", root, client, project())
    assert not client.calls


def test_status_never_creates_a_new_workflow(corpus):
    client = FakeForge()
    with pytest.raises(ValueError, match="saved workflow"):
        trainer.train_or_status("status", corpus[0], client, project())
    assert not client.calls


@pytest.mark.parametrize("mutation", ["candidate_recipe", "state_source", "state_project", "missing_candidate"])
def test_existing_identity_mismatch_blocks_every_remote_call(corpus, mutation):
    root = corpus[0]
    client = FakeForge()
    trainer.train_or_status("train", root, client, project())
    folder = root / "training/candidate"
    spec_path, state_path = folder / "candidate.json", folder / "forge-state.json"
    if mutation == "candidate_recipe":
        spec = trainer.read(spec_path)
        spec["training_config"]["epochs"] = 7
        spec_path.write_bytes(trainer.encoded(spec))
    elif mutation == "missing_candidate":
        spec_path.unlink()
    else:
        state = trainer.read(state_path)
        state["source_sha256" if mutation == "state_source" else "project_id"] = "changed"
        state_path.write_bytes(trainer.encoded(state))
    client.calls.clear()
    with pytest.raises(ValueError):
        trainer.train_or_status("train", root, client, project())
    assert not client.calls
