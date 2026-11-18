"""Synthetic local evidence checks; no generation, scoring or service requests."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from bufogen.evaluation import benchmark
from bufogen.evaluation.scorers.scene_quality import failure_result, scene_result
from bufogen.generation import BufoGeneration

PATH = Path(__file__).resolve().parents[1] / "docs/forge-contact-pilot/fit_compare.py"
SPEC = importlib.util.spec_from_file_location("contact_fit_compare_tests", PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compare.encoded(value))


def save_rows(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    prompts = [
        {
            "id": f"train-{i}",
            "prompt": f"Synthetic Bufo request {i}.",
            "family": f"train_family_{i}",
            "split": "train",
            "requirements": ["Visible contact"],
        }
        for i in range(24)
    ]
    save(tmp_path / "train-prompts.json", prompts)
    checkpoint = "wandb-artifact:///synthetic/project/fixed:v0"
    identity = {"checkpoint": checkpoint, "finetune_id": "fixed-job", "recipe": compare.RECIPE}
    exclusions = {
        "heldout_ids": [f"excluded-{i}" for i in range(56)],
        "heldout_families": [f"excluded_family_{i}" for i in range(20)],
    }
    monkeypatch.setattr(compare, "candidate_identity", lambda _: (identity, exclusions, {}))
    source = compare.HERE / "evaluate.py"
    source_key = str(source.relative_to(compare.ROOT))
    configurations, fixture_hashes = {}, None
    for name, (directory, arm_name) in compare.ARMS.items():
        folder = tmp_path / directory
        folder.mkdir()
        protocol_folder = folder / "protocol-evidence"
        protocol_folder.mkdir()
        for filename, content in {
            "fixture-manifest.json": b"synthetic fixture",
            "fixture-assistant-review.json": b"synthetic review",
            "fixture-verifier.py": source.read_bytes(),
        }.items():
            (protocol_folder / filename).write_bytes(content)
        evidence = {p.name: compare.sha(p) for p in protocol_folder.iterdir()}
        fixture_hashes = {
            "manifest_sha256": evidence["fixture-manifest.json"],
            "review_sha256": evidence["fixture-assistant-review.json"],
        }
        frozen = folder / "frozen" / source_key
        frozen.parent.mkdir(parents=True)
        frozen.write_bytes(source.read_bytes())
        model = compare.BASE if name == "baseline" else checkpoint
        config = {
            "prompts": prompts,
            "prompt_hash": benchmark.digest(prompts),
            "prompt_version": "bufo-contact-v2-synthetic",
            "arms": [
                {
                    "name": arm_name,
                    "model": model,
                    "label": model,
                    "model_env": "synthetic",
                    "base_model": compare.BASE,
                    "tuning_method": "" if name == "baseline" else "sft",
                    "provider": "coreweave",
                    "sampling": {"temperature": 0.7, "max_tokens": 6000},
                }
            ],
            "source_hashes": {source_key: compare.sha(source)},
            "policy": {"concurrency": 8, "max_tokens": 6000, "max_turns_per_attempt": 8},
            "runtime": {"instructions_hash": "synthetic"},
            "scorer": {"threshold": 0.65},
        }
        configurations[name] = config
        settings = {
            "configuration": config,
            "configuration_hash": benchmark.digest(config),
            "comparison_fingerprint": benchmark.comparison_fingerprint(config),
            "run_id": name,
        }
        save(folder / "benchmark.json", settings)
        save(
            folder / "experiment-protocol.json",
            {
                "schema_version": 1,
                "configuration": config,
                "configuration_hash": benchmark.digest(config),
                "fixture_evidence": fixture_hashes,
                "evidence_sha256": evidence,
            },
        )
        original, scored = [], []
        for i, prompt in enumerate(prompts):
            missing = name == "baseline" and i == 1
            failed_turn = name == "baseline" and i in (0, 1)
            png = folder / f"synthetic-{i}.png"
            if not missing:
                Image.new("RGBA", (512, 512), (i, 0, 0, 255)).save(png)
            row = BufoGeneration(
                id=f"{name}-{i}",
                prompt_id=prompt["id"],
                prompt=prompt["prompt"],
                backend=arm_name,
                model=model,
                group_id=f"group-{name}-{i}",
                svg="",
                valid=not missing,
                latency_s=1,
                png_path=str(png) if not missing else None,
                error="synthetic failed final reply" if failed_turn else None,
                prompt_version=config["prompt_version"],
                benchmark_metadata=benchmark._metadata(prompt, name, config)
                | {
                    "status": "generation_error" if failed_turn else "completed",
                    "png_sha256": compare.sha(png) if not missing else None,
                },
            )
            original.append(row.to_dict())
            score = (
                failure_result("missing_image")
                if missing
                else scene_result(
                    {
                        "bufo_present": True,
                        "similarity": 1,
                        "anatomy": 1,
                        "composition": 1,
                        "readability": 1,
                        "reason": "Synthetic score.",
                        "prompt_checks": [
                            {
                                "requirement": "Visible contact",
                                "satisfied": True,
                                "evidence": "Synthetic check.",
                            }
                        ],
                    },
                    prompt["requirements"],
                )
            )
            score.update(
                generation_id=row.id,
                scorer_configuration=config["scorer"],
                requirements=prompt["requirements"],
            )
            row.scores = {"scene_quality": score}
            scored.append(row.to_dict())
        save_rows(folder / "manifest.jsonl", original)
        save_rows(folder / "scored.jsonl", scored)
        save(
            folder / "summary.json",
            benchmark._summary([BufoGeneration(**r) for r in scored], settings),
        )
    monkeypatch.setattr(
        compare, "current_configurations", lambda *_: (configurations, fixture_hashes)
    )
    return tmp_path, configurations


def write_reviews(data, candidate_failures=4):
    compare.create_packets(data)
    folder = data["root"] / "fit-comparison/visual-review"
    ordered = compare.indexed(data)
    for reviewer in range(3):
        packet = folder / f"reviewer-{reviewer}"
        inputs = compare.read(packet / "inputs.json")
        decisions = []
        for item in inputs:
            source = ordered[item["index"]]
            candidate_fail = (
                source["arm"] == "candidate"
                and int(source["generation"]["prompt_id"].split("-")[-1]) >= 24 - candidate_failures
            )
            passed = bool(item["png_path"]) and not candidate_fail
            decisions.append(
                {key: item[key] for key in ("index", "label", "png_sha256")}
                | {
                    "reviewer_kind": "assistant",
                    "passed": passed,
                    "evidence": "Synthetic visual review.",
                    "failed_requirements": [] if passed else item["requirements"],
                }
            )
        save_rows(packet / "decisions.jsonl", decisions)
        (packet / "review-notes.md").write_text("Synthetic review; no real generated images.")


def test_all_48_attempts_and_partial_image_survive_neutral_packet_creation(comparison):
    root, _ = comparison
    data = compare.validate(root)
    result = compare.create_packets(data)
    assert result["attempts"] == 48
    folder = root / "fit-comparison/visual-review"
    assert len(list((folder / "images").glob("*.png"))) == 47
    for reviewer in range(3):
        packet = folder / f"reviewer-{reviewer}"
        inputs = compare.read(packet / "inputs.json")
        assert len(inputs) == 16
        assert all(
            set(item) == {"index", "label", "prompt", "requirements", "png_path", "png_sha256"}
            for item in inputs
        )
        assert all(
            item["png_path"] is None or "/visual-review/images/image-" in item["png_path"]
            for item in inputs
        )
        sheets = list(packet.glob("sheet-*.png"))
        assert len(sheets) == 4
        with Image.open(sheets[0]) as image:
            assert image.size == (1080, 1800)
    before = (folder / "packet-manifest.json").read_bytes()
    compare.create_packets(data)
    assert (folder / "packet-manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "mutation", ["missing_attempt", "duplicate_attempt", "png_bytes", "missing_png", "source_bytes"]
)
def test_missing_or_corrupted_generation_evidence_is_rejected(comparison, mutation):
    root, configurations = comparison
    folder = root / "fit-baseline"
    manifest = folder / "manifest.jsonl"
    rows = compare.jsonl(manifest)
    if mutation == "missing_attempt":
        rows.pop()
        save_rows(manifest, rows)
    elif mutation == "duplicate_attempt":
        rows[-1] = deepcopy(rows[0])
        save_rows(manifest, rows)
    elif mutation == "png_bytes":
        Path(rows[0]["png_path"]).write_bytes(b"changed")
    elif mutation == "missing_png":
        Path(rows[0]["png_path"]).unlink()
    else:
        relative = next(iter(configurations["baseline"]["source_hashes"]))
        (folder / "frozen" / relative).write_text("changed")
    with pytest.raises((ValueError, OSError)):
        compare.validate(root)


@pytest.mark.parametrize("field", ["policy", "source_hashes", "runtime", "scorer", "prompts"])
def test_matching_harness_requires_every_non_arm_condition(comparison, field):
    _, configurations = comparison
    changed = deepcopy(configurations)
    changed["candidate"][field] = {"changed": True}
    with pytest.raises(ValueError, match="different"):
        compare.verify_arm_conditions(changed)


def test_arm_settings_cannot_hide_provider_or_sampling_difference(comparison):
    changed = deepcopy(comparison[1])
    changed["candidate"]["arms"][0]["sampling"]["temperature"] = 0
    with pytest.raises(ValueError, match="serving/sampling"):
        compare.verify_arm_conditions(changed)


def test_fit_gate_requires_20_same_attempt_conjunction_passes_and_completed_turns(comparison):
    root, _ = comparison
    data = compare.validate(root)
    write_reviews(data)
    result = compare.summarize(data)
    assert result["gate"]["passed"] is True
    assert result["models"]["candidate"]["automated_passes"] == 24
    assert result["models"]["candidate"]["combined_passes"] == 20
    assert result["models"]["candidate"]["disagreements"] == 4
    baseline = result["models"]["baseline"]
    assert baseline["attempts"] == 24 and baseline["generation_errors"] == 2
    assert baseline["assistant_image_passes"] == 23
    assert (
        baseline["assistant_completed_passes"]
        == baseline["automated_passes"]
        == baseline["combined_passes"]
        == 22
    )
    assert compare.summarize(data) == result


def test_19_combined_passes_fail_fixed_gate(comparison):
    data = compare.validate(comparison[0])
    write_reviews(data, candidate_failures=5)
    result = compare.summarize(data)
    assert result["models"]["candidate"]["combined_passes"] == 19
    assert result["gate"]["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    ["omit_review", "duplicate_review", "changed_copy", "omit_score", "alter_scored_generation"],
)
def test_incomplete_or_corrupted_review_and_scoring_evidence_cannot_freeze_summary(
    comparison, mutation
):
    root, _ = comparison
    data = compare.validate(root)
    write_reviews(data)
    if mutation in {"omit_review", "duplicate_review"}:
        path = root / "fit-comparison/visual-review/reviewer-0/decisions.jsonl"
        values = compare.jsonl(path)
        if mutation == "omit_review":
            values.pop()
        else:
            values[-1] = deepcopy(values[0])
        save_rows(path, values)
    elif mutation == "changed_copy":
        next((root / "fit-comparison/visual-review/images").glob("*.png")).write_bytes(b"changed")
    else:
        path = root / "fit-candidate/scored.jsonl"
        values = compare.jsonl(path)
        if mutation == "omit_score":
            values.pop()
        else:
            values[0]["latency_s"] = 999
        save_rows(path, values)
    with pytest.raises(ValueError):
        compare.summarize(data)
    assert not (root / "fit-comparison/comparison.json").exists()


def test_default_is_validation_only(comparison):
    root, _ = comparison
    result = compare.run(root)
    assert result["network_calls"] == 0 and result["attempts"] == 48
    assert not (root / "fit-comparison").exists()


def test_generation_change_after_validation_blocks_packet_creation(comparison):
    data = compare.validate(comparison[0])
    path = Path(data["rows"][0]["generation"]["png_path"])
    path.write_bytes(b"changed after initial validation")
    with pytest.raises(ValueError, match="changed before freezing"):
        compare.create_packets(data)
    assert not (comparison[0] / "fit-comparison/visual-review").exists()


@pytest.fixture
def candidate(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    source, upload = training / "reviewed-examples.jsonl", training / "upload-entries.json"
    source.write_text("synthetic immutable source")
    upload.write_text("synthetic canonical upload bytes")
    witness = training / "source-witness"
    witness.write_text("synthetic reviewed source witness")
    receipt = {
        "source_sha256": compare.sha(source),
        "upload_sha256": compare.sha(upload),
        "contract_sha256": "synthetic-contract",
        "entry_counts": {"train": 24, "val": 0},
        "training_request_policy": compare.RECIPE,
        "heldout_ids": ["excluded"],
        "heldout_families": ["excluded_family"],
        "source_files": [{"path": str(witness), "sha256": compare.sha(witness)}],
    }
    save(training / "source-receipt.json", receipt)
    spec = {
        "candidate": "contact-v2-fit",
        "base_model": compare.BASE,
        "model_name": compare.MODEL_NAME,
        "display_model_id": f"ft-{compare.BASE}",
        "initialization": "independent_base_model",
        "training_config": compare.RECIPE,
        "source_receipt_sha256": compare.sha(training / "source-receipt.json"),
        "source_path": str(source),
        "source_sha256": receipt["source_sha256"],
        "upload_sha256": receipt["upload_sha256"],
        "contract_sha256": receipt["contract_sha256"],
        "dataset_name": "synthetic-dataset",
        "project": {
            "entity": "synthetic",
            "project": "synthetic",
            "project_id": "synthetic-project",
        },
    }
    save(training / "candidate/candidate.json", spec)
    state = (
        {
            key: spec[key]
            for key in (
                "base_model",
                "model_name",
                "dataset_name",
                "source_path",
                "source_sha256",
                "upload_sha256",
            )
        }
        | spec["project"]
        | {
            "training_config": compare.RECIPE,
            "heldout_ids": receipt["heldout_ids"],
            "heldout_families": receipt["heldout_families"],
            "status": "deployed",
            "pending_operation": None,
            "finetune_id": "synthetic-job",
            "entries_imported": True,
            "entry_counts": {"train": 24, "val": 0},
            "model_artifact_ref": "wandb-artifact:///synthetic/project/fixed:v0",
        }
    )
    save(training / "candidate/forge-state.json", state)
    return tmp_path


def test_candidate_identity_binds_fixed_recipe_source_and_immutable_final_checkpoint(candidate):
    identity, _, hashes = compare.candidate_identity(candidate)
    assert identity["recipe"] == compare.RECIPE
    assert identity["checkpoint"].endswith(":v0")
    assert identity["finetune_id"] == "synthetic-job"
    assert str(candidate / "training/source-witness") in hashes
    state_path = candidate / "training/candidate/forge-state.json"
    state = compare.read(state_path)
    state["updated_at"] = "a later read-only poll"
    save(state_path, state)
    assert compare.candidate_identity(candidate)[0] == identity


@pytest.mark.parametrize(
    "mutation",
    ["epoch_selection", "mutable_alias", "pending", "source", "upload", "witness", "state_recipe"],
)
def test_candidate_drift_or_unresolved_state_is_rejected(candidate, mutation):
    training = candidate / "training"
    if mutation == "epoch_selection":
        path = training / "candidate/candidate.json"
        spec = compare.read(path)
        spec["training_config"]["epochs"] = 3
        save(path, spec)
    elif mutation in {"source", "upload", "witness"}:
        filename = {
            "source": "reviewed-examples.jsonl",
            "upload": "upload-entries.json",
            "witness": "source-witness",
        }[mutation]
        (training / filename).write_text("changed")
    else:
        path = training / "candidate/forge-state.json"
        state = compare.read(path)
        if mutation == "mutable_alias":
            state["model_artifact_ref"] = "wandb-artifact:///synthetic/project/fixed:latest"
        elif mutation == "pending":
            state["pending_operation"] = {"name": "create_finetune"}
        else:
            state["training_config"]["learning_rate"] = 1e-3
        save(path, state)
    with pytest.raises(ValueError):
        compare.candidate_identity(candidate)
