"""Offline guardrails for the matched follow-up evaluator."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "docs/forge-gap-work"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "lr_evaluation", SCRIPTS / "evaluate_lr_candidates.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    source = tmp_path / "targets.jsonl"
    source.write_text("reviewed source bytes")
    monkeypatch.setattr(runner, "TRAINING_SHA256", runner.sha(source))
    config = {"epochs": 1, "batch_size": 2, "assistant_turns": "all", "learning_rate": 2e-5}
    state = {
        "status": "deployed",
        "pending_operation": None,
        "base_model": runner.BASE,
        "training_config": config,
        "source_path": str(source),
        "source_sha256": runner.sha(source),
        "upload_sha256": runner.UPLOAD_SHA256,
        "entry_counts": {"train": 124, "val": 23},
        "model_artifact_ref": "wandb-artifact:///team/project/adapter:v1",
    }
    path = tmp_path / "state.json"
    dump(path, state)
    spec = {
        "state_path": str(path),
        "training_config": config,
        "metadata": {"requested_peak_lr": 2e-5},
    }
    calls = []

    def verify(*args, **kwargs):
        calls.append(args)
        return {
            "artifact": state["model_artifact_ref"],
            "training_config": state["training_config"],
            "state_sha256": runner.sha(path),
        }

    monkeypatch.setattr(runner, "verify_candidate", verify)
    return path, state, spec, calls


def test_exact_config_and_source_precede_live_identity_verification(candidate):
    _, _, spec, calls = candidate
    result = runner.verify_declared_candidate(spec)
    assert len(calls) == 1
    assert result["metadata"] == {"requested_peak_lr": 2e-5}


@pytest.mark.parametrize(
    "mutation", ["lr", "batch", "turns", "source", "upload", "status", "mutable", "counts"]
)
def test_invalid_candidate_rejected_before_network(candidate, mutation):
    path, state, spec, calls = candidate
    # Keep independent expected plan config when mutating the actual receipt.
    spec = copy.deepcopy(spec)
    if mutation == "lr":
        state["training_config"]["learning_rate"] = 1e-4
    elif mutation == "batch":
        state["training_config"]["batch_size"] = 4
    elif mutation == "turns":
        state["training_config"]["assistant_turns"] = "last"
    elif mutation == "source":
        Path(state["source_path"]).write_text("changed")
    elif mutation == "upload":
        state["upload_sha256"] = "changed"
    elif mutation == "status":
        state["status"] = "training"
    elif mutation == "mutable":
        state["model_artifact_ref"] = "wandb-artifact:///team/project/adapter:latest"
    else:
        state["entry_counts"]["train"] = 123
    dump(path, state)
    with pytest.raises(ValueError):
        runner.verify_declared_candidate(spec)
    assert not calls


def test_original_control_artifact_and_state_cannot_be_replaced(candidate):
    _, _, spec, calls = candidate
    spec["required_state_sha256"] = "different-control"
    with pytest.raises(ValueError):
        runner.verify_declared_candidate(spec)
    assert not calls


def test_new_protocol_requires_empty_directory_and_exact_resume(tmp_path):
    protocol = {"configuration": {"policy": {"max_tokens": 6000}}, "environment": {"version": 1}}
    directory = tmp_path / "evaluation"
    runner.freeze_protocol(directory, protocol)
    runner.freeze_protocol(directory, protocol)
    changed = copy.deepcopy(protocol)
    changed["environment"]["version"] = 2
    with pytest.raises(ValueError, match="differs"):
        runner.freeze_protocol(directory, changed)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "manifest.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="without its protocol"):
        runner.freeze_protocol(foreign, protocol)


def test_only_three_immutable_distinct_ft_arms(monkeypatch):
    monkeypatch.setenv("OPEN_MODEL", runner.BASE)
    candidates = {
        k: {"artifact": f"wandb-artifact:///team/project/{k}:v1"} for k in runner.DEFAULT_NAMES
    }
    arms = runner.arms_for(candidates)
    assert len(arms) == 3 and {a.name for a in arms} == set(candidates)
    assert all(a.label == "ft-" + runner.BASE and a.base_model == runner.BASE for a in arms)
    candidates[runner.DEFAULT_NAMES[1]]["artifact"] = candidates[runner.DEFAULT_NAMES[0]][
        "artifact"
    ]
    with pytest.raises(ValueError, match="distinct"):
        runner.arms_for(candidates)


def test_confirmation_and_original_output_reuse_are_forbidden(tmp_path):
    with pytest.raises(ValueError, match="stage"):
        runner.check_output("confirmation", tmp_path / "new", tmp_path / "old")
    with pytest.raises(ValueError, match="historical"):
        runner.check_output("validation", tmp_path / "old", tmp_path / "old")
    with pytest.raises(ValueError, match="historical"):
        runner.check_output("train-fit", tmp_path / "old/sub", tmp_path / "old")


def generation_config():
    historical = {
        "arms": [
            {
                "name": "open_tuned_epoch1",
                "model": "original",
                "provider": "wandb",
                "base_url": "https://example.invalid/v1",
                "sampling": {"temperature": 0.7, "max_tokens": 6000},
            }
        ],
        "scorer": {"judge_model": runner.TEACHER},
        "policy": {"max_turns_per_attempt": 8, "max_tokens": 6000, "requested_temperature": 0.7},
        "source_hashes": {"agent.py": "same-source"},
    }
    current = copy.deepcopy(historical)
    current["arms"] = [
        {**historical["arms"][0], "name": name, "model": name + ":v1"}
        for name in runner.DEFAULT_NAMES
    ]
    return historical, current


@pytest.mark.parametrize("mutation", ["policy", "source", "endpoint", "sampling"])
def test_changed_generation_conditions_rejected(mutation):
    historical, current = generation_config()
    runner.validate_generation_configuration(current, historical)
    if mutation == "policy":
        current["policy"]["max_tokens"] = 12000
    elif mutation == "source":
        current["source_hashes"]["agent.py"] = "changed"
    elif mutation == "endpoint":
        current["arms"][0]["base_url"] = "https://another.invalid/v1"
    else:
        current["arms"][0] = copy.deepcopy(current["arms"][0])
        current["arms"][0]["sampling"]["temperature"] = 0.2
    with pytest.raises(ValueError):
        runner.validate_generation_configuration(current, historical)


def test_cross_stage_freeze_rejects_changed_environment(tmp_path):
    historical, current = generation_config()
    protocol = {
        "configuration": current,
        "candidate_plan": {"candidates": {}},
        "candidates": {},
        "environment": {"version": 1},
        "training_source_sha256": "same",
    }
    peer = tmp_path / "validation"
    runner.freeze_run(peer, protocol, tmp_path / "absent")
    altered = copy.deepcopy(protocol)
    altered["environment"]["version"] = 2
    with pytest.raises(ValueError, match="stages differ"):
        runner.freeze_run(tmp_path / "training-fit", altered, peer)
    assert not (tmp_path / "training-fit").exists()


async def test_execution_is_matched_three_arm_diagnostic_and_publishes_native_eval(
    tmp_path, monkeypatch
):
    import bufogen.integrations.telemetry

    historical, config = generation_config()
    prompts = [{"id": f"fixed-{i}", "split": "train"} for i in range(16)]
    source = tmp_path / "source"
    source.write_text("source")
    monkeypatch.setattr(
        runner, "local_inputs", lambda *a: (prompts, [], {"configuration": historical}, source)
    )
    monkeypatch.setattr(
        runner, "declared_plan", lambda *a: {"candidates": {n: {} for n in runner.DEFAULT_NAMES}}
    )
    counter = iter(range(3))
    monkeypatch.setattr(
        runner,
        "verify_declared_candidate",
        lambda s: {"artifact": f"wandb-artifact:///team/project/a{next(counter)}:v1"},
    )
    monkeypatch.setattr(runner, "_configuration", lambda *a, **kw: config)
    monkeypatch.setattr(runner, "current_environment", lambda: {"version": 1})
    original_sha = runner.sha
    monkeypatch.setattr(
        runner,
        "sha",
        lambda p: "historical-config" if str(p).endswith("benchmark.json") else original_sha(p),
    )
    calls = []

    class Client:
        def finish(self):
            calls.append("finished")

    monkeypatch.setattr(bufogen.integrations.telemetry, "init_weave", lambda: Client())

    async def generate(selected, arms, directory, **policy):
        protocol = runner.read(directory / "protocol.json")
        assert protocol["not_candidate_selection"] and protocol["not_heldout_evidence"]
        assert not protocol["historical_results_pooled"]
        assert selected == prompts and len(arms) == 3 and policy == runner.POLICY
        (directory / "manifest.jsonl").write_text(
            "".join(json.dumps({"id": str(i)}) + "\n" for i in range(48))
        )
        calls.append("generate")

    async def score(directory, **options):
        assert options == {"concurrency": 8, "publish": True}
        dump(
            directory / "weave-publication.json",
            {
                "errors": {},
                "feedback": {str(i): {"call_id": f"call-{i}"} for i in range(48)},
                "evaluations": {name: {"scored_rows": 16} for name in runner.DEFAULT_NAMES},
            },
        )
        calls.append("score")

    monkeypatch.setattr(runner, "run_benchmark", generate)
    monkeypatch.setattr(runner, "score_benchmark", score)
    monkeypatch.setattr(
        runner,
        "_benchmark",
        lambda d: {
            "status": "scored",
            "generated": 48,
            "rows": [{"id": str(i)} for i in range(48)],
        },
    )
    await runner.execute("train-fit", tmp_path / "new", tmp_path / "old", "unused")
    assert calls == ["generate", "score", "finished"]
    assert (
        len(
            list(
                (tmp_path / "new/training-fit-candidates").glob(
                    "weave-publication-reconciliation-*.json"
                )
            )
        )
        == 1
    )


async def test_protocol_drift_blocks_paid_calls(tmp_path, monkeypatch):
    historical, config = generation_config()
    source = tmp_path / "source"
    source.write_text("source")
    monkeypatch.setattr(
        runner,
        "local_inputs",
        lambda *a: ([{"id": "fixed"}], [], {"configuration": historical}, source),
    )
    monkeypatch.setattr(
        runner, "declared_plan", lambda *a: {"candidates": {n: {} for n in runner.DEFAULT_NAMES}}
    )
    counter = iter(range(3))
    monkeypatch.setattr(
        runner,
        "verify_declared_candidate",
        lambda s: {"artifact": f"wandb-artifact:///team/project/a{next(counter)}:v1"},
    )
    monkeypatch.setattr(runner, "_configuration", lambda *a, **kw: config)
    monkeypatch.setattr(runner, "current_environment", lambda: {"version": 2})
    monkeypatch.setattr(runner, "sha", lambda p: "same")
    dump(tmp_path / "new/validation-candidates/protocol.json", {"different": "old-protocol"})

    def forbidden(*args, **kwargs):
        pytest.fail("Changed protocol reached paid calls")

    monkeypatch.setattr(runner, "run_benchmark", forbidden)
    with pytest.raises(ValueError, match="differs"):
        await runner.execute("validation", tmp_path / "new", tmp_path / "old", "unused")


def publication_example():
    rows = [
        {"id": "first", "weave_call_id": "call-first"},
        {"id": "fallback", "weave_call_id": None},
    ]
    publication = {
        "feedback": {"first": {"call_id": "call-first"}, "fallback": {"call_id": "call-fallback"}},
        "evaluations": {"arm": {"scored_rows": 2}},
        "errors": {"first": "TransientError", "fallback": "TimeoutError"},
    }
    return publication, rows


def test_recovered_row_errors_are_reconciled_without_deleting_history():
    publication, rows = publication_example()
    before = copy.deepcopy(publication)
    result = runner.reconcile_publication(publication, rows, {"arm": {}}, 2)
    assert result["status"] == "complete"
    assert result["resolved_historical_row_errors"] == before["errors"]
    assert result["verified_generation_ids"] == ["fallback", "first"]
    assert publication == before


@pytest.mark.parametrize(
    "mutation",
    [
        "same_count_wrong_id",
        "wrong_call",
        "empty_call",
        "duplicate_call",
        "unresolved_eval",
        "unknown_error",
    ],
)
def test_invalid_publication_receipts_never_count_as_recovered(mutation):
    publication, rows = publication_example()
    if mutation == "same_count_wrong_id":
        publication["feedback"]["foreign"] = publication["feedback"].pop("fallback")
    elif mutation == "wrong_call":
        publication["feedback"]["first"]["call_id"] = "another-call"
    elif mutation == "empty_call":
        publication["feedback"]["fallback"]["call_id"] = ""
    elif mutation == "duplicate_call":
        publication["feedback"]["fallback"]["call_id"] = "call-first"
    elif mutation == "unresolved_eval":
        publication["errors"]["evaluation:arm"] = "EvaluationError"
    else:
        publication["errors"]["unknown-row"] = "UnknownError"
    with pytest.raises(ValueError):
        runner.reconcile_publication(publication, rows, {"arm": {}}, 2)
