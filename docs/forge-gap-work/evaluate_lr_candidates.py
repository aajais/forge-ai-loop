"""Controlled follow-up diagnostics; default invocation is local preflight only.

--execute runs the fixed training-fit or validation set against three verified
immutable FT checkpoints. No confirmation stage, selection, serving-route update,
or training submission exists here. Training-fit results cannot select a model.
Original validation is a historical reference, never pooled with these reruns.
An optional frozen candidate-plan JSON may declare supported future recipes;
its exact configurations and identities are verified, not assumed to exist.
"""

import argparse
import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path

from evaluate_candidates import (
    BASE,
    POLICY,
    TEACHER,
    common_configuration,
    curriculum,
    environment,
    exclusive_json,
    read,
    sha,
    verify_candidate,
)
from evaluate_training_fit import (
    CURRICULUM_SHA256,
    TRAINING_SHA256,
    fixed_prompts,
    freeze_protocol,
)

from bufogen.evaluation.benchmark import _configuration
from bufogen.evaluation.benchmark import digest
from bufogen.evaluation.benchmark import run_benchmark
from bufogen.evaluation.benchmark import score_benchmark
from bufogen.workflows.loop_report import _benchmark
from bufogen.models import load_backends
from bufogen.settings import ROOT

UPLOAD_SHA256 = "87bff51d3698176f09ed16c494e08bdfc7412687380af83351dcf72d5e693273"
DEFAULT_NAMES = ("open_tuned_control", "open_tuned_lr2e5", "open_tuned_lr1e4")
IMMUTABLE = r"wandb-artifact:///[^/]+/[^/]+/[^/]+:v[0-9]+"


def check_output(stage, root, historical):
    if stage not in {"train-fit", "validation"}:
        raise ValueError("Unsupported stage; no confirmation reuse or selection is allowed")
    root, historical = Path(root).resolve(), Path(historical).resolve()
    if root == historical or root.is_relative_to(historical) or historical.is_relative_to(root):
        raise ValueError("New output must be separate from historical experiment evidence")


def declared_plan(root, historical, path=None):
    selection = read(Path(historical) / "selection.json")
    if (
        digest({k: v for k, v in selection.items() if k != "receipt_sha256"})
        != selection["receipt_sha256"]
    ):
        raise ValueError("Historical selection receipt changed")
    chosen = selection["candidates"][selection["selected_arm"]]
    if chosen["epochs"] != 1 or chosen["artifact"] != selection["selected_artifact"]:
        raise ValueError("The reference must be the immutable selected one-epoch checkpoint")
    control = {
        "state_path": chosen["state_path"],
        "training_config": {"epochs": 1, "batch_size": 2, "assistant_turns": "all"},
        "required_state_sha256": chosen["state_sha256"],
        "required_artifact": chosen["artifact"],
        "metadata": {"role": "existing_control", "observed_peak_lr": 0.0002},
    }
    default = {DEFAULT_NAMES[0]: control}
    for name, suffix, lr in (
        (DEFAULT_NAMES[1], "lr2e-5", 2e-5),
        (DEFAULT_NAMES[2], "lr1e-4", 1e-4),
    ):
        default[name] = {
            "state_path": str((Path(root) / "training" / suffix / "forge-state.json").resolve()),
            "training_config": {
                "epochs": 1,
                "batch_size": 2,
                "assistant_turns": "all",
                "learning_rate": lr,
            },
            "metadata": {"role": "planned_followup", "requested_peak_lr": lr},
        }
    plan_path = Path(path) if path else Path(root) / "plan.json"
    plan = read(plan_path) if plan_path.exists() else {"schema_version": 1, "candidates": default}
    if path and not plan_path.exists():
        raise FileNotFoundError(plan_path)
    # Accept the training worker's immutable two-candidate plan directly.
    if plan.get("purpose") == "optional_managed_lr_development_experiment":
        if (
            digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]
            or plan["source_sha256"] != TRAINING_SHA256
            or plan["upload_sha256"] != UPLOAD_SHA256
            or plan["prior_control"]["state_sha256"] != control["required_state_sha256"]
            or plan["prior_control"]["artifact"] != control["required_artifact"]
            or sha(plan["source_path"]) != TRAINING_SHA256
        ):
            raise ValueError("Frozen training plan/source/control differs")
        converted = {DEFAULT_NAMES[0]: control}
        for entry in plan["candidates"].values():
            if (
                entry.get("initialization") != "independent_base_model"
                or entry.get("base_model") != BASE
            ):
                raise ValueError("Training plan does not declare independent exact-base training")
            converted[entry["arm_id"]] = {
                "state_path": entry["state_path"],
                "training_config": entry["training_config"],
                "required_lineage": {
                    "experiment_plan_sha256": plan["plan_sha256"],
                    "initialization": "independent_base_model",
                },
                "metadata": {"requested_peak_lr": entry["training_config"].get("learning_rate")},
            }
        plan = {
            "schema_version": 1,
            "candidates": converted,
            "training_plan_path": str(plan_path.resolve()),
            "training_plan_sha256": sha(plan_path),
        }
    candidates = plan.get("candidates", {})
    if (
        plan.get("schema_version") != 1
        or len(candidates) != 3
        or candidates.get(DEFAULT_NAMES[0]) != control
    ):
        raise ValueError("Plan must retain the exact original control and two declared candidates")
    if any(not re.fullmatch(r"open_tuned_[a-z0-9_]+", name) for name in candidates):
        raise ValueError("Candidate arm names must be stable internal identifiers")
    for spec in candidates.values():
        cfg = spec.get("training_config", {})
        if any(
            cfg.get(k) != v
            for k, v in {"epochs": 1, "batch_size": 2, "assistant_turns": "all"}.items()
        ):
            raise ValueError("Follow-up must keep one epoch, batch two and all assistant turns")
    return plan


def local_inputs(stage, historical, curriculum_path):
    all_prompts, curriculum_hash = curriculum(curriculum_path)
    if curriculum_hash != CURRICULUM_SHA256:
        raise ValueError("Curriculum differs from the frozen experiment")
    historical = Path(historical)
    source = historical / "training/reviewed-examples.jsonl"
    if sha(source) != TRAINING_SHA256:
        raise ValueError("Reviewed teacher source changed")
    targets = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    training, provenance = fixed_prompts(all_prompts, targets)
    selection = read(historical / "selection.json")
    benchmark_path = historical / "validation-candidates/benchmark.json"
    if sha(benchmark_path) != selection["validation_hashes"]["benchmark.json"]:
        raise ValueError("Historical validation configuration changed")
    prior = read(benchmark_path)
    validation = [p for p in all_prompts if p["split"] == "validation"]
    if len(validation) != 24 or validation != prior["configuration"]["prompts"]:
        raise ValueError("Expected the exact 24 frozen validation prompts")
    if stage == "train-fit":
        prompts = training
    elif stage == "validation":
        prompts = validation
    else:
        raise ValueError("Unsupported stage")
    return prompts, provenance if stage == "train-fit" else [], prior, source


def verify_declared_candidate(spec):
    path = Path(spec["state_path"])
    state_hash = sha(path)
    state = read(path)
    expected = spec["training_config"]
    if (
        state.get("status") != "deployed"
        or state.get("pending_operation")
        or state.get("base_model") != BASE
        or state.get("training_config") != expected
        or state.get("source_sha256") != TRAINING_SHA256
        or state.get("upload_sha256") != UPLOAD_SHA256
        or state.get("entry_counts") != {"train": 124, "val": 23}
        or not re.fullmatch(IMMUTABLE, state.get("model_artifact_ref") or "")
    ):
        raise ValueError(
            "Candidate deployment, training config or source identity differs from plan"
        )
    if sha(state["source_path"]) != TRAINING_SHA256:
        raise ValueError("Candidate training source bytes changed")
    if spec.get("required_state_sha256") and sha(path) != spec["required_state_sha256"]:
        raise ValueError("Frozen control state changed")
    if spec.get("required_artifact") and state["model_artifact_ref"] != spec["required_artifact"]:
        raise ValueError("Frozen control artifact changed")
    if any(
        state.get("lineage", {}).get(key) != value
        for key, value in spec.get("required_lineage", {}).items()
    ):
        raise ValueError("Candidate lineage differs from the frozen training plan")
    # Checks deployed live Forge job + W&B immutable artifact/base identity.
    result = verify_candidate(path, 1)
    if (
        result["artifact"] != state["model_artifact_ref"]
        or result["training_config"] != expected
        or result["state_sha256"] != state_hash
    ):
        raise ValueError("Live candidate differs from declared checkpoint or recipe")
    return {**result, "metadata": spec.get("metadata", {})}


def arms_for(candidates):
    if len(candidates) != 3 or len({c["artifact"] for c in candidates.values()}) != 3:
        raise ValueError("Exactly three distinct immutable checkpoints are required")
    template = load_backends()["open_tuned"]
    arms = []
    for name, candidate in candidates.items():
        if not re.fullmatch(IMMUTABLE, candidate["artifact"]):
            raise ValueError("Mutable candidate artifact")
        arms.append(
            replace(
                template,
                name=name,
                model=candidate["artifact"],
                label="ft-" + BASE,
                base_model=BASE,
                tuning_method="sft",
            )
        )
    return arms


def current_environment():
    return {
        **environment(),
        "followup_runner_sha256": sha(__file__),
        "training_fit_helper_sha256": sha(Path(__file__).with_name("evaluate_training_fit.py")),
    }


def validate_generation_configuration(configuration, historical_configuration):
    if common_configuration(configuration) != common_configuration(historical_configuration):
        raise ValueError("Generation/scoring source, runtime or .7/6000/8 policy changed")
    previous = next(
        arm for arm in historical_configuration["arms"] if arm["name"] == "open_tuned_epoch1"
    )

    def serving_fields(arm):
        return {key: value for key, value in arm.items() if key not in {"name", "model"}}

    if any(serving_fields(arm) != serving_fields(previous) for arm in configuration["arms"]):
        raise ValueError("Candidate serving, tool support or sampling settings changed")
    if configuration["scorer"].get("judge_model") != TEACHER:
        raise ValueError("Scene judge differs from frozen comparison")


def freeze_run(directory, protocol, peer):
    if (Path(peer) / "protocol.json").exists():
        previous = read(Path(peer) / "protocol.json")
        for key in ("candidates", "candidate_plan", "environment", "training_source_sha256"):
            if previous[key] != protocol[key]:
                raise ValueError("Follow-up stages differ in candidate identity or environment")
        if common_configuration(previous["configuration"]) != common_configuration(
            protocol["configuration"]
        ):
            raise ValueError("Follow-up stages differ in generation/scoring settings")
    freeze_protocol(directory, protocol)


def reconcile_publication(publication, rows, candidates, prompt_count):
    """Validate exact native receipts; preserve historical errors after recovery."""
    expected = {row["id"]: row for row in rows}
    feedback = publication.get("feedback", {})
    if len(expected) != len(rows) or set(feedback) != set(expected):
        raise ValueError("Native feedback generation IDs differ from the complete manifest")
    call_ids = []
    for generation_id, row in expected.items():
        receipt = feedback[generation_id]
        call_id = receipt.get("call_id") if isinstance(receipt, dict) else None
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("Native feedback needs a nonempty call ID for each generation")
        if row.get("weave_call_id") and call_id != row["weave_call_id"]:
            raise ValueError("Native feedback call ID differs from its generation trace")
        call_ids.append(call_id)
    if len(set(call_ids)) != len(call_ids):
        raise ValueError("Native feedback repeats one call ID for different generations")
    evaluations = publication.get("evaluations", {})
    if set(evaluations) != set(candidates) or any(
        entry.get("scored_rows") != prompt_count for entry in evaluations.values()
    ):
        raise ValueError("Native Weave evaluation coverage incomplete")
    errors = publication.get("errors", {})
    # _publish retains generation-keyed errors after a successful retry. Exact
    # successful feedback coverage above supersedes only those row-level errors.
    # Evaluation/unknown errors have no corresponding validated recovery contract.
    unresolved = {key: value for key, value in errors.items() if key not in expected}
    if unresolved:
        raise ValueError("Native Weave publication has unresolved non-row errors")
    return {
        "schema_version": 1,
        "status": "complete",
        "verified_generation_ids": sorted(expected),
        "feedback_count": len(feedback),
        "evaluation_arms": sorted(evaluations),
        "resolved_historical_row_errors": dict(errors),
        "resolution_basis": "Exact successful generation-keyed feedback receipts with unique call IDs; manifest trace IDs matched when present",
        "original_publication_errors_preserved": True,
    }


async def execute(stage, root, historical, curriculum_path, candidate_plan=None):
    root, historical = Path(root), Path(historical)
    check_output(stage, root, historical)
    prompts, provenance, previous, source = local_inputs(stage, historical, curriculum_path)
    plan = declared_plan(root, historical, candidate_plan)
    candidates = {
        name: verify_declared_candidate(spec) for name, spec in plan["candidates"].items()
    }
    arms = arms_for(candidates)
    configuration = _configuration(prompts, arms, **POLICY)
    validate_generation_configuration(configuration, previous["configuration"])
    protocol = {
        "schema_version": 1,
        "stage": stage,
        "purpose": "controlled_followup_diagnostic",
        "not_heldout_evidence": True,
        "not_candidate_selection": True,
        "original_validation_reference_only": True,
        "historical_results_pooled": False,
        "candidate_plan": plan,
        "candidates": candidates,
        "configuration": configuration,
        "environment": current_environment(),
        "curriculum_sha256": CURRICULUM_SHA256,
        "training_source_sha256": sha(source),
        "fixed_prompt_ids": [p["id"] for p in prompts],
        "training_target_provenance": provenance,
        "historical_benchmark_sha256": sha(historical / "validation-candidates/benchmark.json"),
        "native_weave_publish": True,
    }
    directory = root / (
        "training-fit-candidates" if stage == "train-fit" else "validation-candidates"
    )
    peer = root / ("validation-candidates" if stage == "train-fit" else "training-fit-candidates")
    freeze_run(directory, protocol, peer)
    from bufogen.integrations.telemetry import init_weave

    client = init_weave()
    try:
        # The benchmark's durable starts remain failures after an interrupted attempt;
        # guarded resume never silently repeats them or drops failure denominators.
        await run_benchmark(prompts, arms, directory, **POLICY)
        await score_benchmark(directory, concurrency=8, publish=True)
        result = _benchmark(
            directory
        )  # Revalidate complete coverage and exact scored/source binding.
        if result["status"] != "scored" or result["generated"] != len(prompts) * 3:
            raise ValueError("Incomplete follow-up diagnostic evidence")
        publication = read(directory / "weave-publication.json")
        reconciliation = reconcile_publication(
            publication, result["rows"], candidates, len(prompts)
        )
        publication_hash = sha(directory / "weave-publication.json")
        reconciliation["publication_sha256"] = publication_hash
        reconciliation["manifest_sha256"] = sha(directory / "manifest.jsonl")
        reconciliation_path = (
            directory / f"weave-publication-reconciliation-{publication_hash}.json"
        )
        if reconciliation_path.exists():
            if read(reconciliation_path) != reconciliation:
                raise ValueError("Saved publication reconciliation evidence changed")
        else:
            exclusive_json(reconciliation_path, reconciliation)
    finally:
        client.finish()
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train-fit", "validation"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/forge-gap-lr")
    parser.add_argument("--historical-root", type=Path, default=ROOT / "runs/forge-gap")
    parser.add_argument(
        "--curriculum", type=Path, default=ROOT / "runs/forge-gap/curriculum-manifest.json"
    )
    parser.add_argument("--candidate-plan", type=Path)
    parser.add_argument(
        "--execute", action="store_true", help="Run paid generation/scoring after review"
    )
    args = parser.parse_args()
    if args.execute:
        asyncio.run(
            execute(
                args.stage, args.root, args.historical_root, args.curriculum, args.candidate_plan
            )
        )
    else:
        check_output(args.stage, args.root, args.historical_root)
        prompts, _, _, source = local_inputs(args.stage, args.historical_root, args.curriculum)
        plan = declared_plan(args.root, args.historical_root, args.candidate_plan)
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "paid_calls_started": False,
                    "status": "planned_only_not_submitted",
                    "fixed_prompt_ids": [p["id"] for p in prompts],
                    "planned_attempts": len(prompts) * 3,
                    "training_source_sha256": sha(source),
                    "candidate_plan": plan,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
