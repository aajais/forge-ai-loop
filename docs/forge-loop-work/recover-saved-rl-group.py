"""Recover exactly one audited, unsubmitted four-rollout group. Default is local validation.

This is a one-off reviewed recovery, not a general resume scheduler. No generation,
judge call, inferred response or optimizer retry occurs. --execute is required for
remote registration/logging and one training update. Never reuse an output directory.
"""

import argparse
import asyncio
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean

import art
from art.serverless import ServerlessBackend
from art.utils.trajectory_logging import write_trajectory_groups_parquet

from bufogen.integrations.forge import ForgeClient
from bufogen.generation import Manifest
from bufogen.models import load_backends
from bufogen.integrations.registry import RegistryClient
from bufogen.training.rl_contract import require_calibration
from bufogen.evaluation.scorers.scene_quality import SceneQuality
from bufogen.settings import ROOT
from bufogen.settings import project_path
from bufogen.settings import required
from bufogen.training.common import reward
from bufogen.training.rl import _atomic_record
from bufogen.training.rl import require_rl_logging_dependencies
from bufogen.training.rl import resolve_sft_identity
from bufogen.training.rl import validate_student_capture
from bufogen.training.rl import verify_registered_continuation

BASE = "google/gemma-4-26B-A4B-it"
SOURCE = ROOT / "runs/bufo-gemma-sft-rl-pilot-20261015-retry1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_saved_group(source):
    require_rl_logging_dependencies()
    source = Path(source).resolve()
    state_path = source / "rl-training.json"
    state = json.loads(state_path.read_text())
    if (
        state.get("status") != "failed"
        or state.get("error_type") != "ModuleNotFoundError"
        or state.get("arm") != "sft_rl"
        or state.get("base_model") != BASE
        or state.get("start_step") != 1
        or state.get("steps") != 2
        or any(
            key in state for key in ("training_from_step", "completed_step", "checkpoint_artifact")
        )
    ):
        raise ValueError("Source is not the specific pre-training, two-step SFT+RL failure")
    identity = state["sft_identity"]
    entity, project = project_path().split("/")
    if identity["entity"] != entity or identity["project"] != project:
        raise ValueError("Recovery entity/project differs from saved provenance")
    expected_model = f"wandb-artifact:///{entity}/{project}/{identity['storage_name']}:step1"
    scorer_config = SceneQuality().configuration()
    require_calibration(state["calibration"], scorer_configuration=scorer_config)
    config = load_backends()["open_base"]
    manifest_path = source / "manifest.jsonl"
    rows = Manifest(manifest_path).read()
    if len(rows) != 4 or len({r.id for r in rows}) != 4:
        raise ValueError("Recovery requires exactly four unique scored generation rows")
    by_id = {row.id: row for row in rows}
    selected = {}
    all_capture_paths = sorted((source / "captures").glob("*.json"))
    hashes = {str(p): sha(p) for p in [state_path, manifest_path, *all_capture_paths]}
    for path in all_capture_paths:
        capture = json.loads(path.read_text())
        if capture.get("status") == "failed":
            continue  # Actual partial failed captures remain audit evidence, never trainable.
        if capture.get("status") != "scored" or capture.get("step") != 1:
            raise ValueError("Unexpected unscored or different-step capture in source attempt")
        index = capture.get("rollout_index")
        if type(index) is not int or index not in range(4) or index in selected:
            raise ValueError("Expected one scored capture per rollout index 0..3")
        row = by_id.get(capture.get("generation_id"))
        if (
            row is None
            or row.model != expected_model
            or row.group_id != capture.get("group_id")
            or row.prompt_id != capture.get("prompt_id")
            or row.benchmark_metadata.get("split") != "train"
        ):
            raise ValueError("Capture and generation source identity differ")
        score = row.scores.get("scene_quality", {})
        if (
            score.get("status") != "ok"
            or score.get("generation_id") != row.id
            or score.get("scorer_configuration") != scorer_config
            or score.get("requirements") != row.benchmark_metadata["requirements"]
        ):
            raise ValueError("Stored scene judgment is not bound to this row and frozen scorer")
        computed_reward = reward(row, scorer_configuration=scorer_config)
        if (
            type(capture.get("reward")) not in (float, int)
            or not math.isfinite(capture["reward"])
            or capture["reward"] != computed_reward
            or capture.get("metrics", {}).get("scene_reward") != computed_reward
        ):
            raise ValueError("Saved reward does not match bound generation/scorer evidence")
        trajectory = art.Trajectory(
            exchanges=capture["exchanges"], reward=capture["reward"], metrics=capture["metrics"]
        )
        validate_student_capture(
            trajectory, model=expected_model, provider=config.provider, base_url=config.base_url
        )
        if trajectory.exchanges.model_dump(mode="json") != capture["exchanges"]:
            raise ValueError("Reconstruction changed captured exchanges")
        if not row.png_path or not Path(row.png_path).is_file():
            raise ValueError("Scored PNG evidence is missing")
        hashes[str(Path(row.png_path).resolve())] = sha(row.png_path)
        if row.svg_path:
            hashes[str(Path(row.svg_path).resolve())] = sha(row.svg_path)
        selected[index] = (trajectory, capture, path)
    if set(selected) != set(range(4)) or {c[1]["generation_id"] for c in selected.values()} != set(
        by_id
    ):
        raise ValueError("Recovered group does not exactly cover the four manifest rows")
    if len({row.prompt_id for row in rows}) != 1 or len({row.prompt for row in rows}) != 1:
        raise ValueError("One training group must share one prompt")
    evidence = {
        "source_attempt": str(source),
        "source_hashes": hashes,
        "scorer_configuration": scorer_config,
        "expected_inference_model": expected_model,
        "source_requested_steps": 2,
        "recovery_requested_updates": 1,
        "generation_ids": [selected[i][1]["generation_id"] for i in range(4)],
        "rewards": [selected[i][0].reward for i in range(4)],
        "capture_paths": [str(selected[i][2]) for i in range(4)],
        "captured_choice_counts": [
            len(selected[i][0].exchanges.chat_completions) for i in range(4)
        ],
        "failed_attempts_excluded": len(all_capture_paths) - 4,
        "script_sha256": sha(__file__),
    }
    groups = [art.TrajectoryGroup([selected[i][0] for i in range(4)])]
    return state, groups, evidence


def verify_source_hashes(evidence):
    for path, expected in evidence["source_hashes"].items():
        if sha(path) != expected:
            raise ValueError(f"Source evidence changed: {Path(path).name}")


async def execute(source, output):
    state, groups, evidence = validate_saved_group(source)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive creation; refuses replay even after a failed submission.
    record = {
        "arm": "sft_rl",
        "base_model": BASE,
        "initial_checkpoint": state["initial_checkpoint"],
        "dataset_ref": state["dataset_ref"],
        "sft_identity": state["sft_identity"],
        **evidence,
        "status": "validated",
        "actual_completed_updates": 0,
    }
    path = output / "rl-training.json"

    def checkpoint(**changes):
        record.update(changes)
        _atomic_record(path, record)

    checkpoint()
    # Prove actual ART serialization before constructing a remote client.
    parquet_path = output / "recovered-group.parquet"
    write_trajectory_groups_parquet(groups, parquet_path)
    checkpoint(parquet_sha256=sha(parquet_path))
    import wandb

    api = wandb.Api()
    entity, project = project_path().split("/")
    backend = None
    try:
        with ForgeClient(
            api_key=required("WANDB_API_KEY"), entity=entity, project_alias="bufo-gen"
        ) as forge:
            job = forge.get_finetune(state["sft_identity"]["forge_finetune_id"])
        identity = resolve_sft_identity(
            initial_checkpoint=state["initial_checkpoint"],
            forge_job=job,
            api=api,
            entity=entity,
            project=project,
            base_model=BASE,
        )
        if identity != state["sft_identity"]:
            raise ValueError("Live SFT identity differs from captured source")
        backend = ServerlessBackend(api_key=required("WANDB_API_KEY"))
        model = art.TrainableModel(
            name=output.name,
            run_name=identity["storage_name"],
            project=project,
            entity=entity,
            base_model=BASE,
        )
        await model.register(backend)
        step = await verify_registered_continuation(model, identity, api)
        if (
            step != 1
            or model.id != state["model_id"]
            or model.get_inference_name(step=step) != evidence["expected_inference_model"]
        ):
            raise ValueError(
                "Remote training identity/checkpoint differs from the saved rollout model"
            )
        verify_source_hashes(evidence)
        checkpoint(status="registered", model_id=model.id, run_id=model.run_id, start_step=step)
        await model.log(groups, split="train")
        # Recheck after logging; no advancement may occur between capture and submission.
        if await verify_registered_continuation(model, identity, api) != step:
            raise ValueError("Checkpoint advanced before recovered group submission")
        verify_source_hashes(evidence)
        checkpoint(status="training", training_from_step=step, optimizer_submission_intent=True)
        result = await backend.train(model, groups, learning_rate=5e-6, precalculate_logprobs=True)
        # Persist the response immediately, before logging metrics or resolving/linking artifacts.
        checkpoint(
            status="optimizer_returned",
            actual_completed_updates=1,
            completed_step=result.step,
            returned_artifact=result.artifact_name,
            returned_metrics=result.metrics,
        )
        if result.step != step + 1 or not result.artifact_name:
            raise ValueError("Optimizer returned an unexpected step or missing artifact")
        artifact = api.artifact(
            result.artifact_name.removeprefix("wandb-artifact:///"), type="lora"
        )
        logged_by = artifact.logged_by()
        if (
            artifact.entity != entity
            or artifact.project != project
            or not artifact.digest
            or not re.fullmatch(re.escape(identity["storage_name"]) + r":v[0-9]+", artifact.name)
            or not logged_by
            or logged_by.id != model.run_id
        ):
            raise ValueError("Returned checkpoint lacks immutable matching run provenance")
        immutable = f"wandb-artifact:///{entity}/{project}/{artifact.name}"
        checkpoint(
            status="checkpoint_resolved",
            checkpoint_artifact=immutable,
            checkpoint_artifact_id=artifact.id,
            checkpoint_artifact_digest=artifact.digest,
        )
        await model.log(
            metrics={**result.metrics, "reward": mean(evidence["rewards"])},
            step=result.step,
            split="train",
        )
        linked = RegistryClient().link_checkpoint(
            immutable,
            name=output.name,
            sources=[state["dataset_ref"], identity["initial_checkpoint"]],
        )
        checkpoint(status="completed", registry_reference=linked)
        return record
    except Exception as error:
        checkpoint(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if backend is not None:
            await backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/bufo-gemma-sft-rl-recovered-20261015"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Register/log and submit exactly one paid optimizer update",
    )
    args = parser.parse_args()
    if args.execute:
        result = asyncio.run(execute(args.source, args.output))
    else:
        _, groups, result = validate_saved_group(args.source)
        path = Path("/private/tmp/bufo-rl-recovery-group.parquet")
        write_trajectory_groups_parquet(groups, path)
        result |= {
            "mode": "local_validation_only",
            "parquet_path": str(path),
            "parquet_sha256": sha(path),
        }
        _atomic_record(Path("/private/tmp/bufo-rl-recovery-validation.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
