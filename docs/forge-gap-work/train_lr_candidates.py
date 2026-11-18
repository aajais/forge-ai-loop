"""Prepare an optional LR experiment; never submit unless --submit is explicit.

Default: freeze the exact prior reviewed source and experiment plan locally.
--submit: durable import and independent base-model SFT for the selected arms.
--poll: read existing remote jobs once; no import, creation, routing or retries.

The LR experiment is a prepared fallback while a higher-rank route is clarified.
This runner accepts no guessed rank controls. Original candidates stay untouched.
Each operation archives local before/after state; ambiguous remote writes remain
blocked by forge_loop until explicitly reconciled. A new experiment root is
required if source, recipe, identity or frozen implementation changes.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from evaluate_candidates import exclusive_json, read, sha
from train_candidates import atomic

from bufogen.evaluation.benchmark import digest
from bufogen.integrations.forge import STUDENT_MODEL
from bufogen.integrations.forge import ForgeClient
from bufogen.integrations.forge import reviewed_entries
from bufogen.workflows.forge_loop import _locked
from bufogen.workflows.forge_loop import import_reviewed_dataset
from bufogen.workflows.forge_loop import initialize_workflow
from bufogen.workflows.forge_loop import resume_training
from bufogen.workflows.forge_loop import submit_training
from bufogen.settings import ROOT
from bufogen.settings import required

ORIGINAL = ROOT / "runs/forge-gap"
DESTINATION = ROOT / "runs/forge-gap-lr"
SOURCE_SHA256 = "e8f2f4b23c30ae73bd8fd394ac340e0b6d299f3c277307b70ce1d2d3dc746984"
UPLOAD_SHA256 = "87bff51d3698176f09ed16c494e08bdfc7412687380af83351dcf72d5e693273"
RECIPES = {"lr2e-5": 2e-5, "lr1e-4": 1e-4}


def configuration(rate):
    return {"epochs": 1, "batch_size": 2, "assistant_turns": "all", "learning_rate": rate}


def prepared_source(original):
    """Reusable source/identity checks independent of the experimental knob."""
    original = Path(original)
    source = original / "training/reviewed-examples.jsonl"
    receipt_path = original / "training/source-receipt.json"
    receipt = read(receipt_path)
    control_path = original / "training/e1/forge-state.json"
    control = read(control_path)
    if sha(source) != SOURCE_SHA256 or receipt["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Require the exact final 147-example reviewed source")
    targets = [json.loads(line) for line in source.read_text().splitlines()]
    entries = reviewed_entries(
        targets,
        heldout_ids=receipt["heldout_ids"],
        heldout_families=receipt["heldout_families"],
    )
    import hashlib

    upload_hash = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        len(targets) != 147
        or Counter(r["partition"] for r in targets) != {"train": 124, "validation": 23}
        or Counter(e["split"] for e in entries) != {"train": 124, "val": 23}
        or upload_hash != UPLOAD_SHA256
        or receipt["upload_sha256"] != upload_hash
        or receipt["source_generation_ids"] != [r["source_generation_id"] for r in targets]
    ):
        raise ValueError("Reviewed source counts, identities or mapped upload changed")
    if (
        control.get("status") != "deployed"
        or control.get("pending_operation")
        or control.get("source_sha256") != SOURCE_SHA256
        or control.get("upload_sha256") != UPLOAD_SHA256
        or control.get("base_model") != STUDENT_MODEL
        or control.get("training_config")
        != {"epochs": 1, "batch_size": 2, "assistant_turns": "all"}
        or control.get("model_artifact_ref")
        != "wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-gap-e1-20261015:v1"
    ):
        raise ValueError("Prior one-epoch immutable control identity differs")
    return source, receipt_path, receipt, control_path, control


def prepare(root=DESTINATION, *, original=ORIGINAL, write=True):
    root, original = Path(root).resolve(), Path(original).resolve()
    if root == original or root.is_relative_to(original) or original.is_relative_to(root):
        raise ValueError("Use a separate experiment root, never the original experiment")
    source, receipt_path, receipt, control_path, control = prepared_source(original)
    history_path = original / "training/e1/wandb-metrics.json"
    learning_rates = [
        row["sft/loss/learning_rate"]
        for row in read(history_path)["rows"]
        if "sft/loss/learning_rate" in row
    ]
    if not learning_rates or max(learning_rates) != 2e-4:
        raise ValueError("Prior control history does not establish the declared peak LR")
    training = root / "training"
    project = {
        k: control[k]
        for k in (
            "entity",
            "project",
            "project_alias",
            "project_id",
            "base_url",
            "training_base_url",
        )
    }
    candidates = {}
    for name, rate in RECIPES.items():
        tag = name.replace("-", "")
        candidates[name] = {
            "state_path": str(training / name / "forge-state.json"),
            "arm_id": "open_tuned_" + tag,
            "model_id": "ft-" + STUDENT_MODEL,
            "base_model": STUDENT_MODEL,
            "model_name": f"bufo-gemma-gap-{tag}-20261015",
            "dataset_name": f"bufo-gpt-6.1-sol-gap-{tag}-20261015",
            "training_config": configuration(rate),
            "initialization": "independent_base_model",
        }
    plan = {
        "schema_version": 1,
        "purpose": "optional_managed_lr_development_experiment",
        "confirmation_not_reused_as_fresh_evidence": True,
        "source_path": str(training / "reviewed-examples.jsonl"),
        "source_sha256": SOURCE_SHA256,
        "upload_sha256": UPLOAD_SHA256,
        "source_receipt_sha256": sha(receipt_path),
        "entry_counts": {"train": 124, "val": 23},
        "project": project,
        "heldout_ids": receipt["heldout_ids"],
        "heldout_families": receipt["heldout_families"],
        "prior_control": {
            "state_path": str(control_path),
            "state_sha256": sha(control_path),
            "artifact": control["model_artifact_ref"],
            "finetune_id": control["finetune_id"],
            "observed_peak_learning_rate": 2e-4,
            "history_sha256": sha(history_path),
            "contemporaneous_replicate": False,
        },
        "candidates": candidates,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/bufogen/integrations/forge.py",
                ROOT / "src/bufogen/workflows/forge_loop.py",
            )
        },
    }
    plan["plan_sha256"] = digest(plan)
    path = root / "plan.json"
    if not write:
        if read(path) != plan or sha(training / "reviewed-examples.jsonl") != SOURCE_SHA256:
            raise ValueError("Frozen LR plan/source changed")
        if sha(training / "source-receipt.json") != plan["source_receipt_sha256"]:
            raise ValueError("Frozen source receipt changed")
        return plan
    with _locked(root / "prepare"):
        if path.exists() and read(path) != plan:
            raise ValueError("Frozen LR plan changed; use a new experiment root")
        atomic(training / "reviewed-examples.jsonl", source.read_bytes(), immutable=True)
        atomic(training / "source-receipt.json", receipt_path.read_bytes(), immutable=True)
        if not path.exists():
            exclusive_json(path, plan)
    return plan


def check_state(state, candidate, plan):
    for field, expected in {
        "source_path": plan["source_path"],
        "source_sha256": SOURCE_SHA256,
        "upload_sha256": UPLOAD_SHA256,
        "base_model": STUDENT_MODEL,
        "model_name": candidate["model_name"],
        "dataset_name": candidate["dataset_name"],
        "entry_counts": plan["entry_counts"],
        "lineage": lineage(plan, candidate),
        **plan["project"],
    }.items():
        if state.get(field) != expected:
            raise ValueError(f"Candidate state differs at {field}")
    if (
        state.get("training_config") is not None
        and state["training_config"] != candidate["training_config"]
    ):
        raise ValueError("Candidate learning rate or training recipe changed")
    if state.get("finetune_id") and state.get("training_config") != candidate["training_config"]:
        raise ValueError("Existing job lacks the exact frozen training recipe")


def lineage(plan, candidate):
    return {
        "experiment_plan": str(Path(plan["source_path"]).parents[1] / "plan.json"),
        "experiment_plan_sha256": plan["plan_sha256"],
        "model_id": candidate["model_id"],
        "initialization": "independent_base_model",
    }


def archive(candidate, event, value):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(candidate["state_path"]).parent / "history" / f"{stamp}-{event}-{uuid4().hex}.json"
    exclusive_json(path, value)


def run(mode, *, root=DESTINATION, original=ORIGINAL, names=None, client_factory=ForgeClient):
    if mode not in {"prepare", "submit", "poll"}:
        raise ValueError("Unknown LR workflow operation")
    names = list(RECIPES) if names is None else names
    if not names or len(set(names)) != len(names) or set(names) - set(RECIPES):
        raise ValueError("Choose unique declared LR candidates")
    plan = prepare(root, original=original, write=mode != "poll")
    if mode == "prepare":
        return plan
    # Check every requested saved state before any remote operation.
    for name in names:
        candidate = plan["candidates"][name]
        path = Path(candidate["state_path"])
        if path.exists():
            check_state(read(path), candidate, plan)
        elif mode == "poll":
            raise ValueError("Cannot poll a candidate without a saved workflow")
    project = plan["project"]
    with client_factory(
        api_key=required("WANDB_API_KEY"),
        entity=project["entity"],
        project_alias=project["project_alias"],
        base_url=project["base_url"],
        training_base_url=project["training_base_url"],
        timeout=120,
    ) as client:
        results = {}
        for name in names:
            candidate = plan["candidates"][name]
            path = Path(candidate["state_path"])
            if path.exists():
                archive(candidate, "before-" + mode, read(path))
            try:
                if mode == "submit":
                    state = initialize_workflow(
                        path,
                        client,
                        source_path=plan["source_path"],
                        dataset_name=candidate["dataset_name"],
                        model_name=candidate["model_name"],
                        project_id=project["project_id"],
                        project=project["project"],
                        heldout_ids=plan["heldout_ids"],
                        heldout_families=plan["heldout_families"],
                        lineage=lineage(plan, candidate),
                    )
                    check_state(state, candidate, plan)
                    # Existing ambiguous intents block inside the durable workflow.
                    state = import_reviewed_dataset(path, client)
                    check_state(state, candidate, plan)
                    state = submit_training(
                        path,
                        client,
                        epochs=1,
                        batch_size=2,
                        learning_rate=candidate["training_config"]["learning_rate"],
                    )
                else:
                    state = resume_training(path, client)
                check_state(state, candidate, plan)
                results[name] = state
            finally:
                if path.exists():
                    archive(candidate, "after-" + mode, read(path))
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--submit", action="store_true")
    modes.add_argument("--poll", action="store_true")
    parser.add_argument("--root", type=Path, default=DESTINATION)
    parser.add_argument("--candidate", action="append", choices=list(RECIPES))
    args = parser.parse_args()
    mode = "submit" if args.submit else "poll" if args.poll else "prepare"
    results = run(mode, root=args.root, names=args.candidate)
    if mode == "prepare":
        print(
            json.dumps(
                {
                    "plan": str(args.root / "plan.json"),
                    "source_sha256": results["source_sha256"],
                    "candidates": results["candidates"],
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {
                    name: {
                        "model_id": "ft-" + STUDENT_MODEL,
                        **{
                            key: state.get(key)
                            for key in (
                                "status",
                                "finetune_id",
                                "model_artifact_ref",
                                "training_config",
                            )
                        },
                    }
                    for name, state in results.items()
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
