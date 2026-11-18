"""Matched Forge candidate evaluation. Only validation/confirmation make paid calls.

Selection/report are local, immutable evidence operations. This standalone runner
does not change application defaults, adapters, training state, or model routing.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

from paired_statistics import paired_family_interval

from bufogen.evaluation.benchmark import _configuration
from bufogen.evaluation.benchmark import _summary
from bufogen.evaluation.benchmark import _validate_rows
from bufogen.evaluation.benchmark import comparison_fingerprint
from bufogen.evaluation.benchmark import digest
from bufogen.evaluation.benchmark import load_benchmark
from bufogen.evaluation.benchmark import run_benchmark
from bufogen.evaluation.benchmark import score_benchmark
from bufogen.generation import BufoGeneration
from bufogen.models import load_backends
from bufogen.settings import ROOT
from bufogen.settings import project_path
from bufogen.settings import required

BASE = "google/gemma-4-26B-A4B-it"
TEACHER = "gpt-6.1-sol"
CANDIDATES = {"open_tuned_epoch1": 1, "open_tuned_epoch3": 3}
POLICY = dict(concurrency=16, temperature=0.7, max_tokens=6000, max_turns=8)
RULE = "strict passes; scored-only constraint coverage; fewer generation errors; one epoch"
EVIDENCE = ("benchmark.json", "manifest.jsonl", "scored.jsonl", "summary.json", "protocol.json")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive_json(path, value):
    """Publish a complete fsynced JSON file without overwriting existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def environment():
    def normalize(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    return {
        "packages": {normalize(d.metadata["Name"]): d.version for d in distributions()},
        "lock_sha256": sha(ROOT / "uv.lock"),
        "runner_sha256": sha(__file__),
        "statistics_sha256": sha(Path(__file__).with_name("paired_statistics.py")),
    }


def curriculum(path):
    manifest = read(path)
    source = ROOT / manifest["source"]
    if sha(source) != manifest["sha256"]:
        raise ValueError("Curriculum hash changed")
    rows = load_benchmark(source)
    if Counter(r["split"] for r in rows) != {"train": 128, "validation": 24, "heldout": 80}:
        raise ValueError("Unexpected curriculum split coverage")
    return rows, manifest["sha256"]


def verify_candidate(path, epochs, *, api=None, forge_factory=None):
    """Read-only live Forge and W&B identity checks before any paid generation."""
    from bufogen.integrations.forge import ForgeClient
    from bufogen.training.rl import resolve_sft_identity

    state = read(path)
    if (
        state.get("status") != "deployed"
        or state.get("pending_operation")
        or state.get("base_model") != BASE
        or state.get("training_config", {}).get("epochs") != epochs
        or state.get("training_config", {}).get("assistant_turns") != "all"
    ):
        raise ValueError("Candidate needs a completed exact-base Forge deployment receipt")
    if sha(state["source_path"]) != state["source_sha256"]:
        raise ValueError("Training source changed")
    if f"{state['entity']}/{state['project']}" != project_path():
        raise ValueError("Candidate project differs from evaluation project")
    factory = forge_factory or ForgeClient
    with factory(
        api_key=required("WANDB_API_KEY"),
        entity=state["entity"],
        project_alias=state["project_alias"],
    ) as forge:
        job = forge.get_finetune(state["finetune_id"])
    if (
        job.get("id") != state["finetune_id"]
        or job.get("dataset_id") != state["dataset_id"]
        or job.get("model_config", {}).get("config") != state["training_config"]
    ):
        raise ValueError("Live Forge job differs from the candidate receipt")
    if api is None:
        import wandb

        api = wandb.Api()
    identity = resolve_sft_identity(
        initial_checkpoint=state["model_artifact_ref"],
        forge_job=job,
        api=api,
        entity=state["entity"],
        project=state["project"],
        base_model=BASE,
    )
    return {
        "state_path": str(Path(path).resolve()),
        "state_sha256": sha(path),
        "epochs": epochs,
        "source_sha256": state["source_sha256"],
        "upload_sha256": state["upload_sha256"],
        "training_config": state["training_config"],
        "identity": identity,
        "model_id": "ft-" + BASE,
        "artifact": identity["initial_checkpoint"],
    }


def arms_for(candidates):
    configs = load_backends()
    if configs["open_base"].model != BASE or configs["closed_quality_bar"].model != TEACHER:
        raise ValueError("Reference model overrides differ from the declared comparison")
    arms = [configs["open_base"], configs["closed_quality_bar"]]
    if len({c["artifact"] for c in candidates.values()}) != len(candidates):
        raise ValueError("Candidates must be distinct immutable checkpoints")
    for name, candidate in candidates.items():
        uri = candidate["artifact"]
        if not re.fullmatch(r"wandb-artifact:///[^/]+/[^/]+/[^/]+:v[0-9]+", uri):
            raise ValueError("Candidate artifact is not immutable")
        arms.append(
            replace(
                configs["open_tuned"],
                name=name,
                model=uri,
                label="ft-" + BASE,
                base_model=BASE,
                tuning_method="sft",
            )
        )
    return arms


def common_configuration(config):
    # Keep all source/runtime/policy fields, not merely comparison_fingerprint.
    return {k: v for k, v in config.items() if k not in {"prompts", "prompt_hash", "arms"}}


def load_complete(directory, prompts, expected_arms):
    directory = Path(directory)
    settings, saved = read(directory / "benchmark.json"), read(directory / "summary.json")
    config = settings["configuration"]
    if (
        config["prompts"] != prompts
        or {a["name"] for a in config["arms"]} != set(expected_arms)
        or settings["configuration_hash"] != digest(config)
        or settings["comparison_fingerprint"] != comparison_fingerprint(config)
    ):
        raise ValueError("Benchmark prompt/arm/configuration mismatch")
    manifest = [
        json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()
    ]
    scored = [json.loads(line) for line in (directory / "scored.jsonl").read_text().splitlines()]
    expected = {(p["id"], arm) for p in prompts for arm in expected_arms}
    for values in (manifest, scored):
        if (
            len(values) != len(expected)
            or {(r["prompt_id"], r["backend"]) for r in values} != expected
        ):
            raise ValueError("Incomplete or duplicated evaluation coverage")
        if len({r["id"] for r in values}) != len(values):
            raise ValueError("Duplicate generation identities")
    originals = {r["id"]: r for r in manifest}
    for row in scored:
        original = originals.get(row["id"])
        result = row.get("scores", {}).get("scene_quality", {})
        if (
            original is None
            or any(row.get(k) != v for k, v in original.items() if k != "scores")
            or result.get("generation_id") != row["id"]
            or result.get("requirements") != row["benchmark_metadata"]["requirements"]
            or result.get("scorer_configuration") != config["scorer"]
        ):
            raise ValueError("Scored result is not bound to the original generation")
    rows = [BufoGeneration(**r) for r in scored]
    _validate_rows(rows, settings)
    summary = _summary(rows, settings)
    if summary != saved:
        raise ValueError("Saved summary disagrees with actual scored rows")
    protocol = read(directory / "protocol.json")
    if protocol["configuration"] != config:
        raise ValueError("Run protocol differs from the frozen benchmark")
    arm_map = {arm["name"]: arm for arm in config["arms"]}
    if arm_map["open_base"]["model"] != BASE or arm_map["closed_quality_bar"]["model"] != TEACHER:
        raise ValueError("Reference IDs differ from the experiment")
    if set(protocol["candidates"]) != set(expected_arms) - {"open_base", "closed_quality_bar"}:
        raise ValueError("Candidate identities differ from evaluated arms")
    for name, candidate in protocol["candidates"].items():
        if (
            arm_map[name]["model"] != candidate["artifact"]
            or arm_map[name].get("base_model") != BASE
        ):
            raise ValueError("Evaluated candidate differs from verified artifact")
    return settings, rows, summary, protocol


def evidence_hashes(directory):
    return {name: sha(Path(directory) / name) for name in EVIDENCE}


def rank(summary):
    def key(name):
        values = summary["arms"][name]
        coverage = values["metric_means_scored_only"]["constraint_coverage"]
        if coverage is not None and not math.isfinite(coverage):
            raise ValueError("Nonfinite coverage")
        return (
            values["provisional_passes"],
            coverage if coverage is not None else -1,
            -values["generation_errors"],
            -CANDIDATES[name],
        )

    return max(CANDIDATES, key=key)


def select(root, curriculum_path):
    root = Path(root)
    confirmation = root / "confirmation-selected"
    if confirmation.exists() and any(confirmation.iterdir()):
        raise ValueError("Confirmation evidence already exists; selection must precede it")
    all_prompts, curriculum_hash = curriculum(curriculum_path)
    prompts = [p for p in all_prompts if p["split"] == "validation"]
    directory = root / "validation-candidates"
    settings, _, summary, protocol = load_complete(
        directory, prompts, ["open_base", "closed_quality_bar", *CANDIDATES]
    )
    if protocol["curriculum_sha256"] != curriculum_hash:
        raise ValueError("Validation curriculum differs from selection")
    selected = rank(summary)
    value = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": RULE,
        "curriculum_sha256": curriculum_hash,
        "validation_run_id": settings["run_id"],
        "validation_hashes": evidence_hashes(directory),
        "selected_arm": selected,
        "selected_artifact": protocol["candidates"][selected]["artifact"],
        "candidates": protocol["candidates"],
        "validation_summary": summary,
        "confirmation_not_started": True,
    }
    value["receipt_sha256"] = digest(value)
    exclusive_json(root / "selection.json", value)
    return value


def selection_for_confirmation(root, curriculum_path):
    root = Path(root)
    selection = read(root / "selection.json")  # Must exist before any network/generation call.
    content = {k: v for k, v in selection.items() if k != "receipt_sha256"}
    if digest(content) != selection["receipt_sha256"] or selection["rule"] != RULE:
        raise ValueError("Selection receipt changed")
    all_prompts, curriculum_hash = curriculum(curriculum_path)
    directory = root / "validation-candidates"
    if selection["curriculum_sha256"] != curriculum_hash or selection[
        "validation_hashes"
    ] != evidence_hashes(directory):
        raise ValueError("Validation evidence changed after selection")
    settings, _, summary, protocol = load_complete(
        directory,
        [p for p in all_prompts if p["split"] == "validation"],
        ["open_base", "closed_quality_bar", *CANDIDATES],
    )
    chosen = selection["selected_arm"]
    if (
        rank(summary) != chosen
        or protocol["candidates"] != selection["candidates"]
        or selection["selected_artifact"] != protocol["candidates"][chosen]["artifact"]
    ):
        raise ValueError("Selection does not match the declared ranking/evidence")
    return selection, all_prompts, settings, protocol


async def execute(stage, root, curriculum_path, candidate_paths):
    root = Path(root)
    if stage == "confirmation":
        selection, all_prompts, validation, previous = selection_for_confirmation(
            root, curriculum_path
        )
        chosen = selection["selected_arm"]
        candidates = {
            chosen: verify_candidate(
                selection["candidates"][chosen]["state_path"], CANDIDATES[chosen]
            )
        }
        if candidates[chosen] != selection["candidates"][chosen]:
            raise ValueError("Selected deployment evidence changed")
        split, directory = "heldout", root / "confirmation-selected"
    else:
        all_prompts, _ = curriculum(curriculum_path)
        candidates = {
            name: verify_candidate(candidate_paths[name], epochs)
            for name, epochs in CANDIDATES.items()
        }
        if len({(c["source_sha256"], c["upload_sha256"]) for c in candidates.values()}) != 1:
            raise ValueError("Candidates were trained on different reviewed datasets")
        configs = [dict(c["training_config"], epochs=None) for c in candidates.values()]
        if configs[0] != configs[1]:
            raise ValueError("Candidate training recipes differ beyond epochs")
        split, directory = "validation", root / "validation-candidates"
    prompts = [p for p in all_prompts if p["split"] == split]
    arms = arms_for(candidates)
    configuration = _configuration(prompts, arms, **POLICY)
    if (
        configuration["scorer"].get("judge_model") != TEACHER
        or configuration["scorer"].get("judge_backend") != "closed_quality_bar"
    ):
        raise ValueError("Expected the declared gpt-6.1-sol scene judge")
    protocol = {
        "schema_version": 1,
        "configuration": configuration,
        "candidates": candidates,
        "curriculum_sha256": read(curriculum_path)["sha256"],
        "environment": environment(),
    }
    if stage == "confirmation":
        if (
            common_configuration(configuration) != common_configuration(validation["configuration"])
            or protocol["environment"] != previous["environment"]
        ):
            raise ValueError(
                "Generation/scoring source, runtime or policy changed after validation"
            )
        earlier = {a["name"]: a for a in validation["configuration"]["arms"]}
        if any(a != earlier[a["name"]] for a in configuration["arms"]):
            raise ValueError("Selected/reference arm serving or sampling configuration changed")
        protocol["selection_sha256"] = sha(root / "selection.json")
    receipt = directory / "protocol.json"
    if receipt.exists():
        if read(receipt) != protocol:
            raise ValueError("Existing evaluation protocol differs; do not silently resume")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Cannot adopt evaluation output without its protocol")
        exclusive_json(receipt, protocol)
    from bufogen.integrations.telemetry import init_weave

    client = init_weave()
    try:
        await run_benchmark(prompts, arms, directory, **POLICY)
        await score_benchmark(directory, concurrency=8, publish=True)
    finally:
        client.finish()


def report(root, curriculum_path):
    root = Path(root)
    selection, prompts, validation, previous = selection_for_confirmation(root, curriculum_path)
    chosen = selection["selected_arm"]
    directory = root / "confirmation-selected"
    settings, rows, summary, protocol = load_complete(
        directory,
        [p for p in prompts if p["split"] == "heldout"],
        ["open_base", "closed_quality_bar", chosen],
    )
    if (
        protocol.get("selection_sha256") != sha(root / "selection.json")
        or common_configuration(settings["configuration"])
        != common_configuration(validation["configuration"])
        or protocol["environment"] != previous["environment"]
    ):
        raise ValueError("Confirmation is not bound to matched validation/selection")
    earlier = {a["name"]: a for a in validation["configuration"]["arms"]}
    if any(a != earlier[a["name"]] for a in settings["configuration"]["arms"]) or protocol[
        "candidates"
    ] != {chosen: selection["candidates"][chosen]}:
        raise ValueError("Confirmation changed selected/reference model or sampling identity")
    improvement = paired_family_interval(rows, chosen, "open_base")
    gap = paired_family_interval(rows, "closed_quality_bar", chosen)
    value = {
        "schema_version": 1,
        "selection_sha256": sha(root / "selection.json"),
        "confirmation_hashes": evidence_hashes(directory),
        "summary": summary,
        "models": {"open_base": BASE, "closed_quality_bar": TEACHER, chosen: "ft-" + BASE},
        "selected_artifact": selection["selected_artifact"],
        "improvement_over_base": improvement,
        "remaining_teacher_gap": gap,
        "point_estimate_within_five_points": gap["difference"] <= 0.05,
        "ci_upper_within_five_points": gap["ci95"][1] <= 0.05,
        "limitations": [
            "16 purpose-built families; approximate cluster interval",
            "One scenario result per prompt; no within-prompt sampling estimate",
            "Teacher and visual judge share gpt-6.1-sol; limited judge calibration",
            "Generation errors and judge abstentions remain in pass denominators",
        ],
    }
    exclusive_json(root / "paired-results.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=["validation", "select", "confirmation", "report"]
    )
    parser.add_argument("--root", type=Path, default=ROOT / "runs/forge-gap")
    parser.add_argument(
        "--curriculum", type=Path, default=ROOT / "runs/forge-gap/curriculum-manifest.json"
    )
    parser.add_argument("--epoch1-state", type=Path)
    parser.add_argument("--epoch3-state", type=Path)
    args = parser.parse_args()
    if args.stage in {"select", "report"}:
        value = (select if args.stage == "select" else report)(args.root, args.curriculum)
        print(json.dumps(value, indent=2))
    else:
        paths = {
            "open_tuned_epoch1": args.epoch1_state or args.root / "training/e1/forge-state.json",
            "open_tuned_epoch3": args.epoch3_state or args.root / "training/e3/forge-state.json",
        }
        asyncio.run(execute(args.stage, args.root, args.curriculum, paths))


if __name__ == "__main__":
    main()
