"""Matched planned-scene evaluation; generation stages make paid inference calls.

Validation compares five arms. Only new e1/e3 checkpoints are selectable. Fit is
an in-sample diagnostic; confirmation is a fresh 48-prompt, three-arm comparison.
No stage trains a model, changes deployment routes, or updates .env.
"""

import argparse
import asyncio
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bufogen.evaluation.benchmark import _configuration
from bufogen.evaluation.benchmark import comparison_fingerprint
from bufogen.evaluation.benchmark import load_benchmark
from bufogen.evaluation.benchmark import run_benchmark
from bufogen.evaluation.benchmark import score_benchmark
from bufogen.generation import Manifest
from bufogen.workflows.loop_report import _benchmark
from bufogen.settings import ROOT

# Reuse evidence validation and read-only identity verification, never old split
# definitions, hardcoded curriculum hashes, selection rules, or stage execution.
HELPERS = ROOT / "docs/forge-gap-work"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))
import evaluate_candidates as shared  # noqa: E402
from paired_statistics import paired_family_interval  # noqa: E402

BASE, TEACHER = shared.BASE, shared.TEACHER
POLICY = dict(concurrency=12, temperature=0.7, max_tokens=6000, max_turns=8)
CANDIDATES = {"open_tuned_epoch1": 1, "open_tuned_epoch3": 3}
PRIOR = "open_tuned_prior"
VALIDATION_ARMS = ["open_base", "closed_quality_bar", PRIOR, *CANDIDATES]
DIRECTORIES = {
    "validation": "validation-candidates",
    "fit": "training-fit-selected",
    "confirmation": "confirmation-selected",
}
RULE = "new candidates only: strict passes; scored-only constraint coverage; fewer generation errors; one epoch"
IMMUTABLE = r"wandb-artifact:///[^/]+/[^/]+/[^/]+:v[0-9]+"
read, sha, digest = shared.read, shared.sha, shared.digest
exclusive_json, evidence_hashes = shared.exclusive_json, shared.evidence_hashes
arms_for, common_configuration = shared.arms_for, shared.common_configuration


def curriculum(path):
    manifest = read(path)
    source = Path(manifest["source"])
    if not source.is_absolute():
        source = ROOT / source
    if sha(source) != manifest["sha256"]:
        raise ValueError("Curriculum hash changed")
    prompts = load_benchmark(source)
    if Counter(p["split"] for p in prompts) != {"train": 80, "validation": 24, "heldout": 48}:
        raise ValueError("Expected frozen 80/24/48 curriculum coverage")
    families = {
        split: {p["family"] for p in prompts if p["split"] == split}
        for split in ("train", "validation", "heldout")
    }
    if any(
        families[a] & families[b]
        for a, b in (("train", "validation"), ("train", "heldout"), ("validation", "heldout"))
    ) or [len(families[s]) for s in ("train", "validation", "heldout")] != [16, 4, 12]:
        raise ValueError("Curriculum families overlap or have unexpected coverage")
    return prompts, manifest["sha256"]


def environment():
    return {**shared.environment(), "iteration2_evaluator_sha256": sha(__file__)}


def freeze_protocol(directory, protocol):
    directory = Path(directory)
    path = directory / "protocol.json"
    if path.exists():
        if read(path) != protocol:
            raise ValueError("Existing evaluation protocol changed; cannot resume")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("Cannot adopt outputs without a frozen protocol")
    else:
        exclusive_json(path, protocol)


def verify_new_candidate(root, name):
    root = Path(root)
    epoch = CANDIDATES[name]
    folder = root / "training" / f"e{epoch}"
    path = folder / "forge-state.json"
    state, spec = read(path), read(folder / "candidate.json")
    receipt_path = root / "training/source-receipt.json"
    receipt = read(receipt_path)
    sources = receipt.get("source_files", [])
    if not any(
        Path(item["path"]).resolve() == (root / "curriculum-manifest.json").resolve()
        for item in sources
    ) or any(sha(item["path"]) != item["sha256"] for item in sources):
        raise ValueError("Reviewed source evidence changed or omits the current curriculum")
    expected = {"epochs": epoch, "batch_size": 2, "learning_rate": 1e-4, "assistant_turns": "all"}
    if (
        state.get("training_config") != expected
        or spec.get("training_config") != expected
        or spec.get("initialization") != "independent_base_model"
        or spec.get("base_model") != BASE
        or state.get("base_model") != BASE
        or spec.get("candidate") != f"e{epoch}"
        or spec.get("source_receipt_sha256") != sha(receipt_path)
        or state.get("lineage", {}).get("source_receipt_sha256") != sha(receipt_path)
        or state.get("lineage", {}).get("initialization") != "independent_base_model"
    ):
        raise ValueError("Candidate recipe or source-receipt lineage changed")
    for key in ("source_sha256", "upload_sha256"):
        if spec.get(key) != receipt.get(key) or state.get(key) != receipt.get(key):
            raise ValueError("Candidate differs from reviewed training source")
    counts = receipt["entry_counts"]
    if state.get("entry_counts") != {"train": counts["train"], "val": counts["validation"]}:
        raise ValueError("Candidate training/validation counts differ from reviewed data")
    if (
        Path(state["source_path"]).resolve()
        != (root / "training/reviewed-examples.jsonl").resolve()
        or sha(state["source_path"]) != receipt["source_sha256"]
    ):
        raise ValueError("Candidate source bytes or path changed")
    value = shared.verify_candidate(path, epoch)
    if not re.fullmatch(IMMUTABLE, value["artifact"]):
        raise ValueError("Candidate must resolve to an immutable artifact version")
    return {
        **value,
        "candidate_spec_sha256": sha(folder / "candidate.json"),
        "source_receipt_sha256": sha(receipt_path),
    }


def verify_prior(historical):
    selection = read(Path(historical) / "selection.json")
    if (
        digest({k: v for k, v in selection.items() if k != "receipt_sha256"})
        != selection["receipt_sha256"]
    ):
        raise ValueError("Historical selection receipt changed")
    prior = selection["candidates"][selection["selected_arm"]]
    if prior["epochs"] != 1 or prior["artifact"] != selection["selected_artifact"]:
        raise ValueError("Prior control must be the selected one-epoch checkpoint")
    live = shared.verify_candidate(prior["state_path"], 1)
    if live != prior:
        raise ValueError("Prior checkpoint/job identity changed")
    return live


def rank(summary):
    def key(name):
        arm = summary["arms"][name]
        coverage = arm["metric_means_scored_only"]["constraint_coverage"]
        if coverage is not None and not math.isfinite(coverage):
            raise ValueError("Nonfinite coverage")
        return (
            arm["provisional_passes"],
            coverage if coverage is not None else -1,
            -arm["generation_errors"],
            -CANDIDATES[name],
        )

    return max(CANDIDATES, key=key)


def validated_evidence(root, stage, curriculum_path=None):
    root = Path(root)
    curriculum_path = curriculum_path or root / "curriculum-manifest.json"
    prompts, curriculum_hash = curriculum(curriculum_path)
    directory = root / DIRECTORIES[stage]
    if read(directory / "protocol.json").get("stage") != stage:
        raise ValueError("Evaluation stage differs from its protocol")
    if stage == "validation":
        selected_prompts = [p for p in prompts if p["split"] == "validation"]
        result = shared.load_complete(directory, selected_prompts, VALIDATION_ARMS)
        if result[3]["curriculum_sha256"] != curriculum_hash:
            raise ValueError("Evaluation curriculum changed")
        return result
    selection, _, validation, previous = selected_context(root, curriculum_path)
    chosen = selection["selected_arm"]
    if stage == "confirmation":
        result = shared.load_complete(
            directory,
            [p for p in prompts if p["split"] == "heldout"],
            ["open_base", "closed_quality_bar", chosen],
        )
    else:
        expected, _ = fixed_fit_prompts(prompts, root, selection["candidates"][chosen])
        protocol = read(directory / "protocol.json")
        settings = read(directory / "benchmark.json")
        evidence = _benchmark(directory)
        if (
            evidence["status"] != "scored"
            or evidence["generated"] != 16
            or settings["configuration"]["prompts"] != expected
            or set(evidence["arms"]) != {chosen}
            or digest(settings["configuration"]) != settings["configuration_hash"]
            or comparison_fingerprint(settings["configuration"])
            != settings["comparison_fingerprint"]
            or protocol["configuration"] != settings["configuration"]
        ):
            raise ValueError("Incomplete or mismatched training-fit evidence")
        result = (
            settings,
            Manifest(directory / "scored.jsonl").read(),
            read(directory / "summary.json"),
            protocol,
        )
    settings, _, _, protocol = result
    if (
        protocol.get("selection_sha256") != sha(root / "selection.json")
        or protocol["curriculum_sha256"] != curriculum_hash
        or protocol["environment"] != previous["environment"]
        or common_configuration(settings["configuration"])
        != common_configuration(validation["configuration"])
        or protocol["candidates"] != {chosen: selection["candidates"][chosen]}
    ):
        raise ValueError("Result does not match frozen selection and validation harness")
    previous_arms = {a["name"]: a for a in validation["configuration"]["arms"]}
    if any(a != previous_arms[a["name"]] for a in settings["configuration"]["arms"]):
        raise ValueError("Selected/reference serving settings changed")
    return result


def select(root, curriculum_path):
    root = Path(root)
    confirmation = root / DIRECTORIES["confirmation"]
    if confirmation.exists() and any(confirmation.iterdir()):
        raise ValueError("Selection must precede confirmation evidence")
    settings, _, summary, protocol = validated_evidence(root, "validation", curriculum_path)
    chosen = rank(summary)
    value = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": RULE,
        "selected_arm": chosen,
        "selected_artifact": protocol["candidates"][chosen]["artifact"],
        "candidates": protocol["candidates"],
        "validation_summary": summary,
        "validation_hashes": evidence_hashes(root / DIRECTORIES["validation"]),
        "validation_run_id": settings["run_id"],
        "curriculum_sha256": protocol["curriculum_sha256"],
        "confirmation_not_started": True,
    }
    value["receipt_sha256"] = digest(value)
    exclusive_json(root / "selection.json", value)
    return value


def selected_context(root, curriculum_path):
    root = Path(root)
    selection = read(root / "selection.json")
    if (
        selection.get("rule") != RULE
        or digest({k: v for k, v in selection.items() if k != "receipt_sha256"})
        != selection["receipt_sha256"]
    ):
        raise ValueError("Selection receipt changed")
    if selection["validation_hashes"] != evidence_hashes(root / DIRECTORIES["validation"]):
        raise ValueError("Validation evidence changed after selection")
    settings, _, summary, protocol = validated_evidence(root, "validation", curriculum_path)
    chosen = selection["selected_arm"]
    prompts, curriculum_hash = curriculum(curriculum_path)
    if (
        rank(summary) != chosen
        or selection["curriculum_sha256"] != curriculum_hash
        or selection["candidates"] != protocol["candidates"]
        or selection["selected_artifact"] != protocol["candidates"][chosen]["artifact"]
        or selection["validation_run_id"] != settings["run_id"]
        or selection["validation_summary"] != summary
    ):
        raise ValueError("Selection does not match declared ranking and evidence")
    return selection, prompts, settings, protocol


def fixed_fit_prompts(prompts, root, candidate):
    chosen = sorted(
        [
            p
            for p in prompts
            if p["split"] == "train"
            and p.get("curriculum_metadata", {}).get("difficulty") == "contact"
        ],
        key=lambda p: p["id"],
    )
    if len(chosen) != 16 or len({p["family"] for p in chosen}) != 16:
        raise ValueError("Need exactly 16 fixed plain contact training prompts")
    source = Path(root) / "training/reviewed-examples.jsonl"
    if sha(source) != candidate["source_sha256"]:
        raise ValueError("Training-fit source changed")
    targets = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    provenance = []
    for prompt in chosen:
        matches = [r for r in targets if r.get("prompt_id") == prompt["id"]]
        if len(matches) != 1:
            raise ValueError("Training-fit needs an accepted target for every fixed contact prompt")
        target = matches[0]
        review = target.get("training_review", {})
        if (
            target.get("partition") != "train"
            or target.get("prompt_family") != prompt["family"]
            or target.get("prompt") != prompt["prompt"]
            or review.get("decision") != "accept"
            or review.get("visual_quality_pass") is not True
            or review.get("prompt_id") != prompt["id"]
            or review.get("partition") != "train"
            or review.get("source_generation_id") != target.get("source_generation_id")
            or not target.get("source_generation_id")
            or not review.get("png_sha256")
        ):
            raise ValueError("Training-fit target provenance differs")
        provenance.append(
            {
                "prompt_id": prompt["id"],
                "target_sha256": digest(target),
                "source_generation_id": target["source_generation_id"],
                "review_png_sha256": review["png_sha256"],
            }
        )
    return chosen, provenance


async def execute(stage, root, curriculum_path, historical):
    root = Path(root)
    if stage not in DIRECTORIES:
        raise ValueError("Unsupported generation stage")
    provenance = []
    if stage == "validation":
        all_prompts, curriculum_hash = curriculum(curriculum_path)
        candidates = {
            PRIOR: verify_prior(historical),
            **{name: verify_new_candidate(root, name) for name in CANDIDATES},
        }
        new = [candidates[name] for name in CANDIDATES]
        if len({(c["source_sha256"], c["upload_sha256"]) for c in new}) != 1:
            raise ValueError("New candidates used different reviewed datasets")
        prompts = [p for p in all_prompts if p["split"] == "validation"]
    else:
        selection, all_prompts, validation, previous = selected_context(root, curriculum_path)
        curriculum_hash = selection["curriculum_sha256"]
        chosen = selection["selected_arm"]
        candidates = {chosen: verify_new_candidate(root, chosen)}
        if candidates[chosen] != selection["candidates"][chosen]:
            raise ValueError("Selected checkpoint identity changed")
        if stage == "fit":
            prompts, provenance = fixed_fit_prompts(all_prompts, root, candidates[chosen])
        else:
            prompts = [p for p in all_prompts if p["split"] == "heldout"]
    arms = arms_for(candidates)
    if stage == "fit":
        arms = [a for a in arms if a.name in candidates]
    configuration = _configuration(prompts, arms, **POLICY)
    if (
        configuration["scorer"].get("judge_model") != TEACHER
        or configuration["scorer"].get("judge_backend") != "closed_quality_bar"
    ):
        raise ValueError("Expected declared Sol visual judge")
    protocol = {
        "schema_version": 1,
        "stage": stage,
        "configuration": configuration,
        "candidates": candidates,
        "curriculum_sha256": curriculum_hash,
        "environment": environment(),
        "native_weave_publish": True,
    }
    if stage != "validation":
        if (
            common_configuration(configuration) != common_configuration(validation["configuration"])
            or protocol["environment"] != previous["environment"]
        ):
            raise ValueError("Generation/scoring source or runtime changed after validation")
        old_arms = {a["name"]: a for a in validation["configuration"]["arms"]}
        if any(a != old_arms[a["name"]] for a in configuration["arms"]):
            raise ValueError("Serving or sampling configuration changed")
        protocol["selection_sha256"] = sha(root / "selection.json")
    if stage == "fit":
        protocol.update(
            purpose="in_sample_training_fit_only",
            not_heldout_evidence=True,
            not_candidate_selection=True,
            training_target_provenance=provenance,
        )
    directory = root / DIRECTORIES[stage]
    freeze_protocol(directory, protocol)
    from bufogen.integrations.telemetry import init_weave

    client = init_weave()
    try:
        await run_benchmark(prompts, arms, directory, **POLICY)
        await score_benchmark(directory, concurrency=8, publish=True)
        validated_evidence(root, stage, curriculum_path)
    finally:
        client.finish()
    return directory


def summarize(root, curriculum_path):
    root = Path(root)
    selection, _, _, _ = selected_context(root, curriculum_path)
    _, rows, summary, _ = validated_evidence(root, "confirmation", curriculum_path)
    chosen = selection["selected_arm"]
    options = dict(families=12, per_family=4, seed=20261016)
    improvement = paired_family_interval(rows, chosen, "open_base", **options)
    gap = paired_family_interval(rows, "closed_quality_bar", chosen, **options)
    value = {
        "schema_version": 1,
        "selection_sha256": sha(root / "selection.json"),
        "confirmation_hashes": evidence_hashes(root / DIRECTORIES["confirmation"]),
        "summary": summary,
        "selected_artifact": selection["selected_artifact"],
        "models": {"open_base": BASE, "closed_quality_bar": TEACHER, chosen: "ft-" + BASE},
        "improvement_over_base": improvement,
        "remaining_teacher_gap": gap,
        "point_estimate_within_five_points": gap["difference"] <= 0.05,
        "ci_upper_within_five_points": gap["ci95"][1] <= 0.05,
        "limitations": [
            "12 purpose-built families; approximate paired family-cluster interval",
            "One result per prompt, including generation failures and judge abstentions",
            "Same Sol teacher and visual judge; provisional scorer can over-credit contacts",
            "Historical runs are not pooled; all current arms share the updated agent",
            "Training fit is not heldout quality evidence",
        ],
    }
    target = root / "paired-results.json"
    if target.exists():
        if read(target) != value:
            raise ValueError("Existing paired result changed")
    else:
        exclusive_json(target, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["validation", "select", "fit", "confirmation", "summarize"],
    )
    parser.add_argument("--root", type=Path, default=ROOT / "runs/forge-iteration2")
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--historical-root", type=Path, default=ROOT / "runs/forge-gap")
    args = parser.parse_args()
    path = args.curriculum or args.root / "curriculum-manifest.json"
    if args.stage in {"select", "summarize"}:
        value = (select if args.stage == "select" else summarize)(args.root, path)
        print(json.dumps(value, indent=2))
    else:
        print(asyncio.run(execute(args.stage, args.root, path, args.historical_root)))


if __name__ == "__main__":
    main()
