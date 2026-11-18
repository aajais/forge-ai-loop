"""Fixed in-sample diagnostic of the immutable validation-selected adapter.

Local preflight (no network or generation):
    uv run python docs/forge-gap-work/evaluate_training_fit.py

After selection.json exists, explicitly run 16 generations and native Weave eval:
    uv run python docs/forge-gap-work/evaluate_training_fit.py --execute

The 16 IDs below were frozen before inspecting diagnostic outputs. Every prompt
must have an accepted target in the exact reviewed training source. This measures
fit on known training actions, not heldout performance, and cannot select a new
candidate or modify confirmation. Outputs go to training-fit-selected, with a
separate protocol binding selection, curriculum, training data, code and policy.
Existing output may resume only if that entire protocol remains identical.
"""

import argparse
import asyncio
import json
from pathlib import Path

from evaluate_candidates import (
    BASE,
    CANDIDATES,
    POLICY,
    arms_for,
    common_configuration,
    curriculum,
    environment,
    exclusive_json,
    read,
    selection_for_confirmation,
    sha,
    verify_candidate,
)

from bufogen.evaluation.benchmark import _configuration
from bufogen.evaluation.benchmark import digest
from bufogen.evaluation.benchmark import run_benchmark
from bufogen.evaluation.benchmark import score_benchmark
from bufogen.settings import ROOT

CURRICULUM_SHA256 = "6f31c0774cf37f8ff901172e74a28856b0a34cf75156dc31e37f606844ac9bfc"
TRAINING_SHA256 = "e8f2f4b23c30ae73bd8fd394ac340e0b6d299f3c277307b70ce1d2d3dc746984"
FIXED_IDS = (
    "gap-train-archery_range-base",
    "gap-train-bakery-base",
    "gap-train-basketball_court-base",
    "gap-train-concert_stage-base",
    "gap-train-f1_presentation-base",
    "gap-train-film_set-base",
    "gap-train-gpu_garden-base",
    "gap-train-laundry_room-base",
    "gap-train-office_coffee-base",
    "gap-train-ping_pong-base",
    "gap-train-pizza_kitchen-base",
    "gap-train-rainy_commute-base",
    "gap-train-skate_park-base",
    "gap-train-tennis_court-base",
    "gap-train-video_call-base",
    "gap-train-workshop_repair-base",
)


def fixed_prompts(all_prompts, targets):
    """Fail on absent/rejected targets, split overlap, or substituted base IDs."""
    base = [
        p for p in all_prompts if p["split"] == "train" and p["id"].endswith("-base")
    ]
    if len(base) != 16 or {p["id"] for p in base} != set(FIXED_IDS):
        raise ValueError("Expected the exact 16 frozen training base prompt IDs")
    families = {p["family"] for p in base}
    outside = {p["family"] for p in all_prompts if p["split"] != "train"}
    if len(families) != 16 or families & outside:
        raise ValueError("Training-fit families overlap or are not 16 distinct families")
    by_id = {p["id"]: p for p in base}
    provenance = []
    for prompt_id in FIXED_IDS:
        prompt = by_id[prompt_id]
        matches = [row for row in targets if row.get("prompt_id") == prompt_id]
        if len(matches) != 1:
            raise ValueError(f"Need one accepted training target for {prompt_id}; no substitution")
        row = matches[0]
        review = row.get("training_review", {})
        if (
            row.get("partition") != "train"
            or row.get("prompt_family") != prompt["family"]
            or row.get("prompt") != prompt["prompt"]
            or review.get("decision") != "accept"
            or review.get("visual_quality_pass") is not True
            or review.get("partition") != "train"
            or review.get("prompt_id") != prompt_id
            or review.get("prompt_family") != prompt["family"]
            or not row.get("source_generation_id")
            or review.get("source_generation_id") != row["source_generation_id"]
            or not review.get("png_sha256")
        ):
            raise ValueError(f"Accepted training-target provenance differs for {prompt_id}")
        provenance.append(
            {
                "prompt_id": prompt_id,
                "source_generation_id": row["source_generation_id"],
                "review_png_sha256": review["png_sha256"],
                "target_sha256": digest(row),
            }
        )
    return [by_id[identifier] for identifier in FIXED_IDS], provenance


def local_preflight(root, curriculum_path):
    all_prompts, curriculum_hash = curriculum(curriculum_path)
    if curriculum_hash != CURRICULUM_SHA256:
        raise ValueError("Training-fit curriculum is not the frozen experiment")
    source = Path(root) / "training/reviewed-examples.jsonl"
    if sha(source) != TRAINING_SHA256:
        raise ValueError("Training-fit source differs from the final reviewed dataset")
    targets = [json.loads(line) for line in source.read_text().splitlines()]
    prompts, provenance = fixed_prompts(all_prompts, targets)
    return prompts, provenance, source


def freeze_protocol(directory, protocol):
    directory = Path(directory)
    receipt = directory / "protocol.json"
    if receipt.exists():
        if read(receipt) != protocol:
            raise ValueError("Existing training-fit protocol differs; refusing to resume")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Cannot adopt training-fit output without its protocol")
        exclusive_json(receipt, protocol)


async def execute(root, curriculum_path):
    root = Path(root)
    # This validates selection and complete validation evidence via load_complete.
    # It reads no confirmation output, and precedes even live metadata requests.
    selection, all_prompts, validation, previous = selection_for_confirmation(
        root, curriculum_path
    )
    prompts, target_provenance, source = local_preflight(root, curriculum_path)
    if [p for p in all_prompts if p["id"] in FIXED_IDS] != [
        p for p in curriculum(curriculum_path)[0] if p["id"] in FIXED_IDS
    ]:
        raise ValueError("Curriculum changed during training-fit preflight")
    chosen = selection["selected_arm"]
    expected = selection["candidates"][chosen]
    candidate = verify_candidate(expected["state_path"], CANDIDATES[chosen])
    if candidate != expected or candidate["artifact"] != selection["selected_artifact"]:
        raise ValueError("Selected immutable deployment evidence changed")
    if (
        candidate["source_sha256"] != TRAINING_SHA256
        or Path(read(candidate["state_path"])["source_path"]).resolve() != source.resolve()
    ):
        raise ValueError("Selected adapter was not trained on this reviewed source")
    arms = [arm for arm in arms_for({chosen: candidate}) if arm.name == chosen]
    if len(arms) != 1:
        raise ValueError("Training-fit must evaluate exactly the selected adapter")
    configuration = _configuration(prompts, arms, **POLICY)
    current_environment = environment()
    if (
        common_configuration(configuration)
        != common_configuration(validation["configuration"])
        or current_environment != previous["environment"]
    ):
        raise ValueError("Source, runtime or policy changed from the selected comparison")
    earlier = {arm["name"]: arm for arm in validation["configuration"]["arms"]}
    if configuration["arms"] != [earlier[chosen]]:
        raise ValueError("Selected arm serving or sampling settings changed")
    protocol = {
        "schema_version": 1,
        "purpose": "in_sample_training_fit_diagnostic_only",
        "not_heldout_evidence": True,
        "not_candidate_selection": True,
        "fixed_prompt_ids": list(FIXED_IDS),
        "selection_sha256": sha(root / "selection.json"),
        "curriculum_sha256": CURRICULUM_SHA256,
        "training_source_sha256": TRAINING_SHA256,
        "training_target_provenance": target_provenance,
        "candidate": candidate,
        "model_id": "ft-" + BASE,
        "configuration": configuration,
        "environment": current_environment,
        "diagnostic_runner_sha256": sha(__file__),
        "native_weave_publish": True,
    }
    directory = root / "training-fit-selected"
    freeze_protocol(directory, protocol)
    from bufogen.integrations.telemetry import init_weave

    client = init_weave()
    try:
        await run_benchmark(prompts, arms, directory, **POLICY)
        await score_benchmark(directory, concurrency=8, publish=True)
    finally:
        client.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/forge-gap")
    parser.add_argument(
        "--curriculum", type=Path, default=ROOT / "runs/forge-gap/curriculum-manifest.json"
    )
    parser.add_argument("--execute", action="store_true", help="Run paid generation/scoring")
    args = parser.parse_args()
    if args.execute:
        asyncio.run(execute(args.root, args.curriculum))
    else:
        prompts, _, source = local_preflight(args.root, args.curriculum)
        print(json.dumps({
            "purpose": "in_sample_training_fit_diagnostic_only",
            "generation_started": False,
            "prompt_ids": [p["id"] for p in prompts],
            "accepted_targets": len(prompts),
            "source_sha256": sha(source),
        }, indent=2))


if __name__ == "__main__":
    main()
