"""Populate the W&B Registry `bufo-gen` with the Stage B models, datasets, prompts and evidence.

Existing project artifacts are *linked* (no re-upload) into named registry collections with
descriptive aliases; small evidence files that are not yet artifacts (prompt set, astra
reference set, five-arm evaluation summaries, checkpoint ledger) are logged once as new
artifacts and linked too. One W&B run records the lineage. Re-running is safe: links and
aliases are idempotent, and new artifacts are only re-uploaded when their content changed.

  uv run python scripts/registry_publish.py [--registry bufo-gen] [--dry-run]
"""

import argparse
import re

from bufogen.settings import ROOT, project_path

PINNED = re.compile(r"^[^/\s]+/[^/\s]+/[^/:\s]+:v\d+$")

# (collection, artifact "entity/project/name:vN", type, aliases, description)
LINKS = [
    (
        "bufo-sft-qwen3.8-27b-1ep",
        "wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v1",
        "lora",
        ["selected", "sft_q3"],
        "Selected student: Qwen/Qwen3.8-27B LoRA (rank 8), 1 epoch SFT on 911 reviewed "
        "gpt-6.1-sol trajectories. Validation strict pass 30/48 (Sol 38/48).",
    ),
    (
        "bufo-rl-qwen3.8-27b",
        "wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v22",
        "lora",
        ["rl-block2", "rl_q3"],
        "Serverless RL (OpenPipe ART, astra reward) continued from the selected SFT "
        "checkpoint: block 1 steps 1-6, block 2 steps 6-22. Not selected (RL - SFT inconclusive).",
    ),
    (
        "bufo-rl-qwen3.8-27b",
        "wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v6",
        "lora",
        ["rl-block1"],
        None,
    ),
    (
        "bufo-sft-qwen3.8-27b-3ep",
        "wandb-smle/bufo-gen/bufo-stageb-sft-q1-20261023:v3",
        "lora",
        ["sft_q1"],
        "Qwen/Qwen3.8-27B LoRA (rank 8), 3 epochs SFT; sweep candidate, not selected.",
    ),
    (
        "bufo-sft-gemma4-26b-1ep",
        "wandb-smle/bufo-gen/bufo-stageb-sft-j3-20261023:v1",
        "lora",
        ["sft_j3"],
        "google/gemma-4-26B-A4B-it LoRA (rank 1), 1 epoch SFT; sweep candidate, not selected.",
    ),
    (
        "bufo-sft-gemma4-26b-3ep",
        "wandb-smle/bufo-gen/bufo-stageb-sft-j1-20261023:v3",
        "lora",
        ["sft_j1"],
        "google/gemma-4-26B-A4B-it LoRA (rank 1), 3 epochs SFT; sweep candidate, not selected.",
    ),
    (
        "bufo-sft-nemotron3.5-lightning-1ep",
        "wandb-smle/bufo-gen/bufo-stageb-sft-n3-20261023:v1",
        "lora",
        ["sft_n3"],
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B LoRA (rank 8), 1 epoch SFT; sweep "
        "candidate, not selected.",
    ),
    (
        "bufo-gemma-iter2-e3",
        "wandb-smle/bufo-gen/bufo-gemma-iter2-e3-20261016:v3",
        "lora",
        ["routing-target"],
        "Earlier Gemma adapter (iteration 2, 3 epochs); 45% target in the bufo-gen proxy routing.",
    ),
    (
        "bufo-sol-reviewed-trajectories-911",
        "wandb-smle/bufo-gen/distillation-dataset-da6bd91a-d7b8-478c-ad00-f6f540925537:v0",
        "dataset",
        ["sft-train"],
        "911 reviewed gpt-6.1-sol agent trajectories (train families only) used for every "
        "Stage B SFT candidate; Forge dataset 280e22a6.",
    ),
    (
        "bufo-rl-train-tasks",
        "wandb-smle/bufo-gen/bufo-stage-b-rl-train-tasks:v0",
        "dataset",
        ["rl-train"],
        "256 train-family RL tasks (16 families) for the serverless RL blocks.",
    ),
]

# (collection, type, local paths, aliases, description, metadata)
UPLOADS = [
    (
        "bufo-distill-prompts",
        "prompts",
        [ROOT / "config/distill_prompts.yaml"],
        ["v2"],
        "384 Bufo prompts with exact requirements: 256 train (16 families), 48 validation "
        "(4 families), 80 heldout (16 families); families are disjoint across splits.",
        {"train": 256, "validation": 48, "heldout": 80},
    ),
    (
        "bufo-astra-references-distill-v2",
        "dataset",
        [ROOT / "runs/references/distill-v2"],
        ["train-validation"],
        "gpt-6.1-sol reference renders (two runs) for the astra reference judge on train and "
        "validation prompts. Heldout references are kept out until confirmation.",
        {},
    ),
    (
        "bufo-stage-b-evidence",
        "evaluation",
        [
            ROOT / "runs/distill/stage-b/checkpoints.json",
            ROOT / "runs/distill/stage-b/five-arm-validation/benchmark.json",
            ROOT / "runs/distill/stage-b/five-arm-validation/astra-v5-k3/summary.json",
            ROOT / "runs/distill/stage-b/five-arm-validation/astra-v5-k3/weave-publication.json",
            *sorted((ROOT / "runs/distill/stage-b/five-arm-validation").glob("paired-*.json")),
        ],
        ["five-arm-validation"],
        "Checkpoint ledger and the five-arm validation evaluation (Gemma base, Qwen base, Sol, "
        "sft_q3, rl_q3; astra v5 k=3, 48 prompts) with paired family-bootstrap intervals.",
        {"weave_dataset": "bufo-validation-48", "judge": "gpt-6-astra", "rubric": "bufo-astra-v5"},
    ),
]


def plan(links=LINKS, uploads=UPLOADS):
    """Validate the manifest; returns (links, uploads)."""
    for collection, ref, kind, aliases, _ in links:
        if not PINNED.match(ref):
            raise ValueError(
                f"{collection}: artifact must be pinned entity/project/name:vN, got {ref}"
            )
        if not aliases:
            raise ValueError(f"{collection}: at least one alias")
    names = [c for c, *_ in links] + [c for c, *_ in uploads]
    for collection, kind, paths, aliases, _, _ in uploads:
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise ValueError(f"{collection}: missing files {missing}")
    return links, uploads, sorted(set(names))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registry", default="bufo-gen")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    links, uploads, collections = plan()
    target = lambda name: f"wandb-registry-{args.registry}/{name}"  # noqa: E731
    if args.dry_run:
        for collection, ref, kind, aliases, _ in links:
            print(f"link   {ref} -> {target(collection)} {aliases}")
        for collection, kind, paths, aliases, _, _ in uploads:
            print(
                f"upload {collection} ({kind}) <- {[str(p.relative_to(ROOT)) for p in paths]} {aliases}"
            )
        return
    import wandb

    entity, project = project_path().split("/")
    descriptions = {}
    with wandb.init(entity=entity, project=project, job_type="registry-publish") as run:
        for collection, ref, kind, aliases, description in links:
            artifact = run.use_artifact(ref, type=kind)
            run.link_artifact(artifact, target(collection), aliases=aliases)
            print(f"linked {ref} -> {collection} {aliases}")
            if description:
                descriptions[collection] = description
        for collection, kind, paths, aliases, description, metadata in uploads:
            artifact = wandb.Artifact(
                collection, type=kind, description=description, metadata=metadata
            )
            for path in paths:
                if path.is_dir():
                    artifact.add_dir(str(path))
                else:
                    artifact.add_file(str(path), name=path.name)
            artifact = run.log_artifact(artifact, aliases=["latest"])
            artifact.wait()
            run.link_artifact(artifact, target(collection), aliases=aliases)
            print(f"uploaded {collection}:{artifact.version} ({kind}) -> {collection} {aliases}")
            descriptions[collection] = description
    api = wandb.Api()
    registry = api.registry(args.registry)
    for col in registry.collections():
        if col.name in descriptions and not (col.description or "").strip():
            col.description = descriptions[col.name]
            col.save()
    print("collections:", [c.name for c in api.registry(args.registry).collections()])


if __name__ == "__main__":
    main()
