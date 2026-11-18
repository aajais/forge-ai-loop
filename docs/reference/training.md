# Training: export, Forge SFT, Serverless RL

## Export reviewed trajectories

```bash
uv run bufogen --stage export-reviewed \
  --manifest runs/my-pilot/manifest.jsonl --reviews reviews.jsonl \
  --include-validation --output runs/my-reviewed
```

Repeat `--reviews` for non-overlapping files. The export reuses existing captures and never
regenerates targets. It writes `reviewed-examples.jsonl`, `curation-diagnostics.json` and a
summary. A row qualifies only with an explicit accept, a valid render, real model tool calls and
frozen instructions. Excluded: heldout IDs and families, rejected or unreviewed rows, image-input
trajectories, external prior conversations. Validation rows are exported as `val`, never trained.
Assistant reviews are labeled as assistant; they are not human queue approvals. Keep the source
export immutable after import. The ARIA curation flow is in `aria_prompts/03_curate_visual_dataset.md`.

## Forge Model Distillation (SFT)

The managed service is the primary path; no GPU image is needed. The Forge project must already
exist; these commands never create or reroute one.

```bash
uv run bufogen --stage forge-preflight --project-id <PROJECT_ID>
uv run bufogen --stage forge-import \
  --source runs/my-reviewed/reviewed-examples.jsonl \
  --dataset-name my-reviewed-bufo --name my-sft \
  --project-id <PROJECT_ID> --forge-state runs/my-forge/state.json
uv run bufogen --stage forge-submit --forge-state runs/my-forge/state.json --epochs 1 --batch-size 2
uv run bufogen --stage forge-status --forge-state runs/my-forge/state.json
```

Import validates locally, preflights eligibility, creates the dataset once, dry-runs the exact
entries, then uploads. Submit trains all assistant turns including tool requests. Status polls the
existing job. The state file records source hashes, remote IDs, pending mutations and errors; an
ambiguous write blocks repetition until the remote state is reconciled. Never delete the state
file to retry.

Only a `deployed` response with a real `model_artifact_ref` makes a checkpoint usable. Put the
immutable `wandb-artifact:///...:vN` URI in the matching environment variable (for example
`SFT_Q3_MODEL`) and restart the studio.

Selected recipe for the current result: Qwen/Qwen3.8-27B, 911 rows, all assistant turns, 1 epoch,
batch 2, peak LR 1e-4, linear warmup then decay, LoRA rank fixed by Forge (8 for this base).
Forge project: `wandb-smle/bufo-gen` (alias `bufo-gen`).
See [preflight notes](../forge-preflight.md) for the API surface.

## Serverless RL

```bash
uv sync --locked --extra rl
uv run bufogen --stage train-rl \
  --dataset-ref <entity>/<project>/<rl-tasks-artifact>:v0 \
  --name my-sft-rl --steps 1 --reward astra \
  --initial-checkpoint wandb-artifact:///<entity>/<project>/<sft-name>:v1 \
  --forge-finetune-id <FINETUNE_ID>
```

- Build the environment-task artifact with `rl_contract.write_environment_tasks` (train split
  only; prompt text, requirements, family). Do not reuse the SFT message dataset.
- ART is pinned to upstream commit `45a17cbd` (package `0.5.20`) because the PyPI release lacks the
  Dynamo exact-token support the Gemma path needed.
- Reward: identity × requirement coverage × mean(anatomy, composition, readability); zero for
  missing Bufo, invalid output or unbound scores. The judge runs outside capture; search is off.
- Continuation verifies the deployed job, the immutable artifact, the source run and a positive
  checkpoint step before any rollout. It never silently starts fresh base RL and calls it a
  continuation; omitting both `--initial-checkpoint` and `--forge-finetune-id` selects explicitly
  labeled base RL.
- Inspect `runs/<name>/rl-training.json` for status, steps and artifact. This is not a
  crash-resume scheduler; check remote state before relaunching.

The evaluated RL checkpoint is `bufo-stageb-sft-q3-20261023:v6` (30/48, equal to SFT). The earlier
Gemma continuation produced one verified optimizer step (`bufo-gemma-sol-pilot-20261015:v2`) with
no measured quality change; see [RL live results](../forge-gap-work/rl-live-results.md).

## Legacy stages

`run`, `demo`, `eval`, `eval-identity`, `review-queue`, `curate`, `evaluate`, source-image builders
and GPU `train-sft` remain for compatibility but are not the managed Forge path. The old `curate`
generate-then-review flow does not qualify reviewed IDs; use `export-reviewed`.

## Restoring an old experiment

```bash
uv run python scripts/restore_experiment_runtime.py RUN_DIR TARGET
```

Each run freezes its source under `runs/<name>/frozen`. Runs recorded before the 2026-10-26
package reorganization also need the pre-refactor scaffold listed in
`docs/source-layout-2026-10-26.json`; pass it with `--project-root`. The restore command refuses
to mix layouts. `docs/cleanup-2026-10-26.json` lists what was moved out of the repository.
