# Benchmarks and judges

How controlled generation runs are produced, scored, compared and published.

## Prompt sets

`config/forge_prompts.yaml` has 12 train, 4 validation and 100 heldout scenarios in disjoint
families; `config/forge_iteration2_prompts.yaml` and the Stage B sets add more. Each prompt
carries explicit, checkable requirements ("Exactly two orange batons are visible"). Train,
validation and heldout families never overlap. `--benchmark-config` selects another validated
family-partitioned YAML.

## Stages

```bash
# Generate (paid). Always pass --split; generation defaults to heldout.
uv run bufogen --stage benchmark --split validation \
  --backends qwen_base closed_quality_bar --output runs/my-pilot

# Explicit served checkpoints (labels must exist in config/models.yaml)
uv run bufogen --stage benchmark --split validation \
  --checkpoint sft_q3=wandb-artifact:///wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v1 \
  --output runs/my-eval

# Score (separate stage), optionally publishing Weave scorer feedback and Evaluations
uv run bufogen --stage benchmark-score --judge astra --output runs/my-eval --publish

# Human review queue for scene failures
uv run bufogen --stage scene-review-queue --output runs/my-eval
```

Useful bounds: `--limit`, `--concurrency` (default 2), `--max-tokens` (default 6000; Qwen arms
use 20000), `--max-turns` (default 100). Without `--backends` only the given `--checkpoint`
labels run; with neither, the default `open_base` and `closed_quality_bar` arms run.

Each run directory contains `benchmark.json`, frozen source/assets/settings, `manifest.jsonl`,
images, `scored.jsonl`, `summary.json`, `comparison.html` and publication receipts. Generation
failures and judge abstentions stay in the denominators; interrupted tasks are recorded as
failures, never replayed silently.

## Judges

**Astra (`--judge astra`, current).** One structured-output call per drawing to `gpt-6-astra`
with the prompt, the exact requirements, the candidate PNG and a teacher reference PNG. Fields:
`bufo_present`, `similarity`, `anatomy`, `composition`, `readability`, per-requirement checks,
`vs_reference`, `judge_confidence`. Three samples per image are merged conservatively
(`--judge-samples 3`) because single verdicts on borderline contact flipped between runs.
Rubric `bufo-astra-v5` was calibrated on a blinded 40-item human packet (agreement 0.875, zero
judge-pass / human-missing after one craft-note amendment). Record:
`runs/distill/calibration-v1/calibration-record.json`, write-up in
[calibration v1](../distill/calibration-v1.md). The reference anchors the standard; a different
drawing that meets every requirement still passes.

**SceneQuality (`--judge scene`, legacy).** VLM rubric (Bufo presence, similarity, requirements,
anatomy, composition, readability) using `JUDGE_MODEL` or the teacher via `SCENE_JUDGE_BACKEND`.
Early results were judged this way, partly by the teacher itself, which is why they were
re-measured.

Rules: judge configuration and rubric are fingerprinted into every score file. Changing the judge
requires fresh calibration and a new run directory; earlier scores are never relabeled. Neither
the teacher nor the student is used as judge.

## Comparing arms

Paired differences use a family-cluster percentile bootstrap (20,000 replicates, seed 20261015),
reported with 95% intervals; see `docs/forge-gap-work/paired_statistics.py` and
`scripts/paired_across_runs.py`. Only arms with matching fingerprints, prompt sets and budgets are
paired. Repeated runs are never pooled.

`--publish` writes native Weave scorer feedback and Evaluation objects; local evidence always
remains authoritative. Publication receipts are safe to retry.

## Loop report

```bash
uv run bufogen --stage loop-report --loop-root runs/forge-loop \
  --forge-state runs/forge-loop/forge-state.json \
  --rl-state runs/forge-gap/rl-direct-result.json --output runs/forge-loop/showcase
```

Builds a standalone HTML showcase plus `loop-report.json` from existing evidence: phase states,
paired images, queue and evaluation links, calibration limits, training artifacts and per-run
metrics. No network writes. The pinned demo comparison is built with
`python -m bufogen.workflows.demo_comparison` from `config/forge-demo-comparison.json`.
