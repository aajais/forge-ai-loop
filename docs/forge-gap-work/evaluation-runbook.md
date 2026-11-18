# Matched validation and confirmation runbook

This is an execution proposal based on the current local benchmark APIs. No generations, scores, training jobs, or confirmation outputs were produced or opened while writing it. The existing teacher-curriculum run is independent development evidence; do not relabel its rows into this comparison.

## Fixed experiment

Use the reviewed, hash-frozen `config/forge_gap_prompts.yaml`:

- Validation: all 24 prompts, four families with six variants each.
- Confirmation: all 80 prompts, 16 families with five variants each. Open this split only after persisting the validation selection receipt.
- Generation: `temperature=0.7`, `max_tokens=6000`, `max_turns=8`, generation concurrency 16, scorer concurrency 8. The current benchmark disables web search, uses the same tool/prompt/reference policy for every arm, and explicitly omits temperature for the OpenAI provider. Report this provider difference rather than claiming identical effective sampling.
- Scorer: the existing frozen `SceneQuality` configuration with `SCENE_JUDGE_BACKEND=closed_quality_bar` and actual `gpt-6.1-sol` judge. Preserve the rubric/configuration hash and its limited calibration status.
- One final scenario result per prompt/arm. The existing benchmark may execute up to two generation attempts under its recorded policy; do not add teacher-only search or student-only retries.
- Generation failures and judge abstentions remain in the strict-pass denominator. Report them separately; do not call a judge abstention a confirmed visual failure.

Validation arms:

| Internal arm ID | Display model ID | Actual request model |
|---|---|---|
| `open_base` | `google/gemma-4-26B-A4B-it` | Exact base ID |
| `closed_quality_bar` | `gpt-6.1-sol` | Exact teacher ID |
| `open_tuned_epoch1` | `ft-google/gemma-4-26B-A4B-it` | Verified immutable one-epoch artifact URI |
| `open_tuned_epoch3` | `ft-google/gemma-4-26B-A4B-it` | Verified immutable three-epoch artifact URI |

Keep the two fine-tuned labels identical as requested, but show epoch recipe and full artifact URI in separate columns/metadata. Never replace the request URI with the display label. Both candidates start from the exact same base and use the same accepted train/validation dataset; the three-epoch candidate is not a continuation of the one-epoch candidate unless a separately declared experiment intends that.

## Calls using existing APIs

Use a dedicated Python orchestration script rather than inventing CLI backend names. `run_benchmark` accepts a list of `BackendConfig` instances with arbitrary unique `name` values. The following functions are a directly usable skeleton; the caller must supply real artifact URIs from completed Forge job and W&B immutable-artifact receipts.

```python
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from bufogen.evaluation.benchmark import load_benchmark, run_benchmark, score_benchmark
from bufogen.models import load_backends
from bufogen.settings import ROOT
from bufogen.integrations.telemetry import init_weave

BASE = "google/gemma-4-26B-A4B-it"
TEACHER = "gpt-6.1-sol"
POLICY = dict(concurrency=16, temperature=0.7, max_tokens=6000, max_turns=8)

def frozen_prompts(split):
    manifest = json.loads((ROOT / "runs/forge-gap/curriculum-manifest.json").read_text())
    path = ROOT / manifest["source"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    rows = [p for p in load_benchmark(path) if p["split"] == split]
    assert len(rows) == {"validation": 24, "heldout": 80}[split]
    return rows

def comparison_arms(checkpoints):
    # checkpoints maps unique candidate arm IDs to verified immutable URIs.
    configs = load_backends()
    assert configs["open_base"].model == BASE
    assert configs["closed_quality_bar"].model == TEACHER
    assert len(set(checkpoints.values())) == len(checkpoints)
    arms = [configs["open_base"], configs["closed_quality_bar"]]
    for name, uri in checkpoints.items():
        assert name not in {a.name for a in arms}
        assert re.fullmatch(r"wandb-artifact:///[^/]+/[^/]+/[^/]+:v[0-9]+", uri)
        arms.append(replace(
            configs["open_tuned"], name=name, model=uri,
            label="ft-" + BASE, base_model=BASE, tuning_method="sft",
        ))
    return arms

async def evaluate_split(split, checkpoints, output_dir):
    prompts = frozen_prompts(split)
    arms = comparison_arms(checkpoints)
    client = init_weave()
    try:
        await run_benchmark(prompts, arms, Path(output_dir), **POLICY)
        rows = await score_benchmark(Path(output_dir), concurrency=8, publish=True)
        assert len(rows) == len(prompts) * len(arms)
        return rows
    finally:
        client.finish()
```

First call:

```python
await evaluate_split(
    "validation",
    {"open_tuned_epoch1": verified_epoch1_uri,
     "open_tuned_epoch3": verified_epoch3_uri},
    ROOT / "runs/forge-gap/validation-candidates",
)
```

This produces **96** final scenario rows. Validate all frozen rows, scored-row bindings, PNG hashes, expected prompt/arm Cartesian product, and native Weave coverage before selection. `score_benchmark` already checks these generation/scoring invariants and caches existing judge errors without silently retrying them.

After selection, call with only the selected adapter and both references:

```python
await evaluate_split(
    "heldout",
    {selection["selected_arm"]: selection["selected_artifact"]},
    ROOT / "runs/forge-gap/confirmation-selected",
)
```

This produces **240** final scenario rows. Do not run both candidates on confirmation and then choose the better one. Preserve the selected candidate arm ID and its immutable URI. A separately launched RL proof is not automatically a candidate in this selection; adding one requires declaring the candidate set before examining validation outcomes and evaluating it under the same policy.

## Selection receipt, before confirmation

Read the completed validation `summary.json`. Among the two SFT candidates only, maximize this predeclared tuple:

1. `provisional_passes` across all 24 prompts.
2. `metric_means_scored_only.constraint_coverage`; missing/null coverage sorts below any observed value.
3. Negative `generation_errors` (fewer is better).
4. If still tied, choose one epoch as the lower-training-cost deterministic tie-break. Record this tie-break before reading candidate results.

The coverage tie-break is scored-only because that is the existing manifest's metric. Report its scored-row denominator and judge-error count beside it; do not silently substitute all-attempt coverage after seeing results. Store both candidates' complete summaries even if the winning margin is one example. Four validation families give a noisy selection signal.

Write `runs/forge-gap/selection.json` exclusively/atomically with:

- Timestamp, full ranking rule and fixed candidate set.
- Curriculum SHA-256, validation `run_id`, `configuration_hash`, and `comparison_fingerprint`.
- Hashes of `benchmark.json`, `manifest.jsonl`, `scored.jsonl`, and `summary.json`.
- Candidate artifact URIs, artifact IDs/digests, Forge job IDs, epochs, and train/validation dataset hash/provenance.
- All ranking tuple values, selected arm ID and selected artifact URI.
- An explicit declaration that confirmation results have not been generated or inspected for selection.

The teacher's reviewed validation targets may be supplied to Forge's validation dataset as planned; they must not appear among gradient-training examples. Actual candidate training exports and job receipts must establish that separation. Do not use teacher filtered acceptance counts as the teacher's evaluation pass rate: every validation prompt gets a fresh matched evaluation attempt.

## Source and policy fingerprints

Use one directory/run for all arms of a split whenever feasible. It freezes one prompt set, source set, scorer and generation policy before any arm runs. `configuration_hash` includes arm names/IDs and concurrency; `comparison_fingerprint` intentionally excludes the arms, generation concurrency, and `benchmark.py` file hash, while retaining generation/scorer source hashes and most runtime policy.

- Separate arm-only runs may be compared only when their `comparison_fingerprint` values match **and** model-specific sampling/budget fields have been audited. The fingerprint excludes the entire arm list, including each arm's sampling metadata; fingerprint equality alone does not establish per-arm sampling equality.
- Validation and confirmation fingerprints will differ because their prompt sets differ. For cross-split continuity, compare policy, source hashes, runtime configuration, prompt version, and scorer directly; do not require their whole fingerprints to match or remove prompts from saved fingerprints.
- The 152-row teacher curriculum contains train plus validation prompts and has another prompt hash; its fingerprint cannot equal a 24-row validation-only run. Do not transplant its teacher rows, mutate frozen manifests, or infer comparability from shared prompt IDs alone.
- Avoid source/dependency changes between candidate validation and confirmation. In particular, `agent.py` and `models.py` are fingerprinted, including naming and RL-only optional changes. Freeze the current approved source after those changes. A running teacher process must not be resumed under changed source; use its own recorded evidence and preserve any mismatch honestly.
- Runtime fingerprints currently enumerate Python, `openai-agents`, `weave`, and `cwsandbox`, not every installed library. Retain a full normalized dependency/version snapshot plus `uv.lock` hash for the comparison. The ART revision affects RL training provenance even though ART is not part of ordinary benchmark package metadata.
- Resume only the identical saved configuration in the same directory. Existing terminal failures remain failures. A deliberate rerun gets a new directory and stays separate; do not pool or choose the better sample retroactively.

## Paired family-cluster uncertainty

Each confirmation family has five related variants. Treat **16 families**, not 80 independent scene tasks, as resampling units. Pair all arms by exact `prompt_id`, and resample the same family indexes for each paired comparison. The estimand is the mean strict-pass difference over these balanced scene families under one generation per scenario.

Define strict pass exactly as `_summary` does:

```python
def strict_pass(row):
    return int(
        row.benchmark_metadata["status"] == "completed"
        and row.valid
        and row.scores["scene_quality"].get("quality_failure") is False
    )
```

Compute family means of paired differences, then a deterministic nonparametric cluster bootstrap with 20,000 replicates and seed `20261015`. For example:

```python
import numpy as np

def paired_family_interval(rows, left, right, *, expected_families=16, per_family=5):
    lookup = {(r.prompt_id, r.backend): r for r in rows}
    assert len(lookup) == len(rows)
    left_rows = [r for r in rows if r.backend == left]
    right_rows = [r for r in rows if r.backend == right]
    assert {r.prompt_id for r in left_rows} == {r.prompt_id for r in right_rows}
    by_family = {}
    for a in left_rows:
        b = lookup[a.prompt_id, right]
        family = a.benchmark_metadata["family"]
        assert family == b.benchmark_metadata["family"]
        by_family.setdefault(family, []).append(strict_pass(a) - strict_pass(b))
    assert len(by_family) == expected_families
    assert all(len(v) == per_family for v in by_family.values())
    differences = np.array([np.mean(by_family[f]) for f in sorted(by_family)])
    rng = np.random.default_rng(20261015)
    indexes = rng.integers(0, len(differences), size=(20000, len(differences)))
    samples = differences[indexes].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"difference": float(differences.mean()), "ci95": [float(lo), float(hi)],
            "families": len(differences), "prompts": len(left_rows),
            "replicates": 20000, "seed": 20261015,
            "method": "paired family-cluster percentile bootstrap"}
```

Report two prespecified contrasts in percentage points:

- **Improvement:** selected fine-tuned model minus `google/gemma-4-26B-A4B-it`. A positive point estimate is observed improvement; a 95% lower bound above zero supports improvement under this resampling analysis.
- **Remaining gap:** `gpt-6.1-sol` minus selected fine-tuned model. A point estimate at most five percentage points reaches the point-estimate target. Claiming the gap is bounded within five points requires the upper confidence limit also be at most five points; otherwise report uncertainty explicitly.

These are approximate intervals over a small, purpose-built family suite, not population guarantees. They do not measure judge calibration error, teacher/judge correlation, or within-prompt generation variability. All-identical family outcomes can yield a degenerate bootstrap interval; report that limitation rather than claiming certainty. Do not choose the interval method or candidate after seeing confirmation results. Preserve per-family and per-prompt paired outcomes for inspection.

## Publication and handoff

Publish actual PNGs, prompt requirements, arm ID, model display ID, immutable artifact URI, source trace, pass/failure reason, generation error, and judge status. Keep model ID and adapter recipe/version in separate columns. Include all-attempt denominators and scored-only metric denominators prominently.

The current native Weave publisher uses model display ID in dataset/evaluation names. Both SFT candidates therefore share a readable object name but receive distinct versioned references; `weave-publication.json` keys them by their unique internal arm IDs and metadata records `model_version`. Link the exact versioned evaluation refs and show artifact URI/epochs so the two candidates are not confused. Do not infer identity from display name alone.

Use `write_loop_report(..., benchmark_dirs=[validation_dir, confirmation_dir])` after both complete, retain each run separately, and add the saved selection receipt plus paired uncertainty artifact to the showcase. Confirm publication coverage is 24 rows for each of four validation evaluations and 80 rows for each of three confirmation evaluations. No live performance claim follows merely from training deployment.
