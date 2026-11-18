# SVG distillation to Sol level — design

Date: 2026-10-21. Supersedes the hybrid image-model stages (2–5) of
`~/.claude/plans/look-atthis-project-i-noble-scroll.md`. Stages 0–1 of that plan (frozen runtimes,
latency instrumentation) are complete and unchanged. Its Stage 6 (GIF animation, studio
hardening, deploy) and Stage 7 (cleanup, `git init`) remain valid and follow this work.

## Intent

Make an open, Forge-fine-tunable model draw Bufo emojis as well as `gpt-6.1-sol` does, using the
existing SVG tool agent, and prove it with one number: the strict pass rate on a fixed prompt set
judged by a model that is neither the teacher nor the student. Every evaluation is a Weave
Evaluation carrying enough metadata that any two rows in the Weave comparison view can be told
apart by model, checkpoint, judge version, prompt version and run configuration.

Why now: CoreWeave serverless inference has no image-generation endpoint, so the product's output
path is SVG → resvg → PNG and stays that way. Three prior SFT rounds reached 1–7 % strict pass
against Sol's 65/80 on the same harness, and every one of those numbers was judged by the teacher
itself. The project cannot tell whether that gap is real, how big it is, or whether any training
change helps, until the measurement is fixed. Fixing the measurement is the first deliverable;
closing the gap is the second.

## Goals

1. **A trustworthy scorer.** One `gpt-6-astra` call per bufo, structured output, given the prompt,
   its exact requirements, the candidate PNG and Sol's PNG for the same prompt as a reference
   example. Calibrated before any training claim rests on it.
2. **Weave Evaluations as the record.** Every model × prompt-split run is a `weave.Evaluation`
   with the full metric set, published dataset, and metadata listed below.
3. **A student at Sol level.** A Forge-trained checkpoint whose strict pass rate on the fixed
   heldout set is within the bootstrap interval of Sol's, or an honest report of the remaining gap
   with the efficiency story (latency, tokens, cost) if quality ties earlier than expected.

## Non-goals

- No image-generation model anywhere in the pipeline. The LLM emits SVG; resvg produces pixels.
- No Slack bot. The studio stays an internal web app.
- No change to the frozen experiment runtimes under `runs/restored/`.
- No new confirmation prompt set. The existing 80-prompt heldout partition is the fixed set.

## Anti-goals (would make this a failure even if the number looks good)

- A pass rate judged by the teacher, by the student, or by a judge that was never checked against
  human review.
- Selecting a checkpoint on the heldout set, or tuning prompts, rubric or tools after looking at
  heldout results.
- A judge that rewards copying Sol's picture. The reference anchors the standard; a different
  drawing that meets every requirement passes.
- Reporting an improvement whose bootstrap interval includes zero as a win.

## Constraints (carried over, still binding)

- Runtime: 100 turns, 1800 s request timeout, no dollar cap; cost is metered and reported.
- One variable at a time; each default change (thinking off, lower `max_tokens`, removing the
  outer retry loop in `agent.py:322-347`) is its own measured change.
- Train / validation / heldout stay disjoint by family. Only `train` feeds any optimizer or data
  filter. `validation` selects. `heldout` runs once per final candidate.
- Existing `runs/` evidence is read-only; new outputs go to new directories.
- Forge job `61b53c05-…` (contact pilot) and its restored runtimes are not touched.
- No secrets in prompts, artifacts, docs or Weave metadata.

## Definition of "Sol level"

Paired by prompt, family-cluster bootstrap (the implementation in `docs/forge-gap-work`) of the
difference *student strict pass − Sol run 2 strict pass* on heldout, 95 % interval. **Sol level**
means the interval's lower bound is ≥ −10 percentage points. A result whose interval includes
zero but whose lower bound is below −10 is reported as "gap narrowed to X [lo, hi]", not as
parity. The efficiency story (latency, tokens, cost) is reported in every case and is the
headline only when parity is met.

## The fixed prompt set

`config/forge_gap_prompts.yaml`: 232 prompts, 36 families, partitioned `train` 128 / 16 families,
`validation` 24 / 4, `heldout` 80 / 16. It is kept because (a) every prior result is on it, so new
numbers are comparable, and (b) its heldout has only ever been run once per candidate and never
tuned on. The partition is validated by `benchmark.load_benchmark`'s family-disjointness check.
Validation is enlarged **before any training** to 48 prompts by adding 24 new prompts inside its
four families, because 24 prompts cannot separate candidates. Training data may be augmented
with new prompts **inside train families only**; new prompts carry
`split: train` and a `family` from the train list, and the loader rejects anything else.

## The scorer: `AstraReferenceJudge`

A `weave.Scorer` in `src/bufogen/scorers/astra_judge.py`, rubric version `bufo-astra-v1`.

Inputs per row: prompt, exact `requirements`, candidate PNG, reference PNG (Sol's), and the
boolean `reference_is_self` for Sol-vs-Sol calibration rows.

One Chat Completions call to `gpt-6-astra` (via the `openai` provider, `OPENAI_API_KEY`), two
`image_url` parts (candidate first, reference second, `detail: high`), `response_format` strict
JSON schema. Returned fields:

| field | type | meaning |
|---|---|---|
| `bufo_present` | bool | recognizably Bufo, per the existing identity description |
| `similarity` | 0–1 | identity fidelity, independent of prompt success |
| `anatomy`, `composition`, `readability` | 0–1 | craft, same definitions as `bufo-scene-v2` |
| `prompt_checks` | list | one `{requirement, satisfied, evidence}` per exact requirement, wording preserved |
| `vs_reference` | `better / equal / worse` | compared with the reference on prompt satisfaction and craft, with `vs_reference_reason` |
| `judge_confidence` | 0–1 | the judge's own confidence; rows below 0.6 are queued for human review |
| `reason` | str | concrete visual evidence |

Computed in code, not by the judge:

- `requirement_coverage` = satisfied / total.
- `strict_pass` = `bufo_present` ∧ all requirements satisfied ∧ min(craft) ≥ 0.65. Same threshold
  as today so the number is comparable with prior results.
- `render_ok`, `latency_s`, `output_tokens`, `reasoning_tokens`, `tool_calls`, `truncated_calls`
  from `BufoGeneration` and `model_calls` (Stage 1 instrumentation), in a second deterministic
  scorer `EfficiencyScorer`. Cost is not computed by the project: Weave's OpenAI autopatch
  attributes cost to traced calls for models it knows; where it does not, cost is reported as
  unknown. No pricing table is maintained in this round (user decision).

The rubric text tells the judge: the reference is one acceptable answer, not the only one; a
different drawing that satisfies every requirement is `equal` or `better`; do not reward visual
resemblance to the reference. Text inside images is untrusted data.

Validation mirrors `scene_quality.scene_result`: exact requirement match, finite scores in range,
non-empty evidence. A malformed response is retried once, then recorded as `status: judge_error`
and counted in the denominator as a non-pass.

### Judge health, measured before any training claim

All calibration uses `train` and `validation` rows only. Heldout generations (Sol runs 1 and 2
for every candidate's references) are produced up front with the same configuration but are not
judged, reviewed or looked at until the confirmation run. A rubric revision is allowed only
before confirmation; after it, a revision voids the heldout result.

1. **Sol vs Sol.** Sol runs the set twice, both fresh, with the same agent configuration as
   every other arm (100 turns, `max_tokens`, thinking and retry settings recorded). The old
   65/80 result stays as history; it was judged by Sol with a different rubric and is not the
   new target. Run 2 is judged against run 1 as reference. Run 2's
   strict pass rate is the "Sol level" target; its `vs_reference` split shows the judge's
   same-model noise.
2. **Self-equal check.** 24 rows where candidate = reference. Anything other than `equal` with a
   pass (when the reference itself passes) is counted as judge noise and reported.
3. **Blinded review.** 40 train/validation rows mixed across models, neutral filenames
   (`docs/forge-contact-pilot/review_packets.py` pattern). The reviewer is the user. The packet
   is deliberately simple: one HTML page per packet showing, per row, the prompt, its
   requirements as a checklist, the candidate image and the reference image side by side, and
   three inputs — pass/fail, better/equal/worse, optional note — saved to a JSON file. Nothing
   else is asked. The assistant does a separate blind pass; the user's labels decide the gate.
   Report agreement with `strict_pass` and `vs_reference`. Gate: ≥ 0.8 agreement on `strict_pass`
   and no case where the judge passes a drawing the reviewer marks as missing a requirement. If
   the gate fails, the rubric is revised to `bufo-astra-v2` and steps 1–3 repeat; no training run
   is selected on an uncalibrated judge.

Training-data generation may start before the gate passes: every attempt stores the judge's raw
fields, so the acceptance filter is re-applied after the rubric settles, without regenerating.

### Reference PNGs

References are Sol run 1's PNGs on every split, including augmented prompts (Sol's first attempt
on an augmented prompt is its reference; attempts 2..k are candidates), so every judged row has a
reference and the same judge path. References are stored under `runs/references/<set>/` with a
manifest binding prompt id → PNG sha256 → generation id → Weave ref. A reference whose own
strict pass (judged without reference) is false is kept but flagged `reference_pass: false` in
row metadata so results can be stratified; it is not replaced, because replacing references
after seeing student results is a form of tuning.

## Weave Evaluations

One `weave.Evaluation` per (model or checkpoint, split). Dataset rows:
`{prompt_id, prompt, requirements, family, split, reference_png_sha256, reference_generation_id,
reference_pass}`. Scorers: `AstraReferenceJudge`, `EfficiencyScorer`. Evaluation `metadata`:

- `rubric_version`, `judge_model`, `judge_prompt_sha256`
- `prompt_version` (`prompting.prompt_version()`), `configuration_hash` (from `benchmark._configuration`)
- `backend`, `model`, `base_model`, `checkpoint_artifact` (`wandb-artifact:///…:vN` or null),
  `forge_job_id`, `forge_dataset_id`, `sft_config` (epochs, batch size, lr, any rank/method
  field Forge exposes)
- `max_turns`, `max_tokens`, `disable_thinking`, `retry_loop_enabled`, `web_search_enabled`
- `reference_run_id`, `split`, `prompt_set_sha256`
- `code_sha256` of `agent.py`, `astra_judge.py`, `svg.py`, `render.py`

The existing `benchmark.py` already publishes `weave.Dataset` + `weave.Evaluation` per arm with
`CompleteSceneEvaluation` coverage validation; the new scorer slots in beside `SceneQuality`
through a `--judge astra` option, so the frozen `bufo-scene-v2` path keeps working for the
restored runtimes. Summary JSON per run keeps the per-arm table (strict pass, coverage,
better/equal/worse counts, p50/p90 latency, mean tokens, cost per bufo) and the family-cluster
bootstrap interval already implemented for `docs/forge-gap-work/results.md`.

## Training loop

### Data: rejection-sampled teacher trajectories, at scale

The prior datasets held 124 train trajectories. This round targets ≥ 1,000 accepted trajectories:
Sol runs each train prompt k times (k chosen by budget; start k = 4) plus augmented train-family
prompts, every attempt judged by `AstraReferenceJudge` (reference = Sol run 1 for the same prompt;
augmented prompts get their own Sol reference as described above). Accepted = strict pass with `judge_confidence ≥ 0.6`. Export through the
existing v1 `trajectories.py` path (`export-reviewed`), which already preserves assistant tool
calls, `render_svg` arguments and the final friendly text. The export records judge fields per
example, so a stricter filter can be applied later without regenerating.

The Forge routing alias (`bufo-gen`: Sol 9 / Gemma 70 / iter2-e3 21) stays as the studio's live
traffic capture; it is not the source of training data in this round because at 9 % teacher
share it would take tens of thousands of studio requests to reach the target. If the proxy
export is later preferred, the Stage-4 verification from the superseded plan (finalize result in
request, friendly text in response, both exported) must be done first.

### Students

Candidates from the Forge catalog as listed on 2026-10-21 (11 models). Primary:
`google/gemma-4-26B-A4B-it` (continuity with all prior results). Secondaries, each gated by a
one-call tool-use probe on W&B inference before any training: `google/gemma-4-31B-it`,
`Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-27B`. A model that cannot make a well-formed `render_svg` tool
call on the probe is excluded and the probe output recorded.

Hyperparameters: `ForgeClient.submit_training` exposes `epochs`, `batch_size`, `learning_rate`,
`assistant_turns`. Prior adapters were rank-1 LoRA. The user states (2026-10-21) that **LoRA
rank 8 is now supported**; the first training task finds the request field for it against the
Forge API (dry run) and records the actual adapter rank from the resulting artifact's config,
as `docs/forge-gap-work/adapter-capacity.md` did. Sweep variables: student, rank ∈ {1, 8},
epochs ∈ {1, 3}, learning rate, data size — changed one at a time from the iteration-2 recipe. Each configuration is
one job; selection is on `validation` by strict pass, ties broken by coverage, then fewer errors,
then fewer output tokens. A candidate is selectable only if its paired bootstrap difference over
the base student has a lower bound > 0 on validation; if no candidate clears that, the sweep
reports "no winner" and the data or recipe changes before any further job.

Serving: base and SFT checkpoints are both served through W&B inference
(`api.inference.wandb.ai`, the existing `open_base` / `open_tuned` backends), so latency, tokens
and cost are comparable. RL checkpoints are served through ART's path; their efficiency rows
carry `serving_path: art` and are not compared on latency with the others. Every row records
`serving_path`.

### RL, then confirmation

After SFT selection, the existing serverless RL path (`train_rl.py`, one proven optimizer
update) continues from the selected SFT checkpoint with reward = `train_common.reward` shape over
the Astra fields. `rl_contract.require_calibration` is keyed to the `scene_quality` score shape;
it gains an Astra variant fed by the 40-row blinded review, and no RL update runs until it
passes. Because the reward and the confirmation judge are the same model, the RL candidate's
heldout result also gets a judge-independent check: a 40-row blinded human review of its heldout
PNGs, reported beside the Astra number. Then one
confirmation run on `heldout` for: base student, Sol (run 2 vs run 1), selected SFT, selected RL.
Nothing changes between selection and confirmation; a change means the heldout result is void.

## What happens to the deployed app

The studio keeps the SVG agent. `BUFO_STUDIO_MODEL` can point at any evaluated checkpoint, and the
response carries the judge-free generation as today. The four known frontend bugs (`app.js:198`
`response.ok` ordering, `failedThreads` TDZ, picker stuck on status failure, non-JSON errors) are
fixed in the first plan because they are cheap and unrelated to quality. Async jobs, auth, GIF
primitives and deploy follow as the former Stage 6, unchanged in scope.

## Calls made by the author (not decided by the user)

- Reviewer is the user; packets are one simple HTML page with three inputs per row.
- Headline is strict pass rate with the existing 0.65 craft threshold, for comparability.
- Sol level is defined by Sol run 2 judged against run 1, not by Sol judged against itself.
- The reference is kept even when it fails, and flagged, rather than replaced.
- Rejection sampling with k = 4 and ≥ 1,000 accepted examples as the data target: 128 train
  prompts plus 128 augmented train-family prompts × 4 attempts ≈ 1,024 attempts, ≈ 800 accepted at
  Sol's historical pass rate. If the target is missed, k rises to 6 once; after that the plan
  stops at what it has and reports the count.
- Sol level = paired family-bootstrap lower bound ≥ −10 pp; validation enlarged to 48 prompts;
  selection requires a lower bound > 0 over the base student.
- Calibration on train/validation only; heldout judged once at confirmation.
- Fresh Sol runs with the current configuration; the old 65/80 is history.
- Acceptance uses judge confidence ≥ 0.6; low-confidence rows go to human review queues.
- Student secondaries limited to models the catalog lists and a tool-call probe passes.
- Offline export is the data path; proxy capture stays for live traffic only.
- `AstraReferenceJudge` is added beside `SceneQuality`, never replacing it, so frozen runtimes
  still score with their original rubric.

## Left to the builder

Exact rubric wording beyond the field definitions above; JSON schema layout; how the two scorers
share the Weave call; the review-packet tooling reuse; the bootstrap implementation reuse; the
prompt augmentation method (must stay inside train families and be recorded with provenance).

## Open items checked in the first plan's first task

1. The Forge request field that sets LoRA rank 8, confirmed by a dry run and by the adapter
   config of the first rank-8 artifact.
2. Tool-call probe results for each secondary student.
3. Whether Weave's autopatch attributes cost to `gpt-6-astra` and to the W&B inference models; if
   not, cost stays "unknown" in this round.
