# Stage B progress

## Tasks 3 and 4: prompts and references

- Task 3: 128 augmented train prompts (`aug-train-<family>-NN`, 8 per train family) with Sol references (`runs/distill/stage-b/sol-aug-ref`, 128/128 valid, 0 errors, 1483 s), manifest `runs/references/distill-aug-v1`.
- Task 3 fix: `aug-train-gpu_garden-07` reworded (requirement overlapped validation); 1 regenerated Sol reference (`sol-aug-ref-fix1`, 66 s, 0 errors), manifest `runs/references/distill-aug-fix1`.
- Merged references `runs/references/distill-v2`: 304 rows (176 + 128), `references.json` sha256 `9d0359ba9bcdaeaf232041479e05c9f94fba101d94c0ce1f49c11d7acd1507c9`.
- `config/distill_prompts.yaml` sha256 `61b964bf06c0a58839600b860c59b4259099967d0d491e2710e3042377fa0264`.
- Task 4, heldout drawings produced up front with the same Sol agent configuration as every other arm (`--concurrency 4 --max-turns 100 --max-tokens 6000`), via `scripts/distill_stage_b.sh`:

| step | output | generations | valid | errors | wall |
|---|---|---|---|---|---|
| `heldout-ref` | `runs/distill/stage-b/heldout-sol-ref` | 80 | 80 | 0 | 1090 s |
| `heldout-ref` (build) | `runs/references/distill-heldout-v1` (sha256 `c643b90b30dd5014abc3513ec048da4f53ecf211b924189c3d037756d94dabd0`) | 80 rows | | | 1 s |
| `heldout-sol` | `runs/distill/stage-b/heldout-sol-arm` | 80 | 80 | 0 | 1033 s |

- The heldout drawings are unjudged and unopened. Nobody looks at them or runs the judge on them until the single confirmation run (Task 10). `tests/test_heldout_guard.py` checks that no `astra*` output or `scored.jsonl` appears under `heldout-*` and that building references does not score.

## Task 5: anti-anchoring probe

Near-copies of Sol reference drawings (`runs/distill/sol-run1`) with the exact sign text changed (READY to LATER, GATE 4 to DOCK 7, FRESH to STALE, ICE to COLD, OPEN to SHUT), rendered through the sandbox renderer and judged against the unmodified reference with `bufo-astra-v5`, `samples=3` (`scripts/anchoring_probe.py`; 8 validation plus the first 4 train rows by `prompt_id`; no heldout; 39 judge calls, 0 errors).

| check | result | gate |
|---|---|---|
| altered requirement unsatisfied | 12/12 | at least 11 |
| `vs_reference == "worse"` | 12/12 | at least 10 |
| `strict_pass` | 0/12 | n/a |
| rows skipped / judge errors | 0 / 0 | n/a |

Gate passed: the judge does not reward copying the reference when a requirement is broken. Different-but-correct is already measured: Sol run 2 vs run 1 gave 153 equal and 2 better of 176. Outputs: `runs/distill/stage-b/anchoring/` (`near-copies/`, `manifest.json`, `verdicts.jsonl`, `report.json`).

Fix 1 (review): the first run replaced ICE with HOT, which is the required text of `aug-train-pizza_kitchen-08` ("A chalkboard menu reads exactly HOT"); the earlier claim that no replacement equalled another prompt's required text was not checked and was wrong for that one word. The probe now collects every "reads exactly" text from all splits of `config/distill_prompts.yaml` (requirement text only, no heldout drawings) and raises on any replacement that equals one or the original word. ICE now becomes COLD; only that row was re-rendered and re-judged (3 extra calls, 39 in total). Result unchanged: 12/12 unsatisfied, 12/12 worse, 0 strict_pass, 0 errors, gate passed. The superseded HOT artefacts, verdicts and report are under `runs/distill/stage-b/anchoring/near-copies/superseded/`.

## Task 6: Sol data at scale (attempt 1 of the pipeline proof)

Pipeline: `scripts/distill_stage_b.sh data-gen|data-judge|data-decide <n>`, `data-export`, `lowconf-queue`; funnel and concatenation in `scripts/data_funnel.py`, low-confidence selection in `scripts/lowconf_queue.py` (tests: `tests/test_stage_b_data.py`). `export-reviewed` is single-manifest, so `data-export` exports per attempt and concatenates with a unique `source_generation_id` assertion.

Attempt 1 (`runs/distill/stage-b/sol-data/attempt-1`, Sol `closed_quality_bar`, 256 train prompts):

| stage | result | wall |
|---|---|---|
| generation | 256 completed, 0 errors | 3103 s |
| judge (`bufo-astra-v5`, k=3, `distill-v2` refs) | 256 ok, 0 errors (no re-run needed) | 1537 s |
| decisions | 228 accepted, 28 rejected, 0 skipped | 2 s |
| export | 228 reviewed examples (`runs/distill/stage-b/export/attempt-1`) | 2 s |
| low-confidence (ok, confidence < 0.6) | 0 rows; `lowconf/NONE.txt` written, no queue created | 1 s |

Weave evaluation: `bufo-astra-gpt-6.1-sol-v5-k3`. Accepted per family: archery_range 12, bakery 12, basketball_court 16, concert_stage 16, f1_presentation 16, film_set 15, gpu_garden 11, laundry_room 16, office_coffee 13, ping_pong 11, pizza_kitchen 13, rainy_commute 16, skate_park 16, tennis_court 16, video_call 15, workshop_repair 14. Funnel: `runs/distill/stage-b/export/funnel.json`.

Augmented vs original (Task 3 review sanity check): strict pass 117/128 (91.4%) for `aug-train-*` against 111/128 (86.7%) for the original train prompts. Augmented prompts are not harder for Sol. The most-missed requirements among augmented prompts each failed on only 2 of 128 rows: "Water connects the can spout to the middle seedling" (gpu_garden-04, -05), "A table and central net are visible" (ping_pong), "An arrow lies on the bow aimed toward a visible target" (archery_range). No requirement is systematically failing.

## Task 7 (part 1): Forge rank field, plumbing, student probes

No Forge job was created; no Sol, judge or benchmark call was made.

### LoRA rank field (design open item 1): not established, the API exposes none

`ForgeClient.preflight` passes (project resolves, `google/gemma-4-26B-A4B-it` catalog-eligible; the catalog entry is a bare model name with no config schema). No OpenAPI is served for the distillation API (`/api/distillation/v1/openapi.json` 404; the unversioned paths redirect to HTML). The training API's OpenAPI (`/api/training/openapi.json`) has `SFTTrainingConfig` with `additionalProperties: false` and only `batch_size`, `learning_rate`, `assistant_turns`, `metric_logging`; the words rank/lora appear nowhere in it.

Probe: `POST /finetunes` against a nonexistent dataset id (cannot create a job). The API validates `model_config.config` keys before looking up the dataset: a valid key returns 404 `not_found`, an unknown key returns 422 `Unrecognized key`.

| config key | HTTP |
|---|---|
| epochs, batch_size, assistant_turns (baseline), learning_rate, metric_logging | 404 (accepted by validation) |
| lora_rank, rank, lora_r, r, adapter_rank, lora, lora_config, peft, rank_r, lora_alpha | 422 |
| weight_decay, warmup_ratio, max_seq_length, max_length, seed, lr_scheduler, gradient_accumulation_steps, dropout, bogus_zz (control) | 422 |

Consequence: `FORGE_LORA_RANK_FIELD = "lora_rank"` in `src/bufogen/forge.py` is a provisional guess, marked NOT ACCEPTED. Per the brief the sweep runs rank 1 only (J2 as written is not possible); `--lora-rank` is plumbed so the API's 422 stays the authoritative answer if it is ever tried. A rank name we did not guess could still exist; only a Forge doc or the first real job could show it.

### Plumbing

`create_finetune(lora_rank=None)`, `submit_training(lora_rank=None)` (durable `training_config` and the existing-job drift check include the rank), `--lora-rank`, `--tls12` (`ForgeClient(tls12=True)`, verified TLS 1.2 context from the iteration-2 script), and `benchmark._astra_metadata` adds `forge_job_id`, `forge_dataset_id`, `sft_config`, `checkpoint_version` (None for non-artifact arms). Publishing refuses a `wandb-artifact:` arm lacking them, before any publication. Source: the arm config, else `runs/distill/stage-b/checkpoints.json` (list; entry `{arm_name, artifact, forge_job_id, forge_dataset_id, sft_config: {epochs, batch_size, learning_rate, lora_rank, assistant_turns}, adapter_rank_observed}`, matched on `arm_name` and `artifact`); `checkpoint_version` is the artifact's `:vN` tag. The file ships as `[]`.

### Student tool probes (`scripts/student_tool_probe.py`, `runs/distill/stage-b/student-probe.json`)

One `render_svg` chat request each on W&B inference (thinking disabled as for `open_base`, max_tokens 2500, SDK retries off).

| model | tool call emitted | error / HTTP | latency |
|---|---|---|---|
| google/gemma-4-31B-it | yes (well-formed `render_svg` with `<svg`) | none | 9.4 s |
| Qwen/Qwen3.8-27B | yes (well-formed) | none | 27.3 s |
| Qwen/Qwen3.6-27B | no | NotFoundError / 404 (2 attempts, the single allowed retry) | 0.1 s |

Qwen3.6-27B is not served under that ID on the endpoint; this says nothing about its tool-call ability.

### Part 1 fix round 1 notes

- Hash drift: the `benchmark.py` edit in efe1fc2 changed the live-tree hash while `runs/distill/stage-b/sol-data/attempt-2` was in flight; its `benchmark.json` froze `benchmark.py` at the 6c2d399 hash. A live-tree resume of attempt-2 would therefore fail loudly with the resume configuration mismatch and must be run from that run's `frozen/` copy per project convention. Scoring and publishing are unaffected: `_check_frozen_scoring_sources` compares only `assets/` and `src/bufogen/scorers/`. Attempts 3+ freeze their own hashes. Verified read-only by calling `_check_frozen_scoring_sources` on the frozen configuration of `runs/distill/arms-v1` (no paid stage run).
- Checkpoint metadata now requires a complete `sft_config` (epochs, batch_size, learning_rate, lora_rank, assistant_turns) and an immutable `vN` `checkpoint_version`.

Attempt 1 Weave evaluation: `weave:///wandb-smle/bufo-gen/object/bufo-astra-gpt-6.1-sol-v5-k3:5tmLGShC06OXAYdbf0GWzrETQRZLKAnra27ZynamTL0`.

## Task 9 (Part 1, code only): Astra RL path and task set

- `train_rl.RLTrainer(reward_kind="astra")` scores each rollout with `AstraReferenceJudge(rubric_version="bufo-astra-v5", samples=3)` and `train_common.astra_reward`; the image, inspector and judge run after the ART capture block. `serving_path: "art"` is recorded in `rl-training.json`, every capture record and each rollout's `benchmark_metadata` (astra mode only; scene mode is unchanged).
- Reference guard: `validate_task_references` checks every task's `reference` (`png_path` exists, `png_sha256` matches the file) before `require_rl_runtime`, ART import or model registration.
- `scripts/build_rl_tasks.py` writes `runs/distill/stage-b/rl/tasks-train.jsonl` (+ `tasks-train.metadata.json` digest sidecar) from `config/distill_prompts.yaml` (train split) and `runs/references/distill-v2`.
- Task set built: 256 tasks, all 256 carry a reference (tasks_sha256 `eee6da3806d1...`; 128 non-train prompts excluded, 0 conversational). No paid calls made.
- The paid update (4 rollouts on the first task, one step) is deferred until the SFT checkpoint exists.


## Published records (Weave + W&B)

Every derived record is published by `scripts/publish_stage_b_records.py` (step `records`; also run at the end of `data-export` and `lowconf-queue`) to `wandb-smle/bufo-gen`: a Weave object `bufo-stage-b-<kind>` and a W&B artifact of the same name (type `stage-b-record`, run group `stage-b`). Idempotent by sha256; ledger: `runs/distill/stage-b/publications.json` (full Weave refs and run ids). Heldout and `sol-data` are never published here.

| kind | Weave object (digest prefix) | W&B artifact |
|---|---|---|
| funnel | `bufo-stage-b-funnel` (`VvtU3eWQjmkw...`) | `wandb-smle/bufo-gen/bufo-stage-b-funnel:v0` |
| decisions-attempt-1 | `bufo-stage-b-decisions-attempt-1` (`YLyxHwAia8YA...`) | `wandb-smle/bufo-gen/bufo-stage-b-decisions-attempt-1:v0` |
| decisions-attempt-2 | `bufo-stage-b-decisions-attempt-2` (`1fnQWDXlD28J...`) | `wandb-smle/bufo-gen/bufo-stage-b-decisions-attempt-2:v0` |
| astra-calibration | `bufo-stage-b-astra-calibration` (`Em8K7EuNF1Bb...`) | `wandb-smle/bufo-gen/bufo-stage-b-astra-calibration:v0` |
| anchoring-probe | `bufo-stage-b-anchoring-probe` (`WiFtC2Bfrw9V...`) | `wandb-smle/bufo-gen/bufo-stage-b-anchoring-probe:v0` |
| student-probe | `bufo-stage-b-student-probe` (`0unVyMEQqi5v...`) | `wandb-smle/bufo-gen/bufo-stage-b-student-probe:v0` |
| rl-tasks | `bufo-stage-b-rl-tasks` (`qcZaKXrP16TX...`) | `wandb-smle/bufo-gen/bufo-stage-b-rl-tasks:v0` |
| checkpoints | `bufo-stage-b-checkpoints` (`izsRUnVZWeuh...`) | `wandb-smle/bufo-gen/bufo-stage-b-checkpoints:v0` |
| lowconf | `bufo-stage-b-lowconf` (`rYDcBO9bqSe0...`) | `wandb-smle/bufo-gen/bufo-stage-b-lowconf:v0` |
| prompt-set | `bufo-stage-b-prompt-set` (`PcRFYIStmXe5...`) | `wandb-smle/bufo-gen/bufo-stage-b-prompt-set:v0` |
| prompt-set-doc | `bufo-stage-b-prompt-set-doc` (`8uWbzaXQ7011...`) | `wandb-smle/bufo-gen/bufo-stage-b-prompt-set-doc:v0` |
| progress-doc | `bufo-stage-b-progress-doc` (`MJKymeAHLcZ8...`) | `wandb-smle/bufo-gen/bufo-stage-b-progress-doc:v1` |
| stage-log | `bufo-stage-b-stage-log` (`x4qvS5J9bfTN...`) | `wandb-smle/bufo-gen/bufo-stage-b-stage-log:v0` |

The `progress-doc` entry predates this table; re-running the publisher publishes the updated doc as a new version.

## Task 6 part B: combined data funnel (4 attempts, 2026-10-23 17:45 UTC)

Data collection stopped at 4 attempts by the user's decision (schedule); the 1,000 target is missed and reported as such. Attempts 5–6 were not run.

| attempt | completed | judged ok | judge errors | accepted | rejected | low-confidence |
|---|---|---|---|---|---|---|
| attempt-1 | 256/256 | 256 | 0 | 228 | 28 | 0 |
| attempt-2 | 253/256 | 253 | 0 | 226 | 30 | 0 |
| attempt-3 | 256/256 | 256 | 0 | 229 | 27 | 0 |
| attempt-4 | 256/256 | 256 | 0 | 228 | 28 | 0 |
| **total** | 1021/1024 | 1021 | 0 | **911** | 113 | 0 |

Exported training examples: 911 (`runs/distill/stage-b/export/accepted.jsonl`, all `train`). Low-confidence queue: none needed (0 rows; `lowconf/summary.json`).

Accepted per train family (16 rows):

| family | accepted |
|---|---|
| archery_range | 38 |
| bakery | 47 |
| basketball_court | 64 |
| concert_stage | 64 |
| f1_presentation | 63 |
| film_set | 62 |
| gpu_garden | 44 |
| laundry_room | 63 |
| office_coffee | 54 |
| ping_pong | 45 |
| pizza_kitchen | 58 |
| rainy_commute | 62 |
| skate_park | 64 |
| tennis_court | 64 |
| video_call | 62 |
| workshop_repair | 57 |

Published records: funnel → wandb-smle/bufo-gen/bufo-stage-b-funnel:v1; decisions → wandb-smle/bufo-gen/bufo-stage-b-decisions-attempt-1:v0, wandb-smle/bufo-gen/bufo-stage-b-decisions-attempt-2:v0, wandb-smle/bufo-gen/bufo-stage-b-decisions-attempt-3:v0, wandb-smle/bufo-gen/bufo-stage-b-decisions-attempt-4:v0.

## Task 7 part 2: SFT sweep (rank 1 only; J1 and J3 run concurrently)

Dataset: `runs/distill/stage-b/export/accepted.jsonl`, 911 accepted Sol trajectories (train only; below the 1,000 target, the user stopped at 4 attempts), imported once with `--tls12` (Forge dataset `280e22a6-ae6b-401e-aafe-cc43182eae49`, revision 2, 911 train / 0 val); J1 and J3 share it. Steps: `scripts/distill_stage_b.sh sweep-import|sweep-job|sweep-status|sweep-eval <label>`. Recipe: lr 1e-4, batch 2, `assistant_turns: all`, rank 1 (no rank field is accepted by Forge). Evaluation: validation split (48 prompts), `bufo-astra-v5`, k=3, references `runs/references/distill-v2`; paired intervals with `scripts/paired_across_runs.py` (family-cluster bootstrap, 95%).

| label | Forge job | epochs | adapter rank (observed) | train wall (upper bound) | strict pass | coverage | errors | output tokens (mean) | paired vs open_base (arms-v1-students) | paired vs Sol (arms-v1) | Weave eval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sft_j1 | `b074a332-7117-4bf6-95aa-5c2574b08d52` (artifact `...sft-j1-20261023:v3`) | 3 | 1 | 53 min | 0/48 | 0.413 | 0 | 1392 | 0.000 [0.000, 0.000] | -0.8125 [-0.875, -0.750] | `bufo-astra-ft-google-gemma-4-26B-A4B-it-v5-k3:h825zSZJ...` |
| sft_j3 | `d69b1e4f-ead2-4ba8-b82d-87712dc84c76` (artifact `...sft-j3-20261023:v1`) | 1 | 1 | 16 min | 0/48 (47 scored; 1 generation error `InternalServerError`) | 0.448 | 1 | 1707 | 0.000 [0.000, 0.000] | -0.8125 [-0.875, -0.750] | `bufo-astra-ft-google-gemma-4-26B-A4B-it-v5-k3:pklLLXFU...` |

References for comparison: base in the same run 0/48 (J1 dir) and 1/48 (J3 dir), coverage 0.417/0.432; `arms-v1-students` open_base 0/48, coverage 0.419; Sol 39/48. Wall times are submit-to-first-poll-showing-deployed, so upper bounds. Neither student passes a single validation prompt strictly; the paired interval against base is degenerate (all family differences 0).

Notes:
- `paired_across_runs.py` refused the pairing because the manifest digest of `distill-v2` differs from `distill-v1` (used by arms-v1). It now accepts differing manifests only when every paired prompt id has the identical reference PNG digest in both (test added); otherwise it refuses as before.
- `sft_j1` publish first failed ("Weave evaluation omitted or changed a saved score"): the `EfficiencyScorer` serving path is read from `config/models.yaml` by arm name, and I added the `sft_*` entries while the J1 scoring was running, so 2 saved rows said `unknown` and the recomputation said `wandb_inference`. I recomputed those 2 rows' efficiency facts with the same scorer (no model calls; original kept as `scored.jsonl.pre-efficiency-fix`) and republished. `sft_j3` was scored before the entries existed, so its rows carry `serving_path: unknown`; this is cosmetic and its publication was clean.
- `--learning-rate` was added to the CLI (it had no way to pass the recipe's 1e-4).

Selection (Step 5, controller ruling T7-J4): **no winner**. The rule (highest validation strict pass; ties by coverage, fewer errors, fewer output tokens; selectable only if the paired 95% lower bound over open_base is > 0) is not met: sft_j1 0/48 and sft_j3 0/48, both with lower bound 0.000. J4 (half data) was not run: the rule cannot be met at full data and a half-data row would be uninformative at the floor; reversible. Record: `runs/distill/stage-b/sweep/selection.json` (`scripts/sweep_select.py`, step `sweep-select`). Tasks 9/10 run with Sol and base only.

Threats (adapter application, `runs/distill/stage-b/sweep/distinctness.json`, `scripts/sweep_distinctness.py`, step `sweep-distinctness`; free and local):
- `served_model` in the rows is only the server's echo of the base model name (also true for `iter2_e3` in `arms-v1-students`), so it cannot show that an adapter was applied.
- sft_j1 is distinct from base on output length (-410 tokens, paired permutation p = 0.0012, while every base-vs-base control has p > 0.05: 0.18, 0.26, 0.93), so its 0/48 is a result about a trained model. SVG length and element count do not separate (all p > 0.08).
- sft_j3 cannot be separated from base by any local statistic (+70 tokens, p = 0.63), so its 0/48 is not certified as a trained-model result; it may be the base model.
- Neither student uses any of the top-30 element ids of Sol's training SVGs (share 0.000 for both students and every base arm; Sol validation rows 0.28), i.e. rank-1 SFT transferred no SVG vocabulary.
- Not run: the direct paid check (5 temperature-0 completions per artifact, including a bogus artifact version to confirm that the version is actually resolved). The permission system denied it; no inference request was made for this.

## Task 10: confirmation freeze statement

Frozen before any heldout drawing is judged, reviewed or opened (2026-10-23):

- Judge rubric `bufo-astra-v5`, rubric hash `3e89d3db7ca42c1d6af67d0e38f79c7271f310565dbc6c1b1851b614d7a3b0cf` (schema hash `694e23a8ebcae180c11365e63185a69655757af92476b6f6044c01b632d6cc23`, judge `gpt-6-astra`, from `runs/distill/arms-v1/astra-v5-k3/scorer.json`); samples 3, strict-pass threshold 0.65, confidence floor 0.6.
- Prompt set: `config/distill_prompts.yaml` sha256 `61b964bf06c0a58839600b860c59b4259099967d0d491e2710e3042377fa0264`; heldout split, 80 prompts.
- Agent configuration for every arm: `--concurrency 4 --max-turns 100 --max-tokens 6000`, thinking on retry on.
- Heldout references: `runs/references/distill-heldout-v1`, `references.json` sha256 `c643b90b30dd5014abc3513ec048da4f53ecf211b924189c3d037756d94dabd0` (80 rows, Sol run `heldout-sol-ref`).
- Arms: `open_base` (`google/gemma-4-26B-A4B-it`, generated by `confirm-gen` into `runs/distill/stage-b/confirmation`) and `closed_quality_bar` (Sol, the Task 4 arm `runs/distill/stage-b/heldout-sol-arm`, benchmark.json configuration hash `d6c164afd918bb12531fb01c7d1fee24ae05a0ed12152a30b71c9cc65c465e4e`, comparison fingerprint `296c6be942e31b98e798619cd48fd1600fa62c97b7656b87e620b9b66c9b886e`).
- SFT arms: none (sweep selected no winner, `sweep/selection.json`). RL arm: none (no selected SFT checkpoint).
- Judging command (both directories, no `--baseline-arm`): `benchmark-score --judge astra --rubric bufo-astra-v5 --judge-samples 3 --references runs/references/distill-heldout-v1 --confirmation --publish`; judge_errors must be 0.
- Sol-level definition: 95% lower bound of (student minus Sol) strict-pass difference, family-cluster bootstrap, at least -10 pp.

Nothing changes after this line.

## Task 10: confirmation run (2026-10-23)

- `confirm-gen`: base Gemma, 80 heldout prompts, 80 completed and valid, 0 errors, 396 s (`runs/distill/stage-b/confirmation`). `--confirmation` is a `benchmark-score` flag only; generation has none.
- `confirm-score` (both directories, no `--baseline-arm`): judge_errors 0 and 0 in both; 480 judge calls; 496 s and 477 s. Weave evaluations `bufo-astra-google-gemma-4-26B-A4B-it-v5-k3` and `bufo-astra-gpt-6.1-sol-v5-k3`.
- Strict pass: open_base 1/80, Sol 60/80; paired (base - Sol) -0.7375 [-0.875, -0.5875], not Sol level (`confirmation/paired-open_base-vs-closed_quality_bar.json`).
- `confirm-packet`: 40-item blinded packet (15 base, 25 Sol items), key in `confirmation/key`, Weave queue `bufo review confirmation` (https://wandb.ai/wandb-smle/bufo-gen/weave/annotation-queues/01a16a66-b6cf-78f7-92c5-fb927d54f303). Awaiting user labels.
- Results draft: `docs/distill/stage-b-results.md`.

## Task 14 secondary students (Qwen/Qwen3.8-27B `sft_q3`, Nemotron-3.5-Lightning-30B-A3B `sft_n3`; Forge rank 8)

Same 911-row dataset (`280e22a6-ae6b-401e-aafe-cc43182eae49`), recipe of J3: 1 epoch, batch 2, lr 1e-4, `assistant_turns: all`; agent configuration unchanged (thinking on, 100 turns, 6000 tokens, cap not changed). Probe for Nemotron (not in the Task 7 probe): one `render_svg` request, tool call emitted, 1.6 s (row appended to `student-probe.json`). Evaluation as before (validation, `bufo-astra-v5` k=3, references `distill-v2`, judge_errors 0 in both runs); each student is selected against its own in-run base arm.

| arm | Forge job | adapter rank observed | train wall (upper bound) | strict pass | scored rows | coverage (scored rows) | truncated calls | errors (non-scored) | paired vs own base | paired vs Gemma base | paired vs Sol |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen_base | n/a | n/a | n/a | 0/48 | 1 | 1.00 | 94 | 47 | n/a | 0.000 [0.000, 0.000] (diff vs open_base 0/0) | n/a |
| sft_q3 | `da6bd91a-d7b8-478c-ad00-f6f540925537` (`...sft-q3-20261023:v1`) | 8 | 30 min | 2/48 | 2 | 1.00 | 90 | 46 (43 "No drawing was rendered", 3 BadRequestError) | +0.042 [0.000, 0.083] | +0.042 [0.000, 0.083] | -0.771 [-0.833, -0.708] |
| nemotron_base | n/a | n/a | n/a | 0/48 | 42 | 0.18 | 18 | 6 | n/a | n/a | n/a |
| sft_n3 | `a16e9b62-80b1-42f0-a1d6-57cebd862c8d` (`...sft-n3-20261023:v1`) | 8 | 11 min | 0/48 | 6 | 0.17 | 81 | 42 (40 "No drawing was rendered", 2 BadRequestError) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -0.8125 [-0.875, -0.750] |

Reading: Qwen3.8-27B with thinking on almost never finishes a drawing inside the 6000-token cap (base 1/48 rendered, student 2/48; 90+ truncated calls), so the comparison is dominated by the cap, not by drawing quality; the only 2 rendered student drawings both strict-passed. Nemotron's base renders 42/48 but its student renders 6/48 (output tokens +6482 versus base, distinct, p < 0.001): training lengthened its output until the cap truncated it. Distinctness: sft_q3 not separable from qwen_base on output tokens (-205, p 0.62; it is below the cap-dominated noise), sft_n3 distinct (+6482).

Selection (all four candidates, each against its own base arm): **no winner**. No candidate has a paired 95% lower bound over its base arm above 0 (sft_q3's interval is [0.000, 0.083]). `selection.json` and `distinctness.json` regenerated. Paid use: 2 Forge jobs, 192 generations, 576 judge calls, 1 probe request, no retries.

## Task 16 — first RL block (sft_q3 → rl_q3)

Config (Ruling T16-config): thinking on, `--max-tokens 20000`, `--max-turns 100`, `BUFO_INFERENCE_TIMEOUT_S=900`, judge `astra` / `bufo-astra-v5` / 3 samples / `runs/references/distill-v2` / `--publish` (threshold 0.65, floor 0.6, gpt-6-astra). RL: OpenPipe ART on CoreWeave Forge, astra reward, 5 steps x 4 rollouts, temperature 1.0, learning rate 5e-6, continuing `wandb-artifact:///wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v1` (Forge fine-tune `da6bd91a-d7b8-478c-ad00-f6f540925537`). Code fixes first (commit 7ec66f6): rollout client timeout from `inference_timeout_s()`, `sft_arm_label` base check, strict `_checkpoint_template`, `benchmark` with no `--backends` runs only the given checkpoints, `rl-eval` compares RL against the unchanged SFT checkpoint.

Training tasks: the brief's `DATASET_REF` (the SFT dataset artifact) is rejected by `load_environment_tasks` ("RL artifact must contain only single-turn train tasks"), before any paid call. The 256-task `tasks-train.jsonl` was pushed unchanged as `wandb-smle/bufo-gen/bufo-stage-b-rl-train-tasks:v0` (with its digest sidecar as `metadata.json`) and used instead. Each step uses one training prompt for its 4 rollouts.

| step | prompt | attempted / completed / token-valid / scored / trained | reward mean | rollout latency p50 (s) |
|---|---|---|---|---|
| 1 | gap-train-concert_stage-base | 4/4/4/4/4 | 0.783 | 670.7 |
| 2 | gap-train-concert_stage-fullbody | 4/4/4/4/4 | 0.790 | 454.5 |
| 3 | gap-train-concert_stage-clothing | 4/4/4/4/4 | 0.786 | 430.5 |
| 4 | gap-train-concert_stage-observer | 4/4/4/4/4 | 0.713 | 337.9 |
| 5 | gap-train-concert_stage-sign | 4/4/4/4/4 | 0.604 | 511.4 |

Per-step rewards are on different prompts, so the decline from 0.79 to 0.60 is not a learning curve. RL artifact: `wandb-artifact:///wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v6` (ART step 6, training job `694533ab-5d82-4547-913a-437c38a99b3b`). The ART registry promotion step reported `CommError`; the checkpoint artifact exists and was served. Record: `runs/distill/stage-b/rl/rl-q3-training.json`.

Evaluation (validation, 48 prompts, one run containing `rl_q3` and `sft_q3`; `judge_errors` 0 for both arms):

| arm | strict passes /48 | scored | requirement coverage | latency p50 / p90 (s) | output / reasoning tokens (mean) | truncated calls |
|---|---|---|---|---|---|---|
| rl_q3 | 30 (0.625) | 45 | 0.892 | 454.8 / 706.7 | 25775 / 21895 | 14 |
| sft_q3 (this run) | 35 (0.729) | 46 | 0.932 | 568.8 / 1295.2 | 26661 / 23730 | 10 |

Paired intervals (family-cluster bootstrap, 20,000 replicates, seed 20261015; per-family differences are airport_marshaller / florist_arranging / ice_cream_cart / locksmith):

| comparison | strict | difference [95% CI] | families |
|---|---|---|---|
| rl_q3 − sft_q3 (same run) | 30 vs 35 | −0.104 [−0.292, +0.146] | −0.167 / −0.167 / −0.333 / +0.250 |
| rl_q3 − qwen_base (cap20k diagnostic) | 30 vs 17 | +0.271 [+0.125, +0.417] | +0.167 / +0.333 / +0.083 / +0.500 |
| rl_q3 − Sol (`closed_quality_bar`, arms-v1) | 30 vs 39 | −0.188 [−0.333, −0.083] | −0.083 / −0.417 / −0.167 / −0.083 |
| sft_q3 (this run) − sft_q3 (cap20k diagnostic) | 35 vs 30 | +0.104 [−0.021, +0.208] | +0.083 / +0.167 / +0.250 / −0.083 |

RL − SFT is −0.104 [−0.292, +0.146]: the interval contains 0, so the result is inconclusive (no gain and no regression established). The unchanged SFT checkpoint moved by +5 prompts between runs (35/48 here, 30/48 in the diagnostic), the same size as the RL gap and consistent with sampling variation; the sft_q1 evaluation was running on the same endpoint at the same time, so run conditions also differed. Sol level (lower bound of student − Sol ≥ −0.10): not met for `rl_q3` (lower bound −0.333).

Block 1 task order: all five steps used `concert_stage` prompts (the first family in the task file, file order), so block 1 trained on one of 16 train families. `train_rl.py` now orders tasks round-robin across families (fix round 1) so later blocks touch different families. The per-step summary is kept in `runs/distill/stage-b/rl/rl-q3-steps.json`; later runs write `rl-steps.json` in their own output folder.

Lineage: ART model `0bd80ce2-968d-46e2-8cf5-6f474cb2aaad`; tasks artifact `bufo-stage-b-rl-train-tasks:v0`; five training jobs `930f54f6-5750-4d9e-b9bc-c0054cbd7335`, `f325081d-d10b-48e8-9b00-51e1588e3072`, `eba4dafc-3663-49bc-a359-c292149bb941`, `cc29fcd9-de4a-49f2-80f6-105457fc8eea`, `694533ab-5d82-4547-913a-437c38a99b3b` (in step order, by first appearance in the update log) (the `forge_job_id` in checkpoints.json is the final-step job; `forge_dataset_id` there is the SFT dataset, as the brief dictated). Adapter rank read from the :v6 artifact: 8.

Weave naming: block 1's evaluation was published under the SFT name `bufo-astra-ft-Qwen-Qwen3.8-27B-v5-k3` for both arms, so the RL evaluation appears as another version of the SFT one. Fix round 1 gives RL arms their own label (`rl-<base>`, in `config/models.yaml` and in the presentation code), effective for later evaluations.

The replication pairing could not run through `scripts/paired_across_runs.py` as invoked (it refuses two arms with the same name); it was computed with the script's own `paired_across` function with the right-hand arm relabeled `sft_q3_prior` in memory (same data, same method).

Paid-call tally: RL 20 rollouts (20 generations, 60 judge calls; no retries); evaluation 96 generations and 288 judge calls. Nothing else paid. The W&B dataset push of the task artifact was free.


## Task 17 — RL block 2 (16 steps × 8 rollouts)

Design (Ruling T17-design, T17-resume, T17-env). Block 2 continues the registered ART model `bufo-stageb-sft-q3-20261023` (model id `0bd80ce2-968d-46e2-8cf5-6f474cb2aaad`) from its current step, with the same astra reward, judge, prompts, agent configuration (thinking on, `--max-tokens 20000`, `--max-turns 100`, `BUFO_INFERENCE_TIMEOUT_S=900`), learning rate 5e-6 and temperature 1.0 as block 1. The one variable that changes is the amount of training: 16 optimizer steps, each on 2 tasks × 4 rollouts (two `TrajectoryGroup`s of 4), rollouts concurrent (semaphore 8), tasks in round-robin-by-family order (picks 1–32: every one of the 16 families twice, 16 different tasks per family pass). Step convention (used everywhere below, in `checkpoints.json` and in the report): a block is described by the registered model's step before and after it, and by the steps its updates produced. Block 1 went from step 1 (the SFT checkpoint) to step 6 (`:v6`), producing steps 2–6 in five updates; block 2 went from step 6 to step 22 (`:v22`), producing steps 7–22 in sixteen updates. In `rl-steps.json` the field `step` is the pre-update step (6..21 for block 2, the checkpoint that generated the rollouts); the update produces `step + 1`, and the tables here use the produced step. The file carries a `step_convention` note saying so. The brief's "step 5" was the controller's miscount of block 1.

Resume: the registered model reported start step 6 (artifact `:v6`, the block 1 artifact recorded in `checkpoints.json`, with the SFT `:v1` still the step-1 ancestor), checked by the new `train-rl --expect-start-step 6`. Final artifact printed by the run: `wandb-artifact:///wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v22` (ART step 22, rank 8 read from its `adapter_config.json`). The ART Registry promotion again reported `CommError` (class only); the checkpoint exists and served.

Two fixes before the successful run. (1) A first launch aborted in its first step with `ValueError: Could not load tokenizer for 'Qwen/Qwen3.8-27B'` from `trajectory.tokenize()`: `transformers` was missing from the `rl` extra, and rollouts whose Dynamo history was re-rendered need the tokenizer. `transformers==5.19.0` is now pinned in the `rl` extra and the lock. The aborted attempt is kept in `runs/distill/stage-b/rl/block2-aborted-tokenizer` (no update ran; the model stayed at step 6). (2) A failed rollout used to cancel its siblings; it is now recorded (exception class only) and the rest continue; a group trains only with at least 2 valid trajectories and a step aborts only if every group falls below that.

Per step (task pair differs every step, so this is not a learning curve; counts are attempted/completed/token-valid/scored/trained, "step" is the ART step the update produced):

| produced step | task ids (family) | counts | reward mean | rollout latency median (s) | training job |
|---|---|---|---|---|---|
| 7 | gap-train-concert_stage-base (concert_stage); gap-train-f1_presentation-base (f1_presentation) | 8/8/8/8/8 | 0.776 | 332.2 | `5c462567-6135-43a7-a978-eb55b717f5c3` |
| 8 | gap-train-gpu_garden-base (gpu_garden); gap-train-office_coffee-base (office_coffee) | 8/8/8/8/8 | 0.813 | 442.1 | `b7cf5f23-c40e-4581-af1e-7e6f8a4cb19b` |
| 9 | gap-train-skate_park-base (skate_park); gap-train-pizza_kitchen-base (pizza_kitchen) | 8/5/8/5/8 | 0.539 | 561.1 | `565e121d-3c78-4671-947f-a95755a10c94` |
| 10 | gap-train-basketball_court-base (basketball_court); gap-train-workshop_repair-base (workshop_repair) | 8/5/8/5/8 | 0.551 | 326.1 | `ea46b720-2c4d-4252-bfc6-e8765d7d6e38` |
| 11 | gap-train-rainy_commute-base (rainy_commute); gap-train-video_call-base (video_call) | 8/8/8/8/8 | 0.862 | 1122.9 | `7d228415-b04f-4eff-878f-c6bdac91ec33` |
| 12 | gap-train-film_set-base (film_set); gap-train-ping_pong-base (ping_pong) | 8/7/7/7/7 | 0.852 | 207.3 | `975993f5-4f7b-452b-ac53-424e407f93da` |
| 13 | gap-train-archery_range-base (archery_range); gap-train-laundry_room-base (laundry_room) | 8/8/8/8/8 | 0.797 | 272.9 | `377be113-19b3-40f8-9822-fd8aa6d858a3` |
| 14 | gap-train-bakery-base (bakery); gap-train-tennis_court-base (tennis_court) | 8/8/8/8/8 | 0.705 | 443.3 | `1d604f84-54be-454b-99f0-e791df4b3cb7` |
| 15 | gap-train-concert_stage-fullbody (concert_stage); gap-train-f1_presentation-fullbody (f1_presentation) | 8/8/8/8/8 | 0.753 | 376.3 | `0c6abef8-4a36-49c0-9920-692bc98bc070` |
| 16 | gap-train-gpu_garden-fullbody (gpu_garden); gap-train-office_coffee-fullbody (office_coffee) | 8/8/8/8/8 | 0.787 | 564.9 | `719840d0-68fb-41ab-a90d-1fc2dee9d589` |
| 17 | gap-train-skate_park-fullbody (skate_park); gap-train-pizza_kitchen-fullbody (pizza_kitchen) | 8/8/8/8/8 | 0.698 | 233.4 | `9d3f2a33-67bc-4a04-b18a-c1dc2c51cf2a` |
| 18 | gap-train-basketball_court-fullbody (basketball_court); gap-train-workshop_repair-fullbody (workshop_repair) | 8/8/8/8/8 | 0.816 | 249.2 | `85654f09-c593-4b02-a7e3-4b3fd1de5b02` |
| 19 | gap-train-rainy_commute-fullbody (rainy_commute); gap-train-video_call-fullbody (video_call) | 8/8/8/8/8 | 0.768 | 189.1 | `26657eaa-0167-4c88-98a4-1169e0dac520` |
| 20 | gap-train-film_set-fullbody (film_set); gap-train-ping_pong-fullbody (ping_pong) | 8/8/8/8/8 | 0.835 | 225.1 | `ce18311e-3eac-4cc4-86c9-95f999a7ac55` |
| 21 | gap-train-archery_range-fullbody (archery_range); gap-train-laundry_room-fullbody (laundry_room) | 8/8/8/8/8 | 0.603 | 198.9 | `07606cc1-11e2-40e0-ac81-d8fa2b3dd5d6` |
| 22 | gap-train-bakery-fullbody (bakery); gap-train-tennis_court-fullbody (tennis_court) | 8/8/8/8/8 | 0.788 | 226.7 | `d06a4af4-0f1b-4cf6-9121-620e693694fe` |

Counts below 8 are real events, not drops: steps 9 (pizza_kitchen) and 10 (workshop_repair) each had 3 rollouts with no render (completed 5/8). In all six the first model call spent the full 20000-token cap on reasoning (`finish_reason: length`) and `render_svg` was never called, so they are model/cap failures, judged `missing_image` (never sent to the judge), and they were trained at reward 0 next to one valid sibling in the second group. They are not sandbox failures. The three step-10 rollouts' second calls ended with `finish_reason: None` after 6.7k–13.3k tokens within a few seconds of each other (about 384–389 s), which could be a shared stream interruption; there is no evidence either way. Separately, a `SandboxError` ("the sandbox is not running") from `render_svg` appeared in 7 valid, completed rollouts (steps 9, 11, 13, 14, 16, 18, 21) and each recovered on the agent's render retry at no cost in reward. The trainer now records `infra_tool_errors` and a `no_render` breakdown (infra / cap_truncated / other) per step, so unrecovered infrastructure zeros can be told from model zeros; for block 2 these were backfilled from `manifest.jsonl` (infra 0, cap_truncated 6, other 0). Step 12 had one rollout fail with `BadRequestError` (HTTP 400), so that step trained 7 rollouts (one group of 3); a 524 `InternalServerError` on one rollout of step 14 was retried by the rollout retry and succeeded. Block 2's tasks overlap block 1's: the round-robin picks reused two block-1 tasks (`gap-train-concert_stage-base` at produced step 7 and `gap-train-concert_stage-fullbody` at produced step 15), as the brief's picks 1–32 dictate. Mean of step rewards 7–14: 0.737; 15–22: 0.756. These are different task pairs (the second pass uses the second task of each family pair), so the difference carries no evidence of learning.

Evaluation (validation, 48 prompts, one run containing `rl_q3` and `sft_q3`; `judge_errors` 0 for both arms; output `runs/distill/stage-b/rl/rl_q3-block2`):

| arm | strict passes /48 | scored | requirement coverage | latency p50 / p90 (s) | output / reasoning tokens (mean) | truncated calls |
|---|---|---|---|---|---|---|
| rl_q3 (block 2, :v22) | 26 (0.542) | 48 | 0.829 | 189.2 / 280.5 | 18684 / 15698 | 3 |
| sft_q3 (this run) | 30 (0.625) | 47 | 0.912 | 230.2 / 520.6 | 28728 / 24264 | 12 |

(One sft_q3 row was an `other_failure`, not a judge error.) Latencies are far lower than in block 1's evaluation for both arms (SFT p50 230 s here vs 569 s), so wall-clock latency varies with serving conditions between runs; token counts are the steadier comparison. The RL arm emits about a third fewer output tokens and truncates less than SFT.

Paired intervals (family-cluster bootstrap, 20,000 replicates, seed 20261015; families airport_marshaller / florist_arranging / ice_cream_cart / locksmith):

| comparison | strict | difference [95% CI] | families |
|---|---|---|---|
| rl_q3 (block 2) − sft_q3 (same run) | 26 vs 30 | −0.083 [−0.208, +0.083] | −0.167 / −0.250 / +0.167 / −0.083 |
| rl_q3 (block 2) − qwen_base (cap20k diagnostic) | 26 vs 17 | +0.187 [+0.167, +0.229] | +0.167 / +0.250 / +0.167 / +0.167 |
| rl_q3 (block 2) − Sol (`closed_quality_bar`, arms-v1) | 26 vs 39 | −0.271 [−0.458, −0.083] | −0.083 / −0.500 / −0.083 / −0.417 |
| rl_q3 block 2 − rl_q3 block 1 (relabelled, `scripts/paired_relabel.py`) | 26 vs 30 | −0.083 [−0.250, +0.042] | 0.000 / −0.083 / +0.083 / −0.333 |
| sft_q3 (this run) − sft_q3 (block 1's run), third replication | 30 vs 35 | −0.104 [−0.312, 0.000] | 0.000 / 0.000 / −0.417 / 0.000 |

Sol-level test (lower bound ≥ −0.10): not met (lower bound −0.458).

Verdict RL − SFT: −0.083 [−0.208, +0.083]. The upper bound is above 0 and the lower bound below 0, so the result is inconclusive: neither an improvement (lower bound > 0) nor a regression (upper bound < 0) is established. The point estimate is again slightly negative for RL, as in block 1; the unchanged SFT checkpoint scored 30, 35 and 30 across its three evaluations (noise of about ±5 prompts), as large as the RL gap. Sixteen further optimizer steps did not produce a measurable gain over SFT at this evaluation size.

Lineage and files: `checkpoints.json` `rl_q3` now records `rl_config.blocks` (block 1: 5 steps, 1 task, 4 rollouts, `:v6`; block 2: 16 steps, 2 tasks, 4 rollouts, `:v22`) and all 21 training job ids. Run outputs: `runs/distill/stage-b/rl/block2` (`rl-steps.json`, `rl-training.json`), evaluation `runs/distill/stage-b/rl/rl_q3-block2`.

Paid-call tally: aborted first attempt 9 rollouts (8 plus one retry attempt) and 15 judge calls; block 2 128 rollouts (129 generations including one retried 524) and 363 judge calls (121 judged rollouts × 3 samples; of the 127 rows in `manifest.jsonl`, 6 were `missing_image` and never judged; the HTTP-400 rollout produced no row; source: manifest count and judge status, not provider billing); evaluation 96 generations and 288 judge calls. Nothing else paid.
