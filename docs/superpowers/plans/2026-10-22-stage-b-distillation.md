# Stage B: Sol data at scale, Forge SFT sweep, RL, confirmation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Forge-trained Gemma checkpoint whose strict pass rate under the calibrated judge (`bufo-astra-v5`, three samples) is at Sol level on the 80 heldout prompts, or an honest report of the remaining gap, with every number a published Weave evaluation.

**Architecture:** The calibrated judge becomes the acceptance filter for rejection-sampled Sol trajectories (k attempts per train prompt, 128 original + 128 augmented train-family prompts), exported through the existing `export-reviewed` path with the judge recorded as the reviewer. A Forge SFT sweep changes one variable at a time from the iteration-2 recipe and selects on validation by paired bootstrap against the base student. One serverless RL update continues from the selected SFT checkpoint with an Astra-shaped reward gated by an Astra calibration report built from the packet v4 labels. Then one confirmation run on heldout for base, Sol, selected SFT and selected RL, with a judge-independent blinded human review beside the Astra number.

**Tech Stack:** Python 3.12 / uv, openai-agents SDK, `AstraReferenceJudge` (gpt-6-astra, k=3), Weave Evaluations in `wandb-smle/bufo-gen`, Forge fine-tuning API via `ForgeClient`, W&B inference serving, ART serverless RL (`train_rl.py`), resvg in cwsandbox.

**Spec:** `docs/superpowers/specs/2026-10-21-svg-distillation-design.md` (sections "Training loop", "Students", "RL, then confirmation", "Calls made by the author"). Calibration evidence: `docs/distill/calibration-v1.md`, `runs/distill/calibration-v1/calibration-record.json`. Stage A ledger with deferred items: `docs/superpowers/plans/2026-10-21-astra-judge-and-calibration-ledger.md` (final line).

## Global Constraints

- Judge: `bufo-astra-v5`, `samples=3`, `gpt-6-astra`, threshold 0.65, confidence floor 0.6; every judged row in this stage uses exactly this configuration. Any rubric change after Task 5 voids the heldout result.
- Train / validation / heldout stay disjoint by family (`config/distill_prompts.yaml`: 16 / 4 / 16 families). Only `train` rows feed any optimizer or data filter. `validation` selects. `heldout` is judged once, in Task 10, with `--confirmation`; never opened before.
- Acceptance for training data = `strict_pass` ∧ `judge_confidence ≥ 0.6` ∧ `status == "ok"` ∧ generation `status == "completed"`.
- Data target ≥ 1,000 accepted trajectories; k = 4 attempts, raised once to 6 if the target is missed; then stop and report the count.
- Sol level = paired family-cluster bootstrap of student − Sol strict pass on heldout, 95 % lower bound ≥ −10 pp (`statistics.paired_family_interval`, 20,000 replicates, seed 20261015). Selection on validation requires lower bound > 0 over the base student.
- Agent configuration for every arm: `--max-turns 100 --max-tokens 6000`, thinking on, retry loop on (as in Stage A). Nothing changes between selection and confirmation.
- Runtime budgets: 100 turns, 1800 s request timeout, no dollar cap; counts of generations and judge calls are reported per task.
- Existing `runs/**` evidence is read-only; new outputs go under `runs/distill/stage-b/**` and `runs/references/**`. Do not touch Forge job `61b53c05-…`, frozen files or restored runtimes. Do not edit files listed in any run's `configuration.source_hashes` without checking `_check_frozen_scoring_sources` still passes for `runs/distill/arms-v1`.
- No secrets in prompts, artifacts, docs, Weave metadata or scripts; judge_error text carries class name and HTTP status only.
- Blinding: before the user labels any packet, nothing they can read names an item's index with its arm, prompt id or verdict; keys live outside packet directories; the Weave queue is the source of truth for labels and is never hand-edited.
- Shadow git: every `git` command is prefixed with `export GIT_DIR=/private/tmp/claude-502/-Users-aayush-jaiswal-Desktop-cw-bufo-gen/e49cc808-fe4a-4d3f-b40b-1953a1b95b51/scratchpad/shadow.git GIT_WORK_TREE=/Users/aayush.jaiswal/Desktop/cw-bufo-gen`; `git add -f` for `docs/`; trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Full suite (`uv run pytest -q`), `uv run ruff check src scripts`, `uv run ruff format --check src scripts` green at every commit.

## Review Focus

1. **Augmented prompt in a validation or heldout family.** `load_benchmark` must reject a `split: train` prompt whose `family` is not a train family; Task 3's test adds one and expects `ValueError`.
2. **Judge decisions reused after a rubric or sample change.** A decisions JSONL built from `astra-v5-k3` rows must be refused by export if the rows' `scorer_configuration` (minus `calibration_status`) differs from the registry entry named in the file; Task 2's test flips `samples` and expects rejection.
3. **A heldout row judged early.** `score_benchmark` already refuses heldout without `--confirmation`; Task 4's test checks that the heldout reference and arm directories hold no `astra*` directory and that `build_reference_manifest` does not judge.
4. **Checkpoint identity.** Every SFT/RL arm's Weave metadata must carry `forge_job_id`, `forge_dataset_id`, `sft_config` and the artifact version; Task 7's test expects publish to raise when an arm's model is a `wandb-artifact:` without them.
5. **One failed sample in a k=3 row.** The whole row becomes `judge_error` and is retried on re-run (existing behaviour); Task 6's smoke run checks `judge_errors == 0` after the second pass and that no row carries fewer than 3 samples.

---

### Task 1: Calibration registry, semantic retry, relative publish paths

**Files:**
- Create: `src/bufogen/calibration.py`
- Modify: `src/bufogen/scorers/astra_judge.py` (configuration, `_judge` retry), `src/bufogen/benchmark.py` (`_score_astra` cache match, `_astra_metadata`, `_astra_dataset_row`), `scripts/paired_across_runs.py` (config comparison)
- Test: `tests/test_calibration_registry.py`, `tests/test_astra_judge.py`, `tests/test_benchmark_astra.py`

**Interfaces:**
- Produces: `calibration.CALIBRATED: dict[tuple[str, int], str]` mapping `(rubric_version, samples)` → record path, initially `{("bufo-astra-v5", 3): "runs/distill/calibration-v1/calibration-record.json"}`; `calibration.status(rubric_version: str, samples: int) -> str` returning `"calibrated"` or `"uncalibrated"`; `calibration.record(rubric_version, samples) -> dict | None` (loads and verifies the record's `gate.passed`).
- Produces: `astra_judge.config_key(config: dict) -> dict` = config without `calibration_status`; all cache / replay comparisons in `benchmark._score_astra`, `AstraReferenceJudge.score.matches`, and `paired_across_runs` use `config_key` on both sides.
- Produces: `AstraReferenceJudge.configuration()["calibration_status"]` derived from `calibration.status`; everything else unchanged.

- [ ] **Step 1: Write failing tests** in `tests/test_calibration_registry.py`:
  `test_v5_k3_is_calibrated_and_v1_is_not` (status values), `test_record_requires_gate_passed` (tmp record with `gate.passed=False` → `status` returns `"uncalibrated"`), `test_v1_rows_still_replay_against_derived_status` (a saved v1 score with `calibration_status: "uncalibrated"` matches `config_key(configuration())`), `test_v5_k3_rows_replay_after_status_change` (saved v5-k3 score recorded as uncalibrated matches the now-calibrated config through `config_key`).
- [ ] **Step 2: Run** `uv run pytest -q tests/test_calibration_registry.py` → FAIL (module missing).
- [ ] **Step 3: Implement** `calibration.py` and `config_key`; route the three comparison sites through it; derive the status in `configuration()`.
- [ ] **Step 4: Semantic retry (deferred Minor 2):** in `_judge`, a response that parses as JSON but fails `astra_result` validation is retried once like a decode failure (wrap the validation in the attempt loop; keep `MAX_ATTEMPTS = 2`). Test `test_invalid_schema_response_is_retried_once` in `tests/test_astra_judge.py`: first content valid JSON with `prompt_checks: []`, second valid → `status == "ok"`, `create.await_count == 2`.
- [ ] **Step 5: Relative paths in published rows (deferred Minor 11):** `_astra_dataset_row` publishes `generation` with `png_path`, `svg_path`, `input_image_path` relative to `ROOT` (use `references._relative_to_root`). Test in `tests/test_benchmark_astra.py`: published rows contain no absolute path and no `/Users/`.
- [ ] **Step 6: Metadata:** `_astra_metadata` adds `calibration_status` and `calibration_record` (record path or `null`). Extend `REQUIRED_METADATA` in the test.
- [ ] **Step 7: Verify replay with no new judge calls:** `uv run bufogen --stage benchmark-score --output runs/distill/arms-v1 --judge astra --rubric bufo-astra-v5 --judge-samples 3 --references runs/references/distill-v1 --baseline-arm closed_quality_bar` (no `--publish`) exits 0 in under 60 s and `runs/distill/arms-v1/astra-v5-k3/scored.jsonl` is byte-identical before and after (sha256 both). Same for `--output runs/distill/arms-v1` with v1 defaults (`astra/`).
- [ ] **Step 8: Full suite, lint, commit** `feat(judge): calibration registry; semantic retry; relative published paths`.

### Task 2: Judge decisions as training reviews

**Files:**
- Create: `src/bufogen/judge_reviews.py`, `scripts/judge_review_decisions.py`
- Modify: `src/bufogen/curate.py` (`_validate_review` reviewer kinds), `src/bufogen/forge.py` (`reviewed_entries` keeps original evidence), `src/bufogen/cli.py` (`export-reviewed` accepts `--scored-dir`)
- Test: `tests/test_judge_reviews.py`, `tests/test_curate.py` (existing file if present, else new)

**Interfaces:**
- Produces: `judge_reviews.decision_from_row(row: BufoGeneration, *, registry_key: tuple[str, int], source_ref: str, reviewed_at: str) -> dict` returning the `schema_version: 1` review with `reviewer_kind: "astra_judge"`, `reviewer_id: "gpt-6-astra/bufo-astra-v5/k3"`, `decision` by the Global Constraints acceptance rule, `evidence` = judge `reason` (first 1000 chars), `source_generation_id`, `prompt_id`, `prompt_family`, `partition` (= `benchmark_metadata.split`), `source_ref` (scored.jsonl path relative to ROOT), `judge: {strict_pass, judge_confidence, rubric_version, samples, rubric_hash, schema_hash, vs_reference}`.
- Produces: `judge_reviews.write_decisions(scored_dir: Path, out: Path, *, registry_key) -> dict` (counts accepted / rejected / skipped, writes JSONL, refuses rows whose `config_key(scorer_configuration) != config_key(AstraReferenceJudge(rubric_version, samples).configuration())`).
- Consumes: Task 1 `config_key`, `calibration.status` (refuses an uncalibrated key).

- [ ] **Step 1: Check frozen hashes before editing:** `python3 -c "import json;print([k for k in json.load(open('runs/distill/arms-v1/benchmark.json'))['configuration']['source_hashes']])"`; if `src/bufogen/curate.py` or `forge.py` is listed, the reviewer-kind change goes into `judge_reviews.py` as a wrapper that pre-validates and the listed file is left untouched (ledger the ruling).
- [ ] **Step 2: Failing tests:** `test_decision_accepts_strict_pass_with_confidence` (row with strict_pass, confidence 0.9 → accept), `test_decision_rejects_low_confidence_and_fail` (0.5 → reject; strict fail → reject), `test_write_decisions_refuses_other_scorer_config` (samples 1 rows vs key k=3 → `ValueError`), `test_write_decisions_refuses_uncalibrated_key` (`("bufo-astra-v1", 1)` → `ValueError`), `test_export_reviewed_accepts_astra_judge_reviews` (end-to-end through `export_reviewed` with a tmp manifest; exported example carries `training_review.reviewer_kind == "astra_judge"` and the `judge` block), `test_validation_rows_export_only_as_val_split` (existing `reviewed_entries` behaviour, asserted).
- [ ] **Step 3: Run** → FAIL. **Step 4: Implement.** Add `"astra_judge"` to the reviewer kinds in `_validate_review` and `rl_contract` (both sets), keeping the three existing kinds.
- [ ] **Step 5: Script:** `uv run python scripts/judge_review_decisions.py --scored-dir runs/distill/arms-v1/astra-v5-k3 --rubric bufo-astra-v5 --samples 3 --out runs/distill/stage-b/decisions/arms-v1-v5-k3.jsonl` prints `{"accepted": 154, "rejected": 22, "skipped": 0}` (matches `calibration-v1.md`).
- [ ] **Step 6: Export smoke on existing data:** `uv run bufogen --stage export-reviewed --manifest runs/distill/arms-v1/manifest.jsonl --reviews runs/distill/stage-b/decisions/arms-v1-v5-k3.jsonl --output runs/distill/stage-b/export-smoke` yields 154 examples from train+validation, with validation rows marked `partition: validation`; count printed; no upload.
- [ ] **Step 7: Full suite, lint, commit** `feat(data): judge verdicts as training reviews (astra_judge reviewer kind)`.

### Task 3: Prompt augmentation inside train families, generation by prompt ids, merged references

**Files:**
- Modify: `config/distill_prompts.yaml` (+128 prompts, `split: train`, `provenance: stage-b-augment-2026-10`, `augmented_from: <base prompt id>`), `src/bufogen/benchmark.py` (`load_benchmark` provenance validation; `run_benchmark(prompt_ids=...)`), `src/bufogen/cli.py` (`benchmark --prompt-ids-file`), `src/bufogen/references.py` (`merge_reference_manifests`)
- Create: `scripts/augment_prompts.py` (deterministic generator from per-family templates; writes the YAML block and a provenance table `docs/distill/augmented-prompts.md`)
- Test: `tests/test_benchmark.py` (prompt validation), `tests/test_references.py` (merge), `tests/test_augment_prompts.py`

**Interfaces:**
- Produces: 128 new prompt ids `aug-train-<family>-<nn>`, 8 per train family, each with 3–6 exact requirements following the R6 rules (no semicolons inside a requirement, no redundant requirement), themes drawn from the family's existing props/poses with one changed element per prompt.
- Produces: `run_benchmark(..., prompt_ids: list[str] | None = None)` and CLI `--prompt-ids-file` for the `benchmark` stage (distinct known ids; combined with `--split`).
- Produces: `references.merge_reference_manifests(paths: list[Path], out_dir: Path) -> dict` writing `runs/references/distill-v2/references.json` with `version: "bufo-references-v1"`, `rows` = union (duplicate prompt id → `ValueError`), `sources: [{path, source_run_id, sha256}]`, images copied; `load_reference_manifest` unchanged.

- [ ] **Step 1: Failing tests:** `test_augmented_prompt_must_be_in_train_family` (a `split: train` prompt with a validation family → `ValueError` from `load_benchmark`), `test_augmented_prompts_carry_provenance` (every `aug-train-*` has `provenance` and `augmented_from` pointing at an existing train prompt), `test_requirements_follow_r6_rules`, `test_run_benchmark_prompt_ids_filter`, `test_merge_reference_manifests_union_and_duplicate_refusal`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** generator, loader validation, filter, merge.
- [ ] **Step 4: Generate** `uv run python scripts/augment_prompts.py --out config/distill_prompts.yaml --seed 20261025` and `uv run bufogen --stage publish-prompts` if that stage records the prompt set (otherwise skip and note); `load_benchmark` passes; prompt hash recorded in `docs/distill/augmented-prompts.md`.
- [ ] **Step 5: Reference run for augmented prompts (paid: 128 Sol generations, ~25 min at concurrency 4):** `uv run bufogen --stage benchmark --benchmark-config config/distill_prompts.yaml --split train --prompt-ids-file runs/distill/stage-b/aug-train-ids.txt --backends closed_quality_bar --output runs/distill/stage-b/sol-aug-ref --concurrency 4`; then `uv run bufogen --stage build-references --output runs/distill/stage-b/sol-aug-ref --references-out runs/references/distill-aug-v1`; then `merge_reference_manifests([distill-v1, distill-aug-v1], runs/references/distill-v2)`; 304 rows (176 + 128); digest recorded.
- [ ] **Step 6: Full suite, lint, commit** `feat(prompts): 128 augmented train prompts with provenance; references distill-v2`.

### Task 4: Heldout references produced up front, unjudged (deferred Minor 13)

**Files:**
- Modify: `scripts/distill_stage_b.sh` (create; steps `heldout-ref`, `heldout-sol`), `docs/distill/stage-b-progress.md` (create)
- Test: `tests/test_heldout_guard.py`

- [ ] **Step 1: Test:** `test_heldout_dirs_hold_no_astra_output` (walks `runs/distill/stage-b/heldout-*` if present and asserts no `astra*` directory and no `scored.jsonl`), `test_build_references_does_not_judge` (tmp run → manifest has no `scores`).
- [ ] **Step 2: Paid: 160 Sol generations.** `heldout-ref`: `benchmark --split heldout --backends closed_quality_bar --output runs/distill/stage-b/heldout-sol-ref`; `build-references --references-out runs/references/distill-heldout-v1`. `heldout-sol`: `benchmark --split heldout --backends closed_quality_bar --output runs/distill/stage-b/heldout-sol-arm`. Record wall time and error counts in `stage-b-progress.md`. **Nobody opens these PNGs or runs the judge on them until Task 10.**
- [ ] **Step 3: Commit** script and progress doc.

### Task 5: Anti-anchoring probe (Stage A review recommendation)

**Files:**
- Create: `scripts/anchoring_probe.py`, `runs/distill/stage-b/anchoring/` (outputs)
- Test: `tests/test_anchoring_probe.py` (SVG text mutation only; no network)

**Interfaces:**
- Produces: `anchoring_probe.mutate_text(svg: str, old: str, new: str) -> str` (exact `<text>` content replacement, fails if `old` absent or ambiguous); `anchoring_probe.build_near_copies(reference_rows, out_dir) -> list[dict]` for validation prompts with a "reads exactly" requirement: reference SVG with the sign text changed to a different word, rendered through the existing sandbox renderer, paired with the unmodified reference as reference.

- [ ] **Step 1: Tests** for `mutate_text` (success, absent, ambiguous) and for the near-copy manifest shape.
- [ ] **Step 2: Paid: ≈ 12 near-copies × 3 samples = 36 judge calls.** Judge the near-copies with v5 k=3 (direct `AstraReferenceJudge.score` with the reference attached). Expect: the altered requirement `satisfied == False` in ≥ 11 of 12; `vs_reference == "worse"` in ≥ 10 of 12. Different-but-correct is already measured (Sol run 2 vs run 1: 153 equal, 2 better of 176); cite it.
- [ ] **Step 3: Report** `runs/distill/stage-b/anchoring/report.json` and a section in `stage-b-progress.md`. If the near-copy gate fails, stop and report; no training starts.
- [ ] **Step 4: Commit.**

### Task 6: Sol data at scale with judge acceptance and a low-confidence queue

**Files:**
- Modify: `scripts/distill_stage_b.sh` (steps `data-gen <attempt>`, `data-judge <attempt>`, `data-decide`, `data-export`, `lowconf-queue`)
- Create: `scripts/lowconf_queue.py` (reuses `review_queue.create_queue` on rows with `judge_confidence < 0.6`)
- Test: `tests/test_lowconf_queue.py`

- [ ] **Step 1 (paid: 4 × 256 Sol generations ≈ 1,024, ~2 h at concurrency 4 per attempt run sequentially):** for attempt in 1..4: `benchmark --split train --backends closed_quality_bar --output runs/distill/stage-b/sol-data/attempt-<n> --concurrency 4`. Each attempt is its own frozen run so resume rules hold.
- [ ] **Step 2 (paid: 4 × 256 × 3 ≈ 3,072 judge calls):** for each attempt: `benchmark-score --judge astra --rubric bufo-astra-v5 --judge-samples 3 --references runs/references/distill-v2 --publish`. Evaluation names end in `-v5-k3`. Record `judge_errors == 0` after at most one re-run.
- [ ] **Step 3:** `judge_review_decisions.py` per attempt → `runs/distill/stage-b/decisions/attempt-<n>.jsonl`; `export-reviewed` over all four manifests → `runs/distill/stage-b/export/accepted.jsonl`; print counts: attempts, completed, judged, accepted, accepted per family (16 rows), low-confidence.
- [ ] **Step 4:** if accepted < 1,000: attempts 5 and 6 (same commands), once. Then stop and report whatever the count is.
- [ ] **Step 5 (deferred Minor 16):** `lowconf_queue.py` builds a Weave queue of the low-confidence rows (neutral calls, key under `runs/distill/stage-b/lowconf/key/`); the user labels at leisure; their labels are reported, not used for training in this stage.
- [ ] **Step 6:** Section in `stage-b-progress.md` with the funnel table and the Weave links; commit.

### Task 7: Forge dataset, LoRA rank, SFT sweep, checkpoint metadata

**Files:**
- Modify: `src/bufogen/forge.py` (`create_finetune(lora_rank=None)`), `src/bufogen/forge_loop.py` (`submit_training(lora_rank=None)`, drift check includes rank), `src/bufogen/cli.py` (`--lora-rank`, `--tls12`), `src/bufogen/benchmark.py` (`_astra_metadata` adds `forge_job_id`, `forge_dataset_id`, `sft_config`, `checkpoint_version` from the arm config; publish refuses a `wandb-artifact:` arm without them), `config/models.yaml` (arm entries for each sweep checkpoint via `--checkpoint name=artifact` plus a sidecar `runs/distill/stage-b/checkpoints.json` the CLI reads for metadata)
- Test: `tests/test_forge_rank.py`, `tests/test_benchmark_astra.py`

**Interfaces:**
- Produces: `ForgeClient.create_finetune(..., lora_rank: int | None)` placing the rank in `model_config.config` under the field name discovered in Step 1 (recorded as `FORGE_LORA_RANK_FIELD` in `forge.py` with the discovery evidence in a comment and in `stage-b-progress.md`).
- Produces: `checkpoints.json` entries `{arm_name, artifact, forge_job_id, forge_dataset_id, sft_config: {epochs, batch_size, learning_rate, lora_rank, assistant_turns}, adapter_rank_observed}`.

- [ ] **Step 1: Discover the rank field (design open item 1):** `forge-preflight` plus a dry-run `create_finetune` with `lora_rank=8` against the Forge API's documented request schema (`ForgeClient.preflight`, then the finetune endpoint with `dry_run` if offered, else the first real job of the sweep). Record the field name and the adapter config's observed rank from the resulting artifact (as `docs/forge-gap-work/adapter-capacity.md` did). If the API rejects any rank field, the sweep runs rank 1 only and the plan says so.
- [ ] **Step 2: Tests** for the rank parameter plumbing, the drift check, and the metadata requirement (Review Focus 4).
- [ ] **Step 3: Dataset upload (paid-free):** `forge-import` of `runs/distill/stage-b/export/accepted.jsonl` (train split only for training; validation-family rows, if any accepted, go as `val`). Record `forge_dataset_id`.
- [ ] **Step 4: Sweep, one job at a time, one variable at a time from the iteration-2 recipe (lr 1e-4, epochs 3, batch 2, rank 1, `assistant_turns: all`):**
  - J1: recipe as is, new data (baseline).
  - J2: rank 8 (if the field exists).
  - J3: epochs 1 on the better of J1/J2.
  - J4: half the data (random by family-balanced sample, seed 20261026) on the best so far, to measure the data-size effect.
  Each job: `forge-submit` → poll `forge-status` → artifact → `benchmark --split validation --backends open_base --checkpoint <arm>=<artifact> --output runs/distill/stage-b/sweep/<arm>` (paid: 48 generations) → `benchmark-score ... --rubric bufo-astra-v5 --judge-samples 3 --references runs/references/distill-v1 --publish` (paid: 144 judge calls) → `paired_across_runs.py` against `runs/distill/arms-v1-students/astra-v5-k3 open_base` and against `runs/distill/arms-v1/astra-v5-k3 closed_quality_bar`.
- [ ] **Step 5: Selection** per the design: highest validation strict pass; ties by coverage, then fewer errors, then fewer output tokens; selectable only if the paired lower bound over `open_base` is > 0. If none clears, report "no winner" and stop the stage here (Task 9 and 10 run with Sol and base only, so the confirmation still records the state).
- [ ] **Step 6: Secondary students (design "Students"):** one-call `render_svg` tool probe on W&B inference for `google/gemma-4-31B-it`, `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-27B` (`scripts/student_tool_probe.py`, output recorded). Training a secondary is **out of this plan's paid scope** unless the primary sweep reports "no winner"; `create_finetune` keeps its exact-student check until then.
- [ ] **Step 7:** Sweep table in `stage-b-progress.md`; commit after each job's evaluation.

### Task 8: Astra reward and Astra calibration contract for RL

**Files:**
- Modify: `src/bufogen/train_common.py` (`astra_reward(row, *, scorer_configuration) -> float`), `src/bufogen/rl_contract.py` (`build_astra_calibration_report(scored_rows, decisions, *, scorer_configuration)` and `require_astra_calibration`), `src/bufogen/train_rl.py` (reward selection by `--reward astra`), `src/bufogen/cli.py`
- Create: `scripts/astra_calibration_from_packet.py` (packet v4 decisions + key + `astra-v5-k3` rows → report JSON)
- Test: `tests/test_astra_reward.py`, `tests/test_rl_contract_astra.py`

**Interfaces:**
- Produces: `astra_reward` = `similarity × requirement_coverage × mean(anatomy, composition, readability)` when `status == "ok"`, `bufo_present`, `config_key(scorer_configuration)` matches, generation valid and completed; else 0.0.
- Produces: calibration report shape as `build_calibration_report` but observations come from packet decisions: `reviewed_pass = decision.pass`, per-requirement checklist from `decision.missing`; `rl_ready` requires ≥ 1 positive, ≥ 1 negative, zero judge-pass / reviewer-missing, agreement ≥ 0.8, and `scorer_configuration` equal through `config_key`.

- [ ] **Step 1: Tests** (reward arithmetic and zero cases; report from a synthetic packet; `require_astra_calibration` rejects a hand-edited `rl_ready`).
- [ ] **Step 2: Build the report** from `runs/distill/calibration-v1/packet-v4`, `key-v4/unblinding-key.json`, `decisions-user-v4.json` and the `astra-v5-k3` scored rows → `runs/distill/stage-b/rl/astra-calibration.json`; expected `rl_ready: true` with agreement 0.875.
- [ ] **Step 3: Commit.**

### Task 9: One serverless RL update from the selected SFT checkpoint

**Files:**
- Modify: `src/bufogen/train_rl.py` (Astra scorer in the rollout scoring path, `serving_path: art`), `scripts/distill_stage_b.sh` (`rl-update`, `rl-eval`)
- Test: `tests/test_train_rl_astra.py` (fake ART; reward uses `astra_reward`; capture validation unchanged)

- [ ] **Step 1 (paid: 4 rollouts + 4 × 3 judge calls + evaluation 48 generations + 144 judge calls):** `train-rl --reward astra --calibration runs/distill/stage-b/rl/astra-calibration.json --initial-checkpoint <selected artifact> --forge-finetune-id <job> --steps 1 --name bufo-rl-stage-b-1` with 4 on-policy rollouts on one **training** task, lr 5e-6 (ART serverless as in `docs/forge-contact-pilot/rl-next-step.md`). Report attempted / completed / token-valid / scored / trained counts separately.
- [ ] **Step 2:** Evaluate the RL checkpoint on validation (ART serving path) beside the unchanged SFT checkpoint; paired interval RL − SFT. Only if the lower bound > 0 does the RL checkpoint go to confirmation; otherwise confirmation runs with base, Sol and SFT, and the RL result is reported as is.
- [ ] **Step 3:** Commit; section in `stage-b-progress.md`.

### Task 10: Confirmation on heldout, once, with a judge-independent human check

**Files:**
- Modify: `scripts/distill_stage_b.sh` (`confirm`), `scripts/make_review_packet.py` (`--split heldout --confirmation` passthrough to `select_review_rows(confirmation=True)`)
- Create: `docs/distill/stage-b-results.md`

- [ ] **Step 1: Freeze statement** in `stage-b-progress.md`: rubric hash, samples, prompt hash, agent config, artifact versions; nothing changes after this line.
- [ ] **Step 2 (paid: heldout generations 80 × {base, SFT, RL if selected} = 160–240; Sol heldout arm already generated in Task 4; judge 80 × 3–4 arms × 3 samples ≈ 720–960 calls):** `benchmark --split heldout --backends open_base --checkpoint sft=<artifact> [--checkpoint rl=<artifact>] --output runs/distill/stage-b/confirmation`; copy or reference the Task 4 Sol heldout arm; `benchmark-score --judge astra --rubric bufo-astra-v5 --judge-samples 3 --references runs/references/distill-heldout-v1 --baseline-arm closed_quality_bar --confirmation --publish` for each directory; `paired_across_runs.py` for every student arm against Sol (16 families).
- [ ] **Step 3: Judge-independent check:** `make_review_packet.py --runs <confirmation dirs> --split heldout --confirmation --out runs/distill/stage-b/confirmation/packet --key-dir runs/distill/stage-b/confirmation/key --seed 20261027 --scored-dir astra-v5-k3` (40 items mixed across the confirmation arms), Weave queue `bufo review confirmation`, user labels, `review_agreement.py`; report agreement and the judge-pass / reviewer-missing count beside the Astra numbers. No rubric change is allowed in response.
- [ ] **Step 4: Results doc** `docs/distill/stage-b-results.md`: per-arm strict pass on heldout, paired intervals, Sol-level verdict per the definition ("Sol level" / "gap narrowed to X [lo, hi]"), efficiency table (latency p50/p90, output tokens, tool calls, serving path), data funnel, sweep table, RL counts, human check, Weave links, every paid call count. Update `docs/implementation-progress.md`; commit.

### Task 11: Close-out

- [ ] Stage B ledger copied to `docs/superpowers/plans/2026-10-22-stage-b-distillation-ledger.md`; memory updated (project direction); `docs/distill/stage-b-results.md` linked from `docs/implementation-progress.md`; final whole-branch review per the SDD skill; commit.

---

## Paid-call budget (counts, not dollars)

| Task | Sol generations | Student generations | Judge calls (k=3) |
|---|---|---|---|
| 3 augmented references | 128 | | 0 |
| 4 heldout references + Sol arm | 160 | | 0 (until Task 10) |
| 5 anchoring probe | | | ≈ 36 |
| 6 data at k=4 (k=6 if needed) | 1,024 (1,536) | | ≈ 3,072 (4,608) |
| 7 sweep, 4 jobs | | 192 | ≈ 576 |
| 9 RL | | ≈ 52 | ≈ 156 |
| 10 confirmation | | 160–240 | ≈ 720–960 |

Stage A's 300-row judge pass took ≈ 30 min at concurrency 2 with k=1; k=3 rows take ≈ 1.3× as long. Expect Task 6 judging at ≈ 2.5 h per attempt at concurrency 2 (raise `--concurrency` to 4 if the provider tolerates it; record the setting).

## Self-review notes

- Spec coverage: data at scale (Task 6), augmentation inside train families with provenance (Task 3), references for augmented and heldout prompts (Tasks 3, 4), students and rank 8 discovery (Task 7), secondaries gated by probe (Task 7 step 6, training out of scope unless no winner), selection rule (Task 7 step 5), RL with Astra reward and Astra calibration gate (Tasks 8, 9), confirmation once with human check (Task 10), Weave as the record (every scoring step publishes), deferred Stage A items: Minor 2 (Task 1), 11 (Task 1), 12 (Task 7), 13 (Task 4), 16 (Task 6), anti-anchoring (Task 5), calibration registry (Task 1).
- Decisions the executor must not make alone: raising k beyond 6; training a secondary student; any rubric or sample change; opening heldout before Task 10.
