# Forge Bufo AI Loop Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement and review each task. Continue through independent work when live product access or human curation blocks a later stage.

**Goal:** Showcase a truthful, repeatable Forge AI Loop: generate Bufo scenes with Gemma and Sol, observe image failures in Weave, curate through ARIA and annotation queues, distill reviewed Sol trajectories into Gemma, improve with Serverless RL where supported, and compare deployed candidates against the unchanged baseline.

**Architecture:** Preserve the existing chat agent, canonical character assets and serverless rendering. Add a durable trajectory export, controlled evaluation harness and documented Forge client, with resumable run state and lineage. Human annotations and platform eligibility are explicit inputs; unsuccessful or blocked stages remain visible rather than being reported as completed training.

**Tech Stack:** Python 3.12, native Agents SDK, Weave, W&B, Forge Model Distillation, cwsandbox, FastAPI, ART Serverless RL when supported.

**Spec:** User requirements in this conversation; concrete visual evidence in `runs/gemma-sol-comparison/index.html`; existing behavior in `docs/implementation-progress.md`.

## Global Constraints

- Teacher: configured Sol reference; student: `google/gemma-4-26B-A4B-it`. Never silently replace Gemma with another training family.
- Use Forge Model Distillation as the primary SFT path, not the older custom GPU sandbox script.
- Preserve rendered PNG outputs even when image quality fails. Never manufacture success, labels, calibrated metrics or trained artifacts.
- Keep render tool calls before the final assistant text and PNG; preserve conversation, model selection, image input and concurrent chats.
- Match prompt version, character assets, context policy and tool budgets across comparison arms. Partition prompt families before generating teacher targets.
- User has authorized paid task execution and no dollar cap. Bound pilots for evidence and scale only after data validation; no production routing changes are implied.
- User performs ARIA curation and has additionally authorized assistant-reviewed pilot targets. Record assistant reviews distinctly with image evidence, reviewer identity and timestamp; do not impersonate human reviews.
- Do not print secrets, commit credentials or contact colleagues. Document platform enablement requirements without sending messages.
- This directory has no Git metadata. Work in place, retain file snapshots/review reports in `docs/forge-loop-work`, and do not initialize or pretend to commit a repository.

## Review Focus

1. Repair trajectories and follow-ups must retain real tool IDs/results, without invented calls or expanded canonical SVG definitions.
2. Missing/negative human review and held-out families must never become accepted SFT targets.
3. Failed generations/judges must remain in denominators, and comparison must reject incompatible run settings.
4. Resuming Forge operations must not create duplicate training jobs or promote undeployed models.
5. RL must capture only student actions, not search/judge models, and must not silently restart from base when claiming SFT continuation.

## Task 1: Persist and export authentic tool trajectories

**Files:** `src/bufogen/generation.py`, `src/bufogen/agent.py`, new `src/bufogen/trajectories.py`, `src/bufogen/curate.py`, `tests/test_training_contract.py`, new `tests/test_trajectories.py`.

**Interfaces:** Persist optional `trajectory: dict` in BufoGeneration with schema version, frozen system instructions, tool schemas, messages and provenance. Implement `export_training_example(row: BufoGeneration) -> dict` returning messages, tools and source identifiers; refuse missing or inconsistent trajectory data. Historical rows remain readable but are not fabricated into trajectories.

- [x] Record expected failures in tests for orphan tool results, missing trajectories, assistant raw SVG, preserved repair history, and legacy manifests.
  ```python
  def test_legacy_generation_is_not_fabricated(legacy_row):
      with pytest.raises(ValueError, match="trajectory"):
          export_training_example(legacy_row)
  ```
- [x] Capture actual SDK run input/output across retries using `result.to_input_list()`, convert Responses function call/output items to portable chat tool-call messages, preserve call IDs and the real final reply, retain system/tool definitions, exclude hidden reasoning and application-only media publication from model actions.
- [x] Validate complete message/tool associations and compact editable SVG arguments. Reject unsupported multimodal export explicitly rather than dropping attachments. Store provenance and diagnostics without secrets.
- [x] Replace raw-SVG curation output with the new exporter; require successful reviewed/qualified examples and reject held-out data. Keep old tests compatible only where their old contract was legitimate.
- [x] Run `.venv/bin/pytest -q tests/test_trajectories.py tests/test_training_contract.py tests/test_pipeline.py tests/test_ui.py` and Ruff for changed files. Record changes and evidence in task report; independent review before completion.

## Task 2: Controlled benchmark and visual quality evaluation

**Files:** new `config/forge_prompts.yaml`, new `src/bufogen/benchmark.py`, new `src/bufogen/scorers/scene_quality.py`, `src/bufogen/eval.py`, `src/bufogen/observability.py`, new `tests/test_benchmark.py`, new `tests/test_scene_quality.py`.

**Interfaces:** YAML rows contain `id`, `family`, `split`, `prompt`, `requirements`. `load_benchmark(path: Path) -> list[dict]` validates IDs and family separation. Async `run_benchmark(prompts, backends, output_dir, *, limit=None)` persists settings and per-item results; resume only matching configuration. `SceneQuality` is a Weave scorer with validated presence, anatomy, prompt-constraint coverage, composition and readability results, explicit rubric version and uncalibrated status.

- [x] Test duplicate IDs/family leakage, invalid render retention, failed judge coverage and mismatched resume settings.
  ```python
  def test_family_cannot_cross_splits(tmp_path):
      # Write two different prompts with one family in train and heldout.
      with pytest.raises(ValueError, match="family"):
          load_benchmark(tmp_path / "leaking.yaml")
  ```
- [x] Define 100 disjoint held-out prompts plus a smaller diverse development/train pool: attached props, action/contact, multiple characters, setting/context, occlusion, small scale, text and conversational edits. Avoid copying heldout families into training variants. Development smoke uses a small explicit subset; never claim a 100-prompt evaluation from that subset.
- [x] Freeze web search disabled for controlled initial runs (or inject identical precomputed context), prompt hash, model IDs and sample settings. Record generation errors as failures and checkpoint incrementally.
- [x] Score rendered pixels with explicit prompt requirements and reference identity; labels alone cannot satisfy visible-action constraints. Expose raw submetrics, failures and judge errors; do not call thresholds calibrated.
- [x] Extend annotation queue criteria for anatomy, composition, readability and acceptance; preserve existing queues. Provide local image comparison with links and raw results.
- [x] Run focused tests; independent review.

## Task 3: Forge client and resumable managed training

**Files:** new `src/bufogen/forge.py`, new `src/bufogen/forge_loop.py`, new `tests/test_forge.py`, `.env.example` and CLI integration later in Task 5.

**Interfaces:** Follow documented API schemas from `docs/forge-preflight.md`. `ForgeClient` accepts explicit entity/project alias, base URL and injected HTTP transport for tests. Workflow state persists project/dataset/relabel/finetune IDs, exact source dataset hash, model, status and deployed artifact. No speculative API endpoints.

- [x] Test sanitized authorization failures, disabled model, malformed dataset, transport interruption and resume without duplicate mutations using `httpx.MockTransport`.
  ```python
  def test_resume_reuses_job(saved_state, fake_client):
      # Existing job ID must be polled, never recreated.
      resume_training(saved_state, fake_client)
      assert fake_client.create_calls == 0
  ```
- [x] Implement read-only project/model eligibility preflight, explicit project setup, reviewed trajectory import and teacher relabel where its tool-call contract is supported. Otherwise import actual locally collected Sol trajectories through the documented dataset API.
- [x] Submit/poll managed SFT, save remote IDs before subsequent work, distinguish transport uncertainty from definite rejection, and return only real deployed model identifiers.
- [x] Preserve Forge/Weave/dataset/model lineage in state and emit actionable unsupported/permission status. Never substitute custom sandbox SFT for a failed Forge request.
- [x] Run focused tests; independent review.

## Task 4: Safe RL reward and SFT continuation contract

**Files:** `src/bufogen/train_rl.py`, `src/bufogen/train_common.py`, new `tests/test_rl_contract.py`, relevant `tests/test_learning.py`.

**Interfaces:** Reward consumes current visual scorer outputs, with explicit rubric/calibration metadata. Training config records `base_model` and optional initial checkpoint separately. Only use documented ART continuation support; unsupported SFT continuation raises an actionable error before creating a job.

- [x] Test missing scorer output, NaN/out-of-range metrics, absent Bufo, generic-label scene, nonstudent captured choices and unsupported initial checkpoints.
- [x] Reward identity and prompt fulfillment jointly with composition/readability, retaining invalid renders as zero reward. No reward from unchecked cached judge scores.
- [x] Isolate external search/judge calls from ART student capture; validate captured provider/model and tool trajectory before training.
- [x] Add verified checkpoint initialization if supported, otherwise keep base-RL and SFT arms explicitly distinct and report continuation blocked. Do not present a new base run as SFT+RL.
- [x] Run focused tests; independent review.

## Task 4b: Calibrate the image judge on the observed pilot failures

Added after real image review found two Qwen false passes and a requirement serialization error. Preserve all original scores and assistant labels. Implement an explicit Sol Responses judge backend and a versioned spatial grounding rubric, validated on all 32 independently reviewed pilot images. Publish the exact configuration and limited pilot agreement; never label this population calibration. Brief: `docs/forge-loop-work/task4b-brief.md`.

- [x] Implement and test backend/request/configuration contract.
- [x] Independent review and fresh calibration report from actual probe responses.
- [x] Freeze scorer before RL and held-out evaluation.

## Task 5: Forge loop CLI, curation handoff and showcase

**Files:** `src/bufogen/cli.py`, `src/bufogen/models.py` only if necessary for explicit artifact configuration, new `src/bufogen/loop_report.py`, `aria_prompts/03_curate_visual_dataset.md`, new `aria_prompts/04_forge_loop.md`, `README.md`, new `tests/test_forge_cli.py`.

- [x] Add explicit CLI commands for preflight, benchmark, export, Forge submit/status and report; bind them to real implemented APIs. Validation must happen before paid side effects.
- [x] Add opt-in Forge proxy traffic through the created `bufo-gen` alias using Chat Completions and `metadata` keys `wandb.entity` and `wandb.thread_id`. Preserve direct Gemma/Sol choices and existing defaults; do not route the Responses teacher through the Chat-only proxy.
- [x] Build an artifact-backed loop view linking Run/Observe/Curate/Improve/Evaluate stages to actual traces, queues, datasets, jobs, model artifacts and evaluations. Label not-started/blocked/failed distinctly; no fictional demo numbers.
- [x] Write ARIA instructions for identifying visual failures in existing and new production traces, approving teacher targets and creating the dataset. Preserve reviewer provenance and heldout separation.
- [x] Document how to select a real deployed Gemma adapter in the existing UI and reproduce A/B comparisons. Do not automatically replace production defaults.
- [x] Run CLI/HTTP-focused tests and existing Node conversation tests; independent review.

## Task 6: Real pilot, curation and training/evaluation execution

**Files:** persistent evidence under `runs/forge-loop/`; final ledger and README status.

- [x] Run live preflight and a small controlled Gemma/Sol pilot through real cwsandbox, then Weave scoring. Inspect each rendered image and show failure evidence without promising improvement.
- [x] Publish the pilot evaluation and review queue, save live URLs, and provide the ARIA handoff. Run the broader baseline benchmark after smoke succeeds.
- [x] Read actual reviewer decisions. Export only accepted Sol targets, validating tool schemas, token lengths and family splits. If human review is pending, finish all independent work and report that concrete remaining dependency; do not invent 500 accepted examples.
- [x] If project eligibility and sufficient reviewed data are available, submit a bounded Forge fine-tune, monitor to terminal status, verify deployed inference, and evaluate on identical held-out prompts. Otherwise preserve a resumable workflow with the exact platform response.
- [ ] If reward calibration and checkpoint support are available, execute a bounded Serverless RL run and evaluate its actual checkpoint; otherwise document the specific unmet prerequisite separately from SFT status.
  - Live attempt submitted but failed during ART/Gemma history alignment; no RL checkpoint. Untouched-capture replay and a complete exact-output-token probe reproduce compatibility failures. See `docs/forge-loop-work/live-results.md` and `runs/forge-loop/rl-tokenization-diagnosis/`. Successful RL and fourth-arm evaluation remain incomplete.
- [x] Finish full Python and Node checks, lint, independent whole-change review, real agent smoke and an honest final showcase report including coverage, failures and remaining external dependencies.

## Execution decisions

- The user explicitly requested planning followed by execution with subagents, so no additional plan-approval pause is needed.
- Use sequential implementation workers with separate review gates; independent read-only product and code research may run concurrently.
- A large teacher dataset is a target for scale, not a reason to fabricate acceptance or spend on hundreds of examples before verifying the training path.
- User approved assistant-reviewed pilot targets during execution. Inspect actual PNGs and record that provenance before export; later human ARIA reviews remain a separate evidence source.
