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

## Task 2: Controlled benchmark and visual quality evaluation

**Files:** new `config/forge_prompts.yaml`, new `src/bufogen/benchmark.py`, new `src/bufogen/scorers/scene_quality.py`, `src/bufogen/eval.py`, `src/bufogen/observability.py`, new `tests/test_benchmark.py`, new `tests/test_scene_quality.py`.

**Interfaces:** YAML rows contain `id`, `family`, `split`, `prompt`, `requirements`. `load_benchmark(path: Path) -> list[dict]` validates IDs and family separation. Async `run_benchmark(prompts, backends, output_dir, *, limit=None)` persists settings and per-item results; resume only matching configuration. `SceneQuality` is a Weave scorer with validated presence, anatomy, prompt-constraint coverage, composition and readability results, explicit rubric version and uncalibrated status.

- [ ] Test duplicate IDs/family leakage, invalid render retention, failed judge coverage and mismatched resume settings.
  ```python
  def test_family_cannot_cross_splits(tmp_path):
      # Write two different prompts with one family in train and heldout.
      with pytest.raises(ValueError, match="family"):
          load_benchmark(tmp_path / "leaking.yaml")
  ```
- [ ] Define 100 disjoint held-out prompts plus a smaller diverse development/train pool: attached props, action/contact, multiple characters, setting/context, occlusion, small scale, text and conversational edits. Avoid copying heldout families into training variants. Development smoke uses a small explicit subset; never claim a 100-prompt evaluation from that subset.
- [ ] Freeze web search disabled for controlled initial runs (or inject identical precomputed context), prompt hash, model IDs and sample settings. Record generation errors as failures and checkpoint incrementally.
- [ ] Score rendered pixels with explicit prompt requirements and reference identity; labels alone cannot satisfy visible-action constraints. Expose raw submetrics, failures and judge errors; do not call thresholds calibrated.
- [ ] Extend annotation queue criteria for anatomy, composition, readability and acceptance; preserve existing queues. Provide local image comparison with links and raw results.
- [ ] Run focused tests; independent review.


## Implementation integration details

- Task1 persists trajectory and review data. Preserve that schema; you may add optional benchmark metadata to BufoGeneration and explicit per-agent web-search settings if needed, but do not rewrite trajectories/curate.
- Make the pilot train/dev subset start with the user-visible concert and F1 presentation failure families (these are not held-out examples). Include at least 12 train prompts and 100 heldout prompts, with split-disjoint families. These sizes are dataset definitions, not claims that all have run.
- Prior completed human negative annotation: docs/forge-loop-work/existing-completed-feedback.json and runs/bufo-identity-probe/manifest.jsonl. Preserve as calibration evidence, do not claim one point calibrates a scorer.
- run_benchmark should allow bounded concurrency (default 2 or 4) with a shared safe renderer and incremental durable output, so full heldout runs are practical. No process-global environment changes while tasks run. Explicit web disabled per agent instance preferred; identical search policy recorded.
- Score exact provided requirements (no missing/extra checks silently accepted) plus Bufo presence/similarity, anatomy, composition and readability. Judge errors stay explicit, not valid zero or dropped denominators.
- Future CLI can call your public functions; provide signatures and output file schemas in report. Root will run the live pilot after review. Do not launch paid calls yourself.
