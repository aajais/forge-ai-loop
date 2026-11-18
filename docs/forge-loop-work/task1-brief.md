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

- [ ] Record expected failures in tests for orphan tool results, missing trajectories, assistant raw SVG, preserved repair history, and legacy manifests.
  ```python
  def test_legacy_generation_is_not_fabricated(legacy_row):
      with pytest.raises(ValueError, match="trajectory"):
          export_training_example(legacy_row)
  ```
- [ ] Capture actual SDK run input/output across retries using `result.to_input_list()`, convert Responses function call/output items to portable chat tool-call messages, preserve call IDs and the real final reply, retain system/tool definitions, exclude hidden reasoning and application-only media publication from model actions.
- [ ] Validate complete message/tool associations and compact editable SVG arguments. Reject unsupported multimodal export explicitly rather than dropping attachments. Store provenance and diagnostics without secrets.
- [ ] Replace raw-SVG curation output with the new exporter; require successful reviewed/qualified examples and reject held-out data. Keep old tests compatible only where their old contract was legitimate.
- [ ] Run `.venv/bin/pytest -q tests/test_trajectories.py tests/test_training_contract.py tests/test_pipeline.py tests/test_ui.py` and Ruff for changed files. Record changes and evidence in task report; independent review before completion.

