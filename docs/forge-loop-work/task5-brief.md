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

## Task 5: Forge loop CLI, curation handoff and showcase

**Files:** `src/bufogen/cli.py`, `src/bufogen/models.py` only if necessary for explicit artifact configuration, new `src/bufogen/loop_report.py`, `aria_prompts/03_curate_visual_dataset.md`, new `aria_prompts/04_forge_loop.md`, `README.md`, new `tests/test_forge_cli.py`.

- [ ] Add explicit CLI commands for preflight, benchmark, export, Forge submit/status and report; bind them to real implemented APIs. Validation must happen before paid side effects.
- [ ] Build an artifact-backed loop view linking Run/Observe/Curate/Improve/Evaluate stages to actual traces, queues, datasets, jobs, model artifacts and evaluations. Label not-started/blocked/failed distinctly; no fictional demo numbers.
- [ ] Write ARIA instructions for identifying visual failures in existing and new production traces, approving teacher targets and creating the dataset. Preserve reviewer provenance and heldout separation.
- [ ] Document how to select a real deployed Gemma adapter in the existing UI and reproduce A/B comparisons. Do not automatically replace production defaults.
- [ ] Run CLI/HTTP-focused tests and existing Node conversation tests; independent review.


## Forge proxy integration

User showcase should include actual Run traffic through stable Forge alias, not only managed training labels. Add an explicit opt-in Forge backend/setting with base URL https://proxy.training.coreweave.com/v1 and model bufo-gen (the existing newly-created v1 baseline route); keep current Gemma/Sol model choices intact. Every Chat Completions request needs extra_body metadata with literal keys wandb.entity and wandb.thread_id (current group_id). Use SDK ModelSettings.extra_body if supported. Do not route Responses model through Chat-only proxy. Record requested alias and resolved serving model if available without pretending alias is raw model ID. Do not change existing production defaults or routing automatically. Official contract https://docs.coreweave.com/model-distillation/quickstart and docs/forge-preflight.md. Tests ensure metadata forwarded, stable thread across turns, existing direct backends untouched. Include all edited files in report for review.
