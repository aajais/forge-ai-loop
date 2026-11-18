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

## Task 6: Real pilot, curation and training/evaluation execution

**Files:** persistent evidence under `runs/forge-loop/`; final ledger and README status.

- [ ] Run live preflight and a small controlled Gemma/Sol pilot through real cwsandbox, then Weave scoring. Inspect each rendered image and show failure evidence without promising improvement.
- [ ] Publish the pilot evaluation and review queue, save live URLs, and provide the ARIA handoff. Run the broader baseline benchmark after smoke succeeds.
- [ ] Read actual reviewer decisions. Export only accepted Sol targets, validating tool schemas, token lengths and family splits. If human review is pending, finish all independent work and report that concrete remaining dependency; do not invent 500 accepted examples.
- [ ] If project eligibility and sufficient reviewed data are available, submit a bounded Forge fine-tune, monitor to terminal status, verify deployed inference, and evaluate on identical held-out prompts. Otherwise preserve a resumable workflow with the exact platform response.
- [ ] If reward calibration and checkpoint support are available, execute a bounded Serverless RL run and evaluate its actual checkpoint; otherwise document the specific unmet prerequisite separately from SFT status.
- [ ] Finish full Python and Node checks, lint, independent whole-change review, real agent smoke and an honest final showcase report including coverage, failures and remaining external dependencies.

