# Forge Demo Readiness Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. The user approved execution of the five-part remediation described in chat. Do not pause for another plan approval.

**Goal:** Make the existing Bufo Forge loop reproducible, traceable, and demonstrable without claiming unproved model parity or promoting an unevaluated checkpoint.

**Architecture:** Share runtime policy between studio and controlled benchmark defaults, persist agent-root identity and attach saved scorer evidence to its source, and build a comparison from frozen completed validation evidence. Keep service publication separate from local preparation with explicit receipts. Preserve ongoing training and historical frozen runs.

**Tech Stack:** Python, FastAPI, OpenAI Agents, Weave/OTel, W&B, existing browser UI.

**Spec:** User-approved five-step remediation in this conversation: runtime parity; trustworthy observed failures; trace-to-reviewed-dataset curation; model comparison; evaluated checkpoint and rollback lineage.

## Global Constraints
- Exact model IDs; fine-tuned display names use `ft-`, checkpoint identity and tuning method separately.
- No paid training/generation is needed to publish existing evidence. Do not interrupt running jobs or mutate frozen evidence.
- Do not treat unscored rows as passes, hide failures, fabricate feedback, or claim RL/Sol parity without evidence.
- Keep user reviews blinded and do not fill in human annotations.
- No .git directory is present. Preserve original edited files in this task's backup directory rather than creating commits or a worktree.
- External publication and deployment must be verified. Network/auth failures must be recorded, not represented as completion.

## Review Focus
- Explicit benchmark settings must continue to override shared defaults; historical manifests remain loadable.
- Unconfigured or mutable checkpoint placeholders cannot become a demo model accidentally.
- Source feedback must target the correct root and remain idempotent on retry; missing historical root IDs must be explicit.
- Comparison must refuse mismatched prompts/rubrics/runtime and retain missing-image/error rows.
- Publication failures must not suppress successful drawings or overwrite prior successful receipts.

## Execution record

Implemented locally and independently reviewed on 2026-10-24. Final verification:
1,073 Python tests passed; 8 JavaScript tests passed; touched Python sources/tests
and notebook pass Ruff. Notebook passes `marimo check` and script execution.
Chrome rendered the actual portable comparison successfully. Whole-repository Ruff
also found 152 pre-existing findings in other historical tests; those were not
mass-reformatted. There is no Git checkout; originals are preserved in
`docs/forge-demo-work/before`.

Task 1 is complete, including injected-client timeout provenance and graceful
invalid configuration handling. Task 2 source capture and publication are implemented;
native custom feedback is supported but does not populate Agent Scores columns.
Installed SDK aggregation passes the real offline evaluation path, so the live N/A
issue is not claimed fixed. Task 3 is complete locally; source pinning, training
lineage labels, path containment and historical-runtime caveats were independently
reviewed and corrected. Task 4 includes notebook, 911-row lineage export, Registry
targeting repair and an explicitly unpromoted release candidate.

External work remains: W&B publication, source feedback backfill, live Agent Scores
monitor setup, ARIA/Notebook import and rehearsal, Registry registration and routing
verification. MCP authentication was rejected; shell DNS cannot resolve api.wandb.ai;
the actual dashboard publication attempt failed to launch wandb-core. Its receipt
is `runs/forge-demo/comparison/publication.json` with status `failed`, not published.
No new inference/training job or traffic change was performed.

Ruling: keep historical Sol descriptive rather than statistically paired, because
runtime budgets/client behavior differ. A new matched run is needed for parity claims.
Ruling: preserve original custom root feedback rather than inventing monitor refs;
the native Scores product requires a real monitor/trigger workflow.
Ruling: prepare candidate and base-model rollback without changing active routing
because the live promotion/evaluation state could not be verified.

### Task 1: Shared inference policy
**Files:** new runtime policy module/config; models.py, ui.py, agent.py defaults/provenance; runtime/UI tests.
- [ ] Add regression tests for Qwen's evaluated 20,000-token, thinking-enabled policy and explicit overrides.
- [ ] Share policy resolution with benchmark/CLI default resolution while retaining frozen settings and source fingerprints.
- [ ] Expose policy in status/provenance, reject placeholder models, consistently use ft- display names.
- [ ] Run focused runtime, UI, agent, and backend tests.

### Task 2: Source trace feedback
**Files:** telemetry.py, generation.py, observability.py, benchmark.py; telemetry and publication tests. Coordinate agent.py integration with Task 1.
- [ ] Persist native root identifiers and execution metadata with backward-compatible defaults.
- [ ] Mark application renderer provenance and clarify tool description without removing useful rendering spans.
- [ ] Attach saved Astra/scene scores to original agent roots using supported SDK APIs; retain classic feedback and explicit missing-root diagnostics.
- [ ] Add readable eval call names and correct Forge artifact links; investigate N/A aggregate publication through SDK semantics.
- [ ] Run focused tests with idempotency and failure cases.

### Task 3: Reproducible comparison
**Files:** new demo comparison module/script/config, tests; no overlap with Tasks 1/2.
- [ ] Validate a manifest selecting existing complete validation arms and pinned checkpoints.
- [ ] Produce portable HTML/JSON with matched images, exact model names, checkpoint, all-attempt strict pass, uncertainty, latency, error counts, trace/eval links.
- [ ] Provide W&B tables/charts/summary publication with provenance artifact and idempotent receipts.
- [ ] Test evidence compatibility and failures, then generate the actual local comparison without new inference.

### Task 4: Curation and release handoff
**Files:** demo runbook, review/export lineage or registry preparation as justified by current APIs.
- [ ] Establish concrete trace → reviewed row → dataset → trained model → evaluation navigation using existing evidence.
- [ ] Prepare candidate/rollback metadata and a notebook for dataset inspection; never label training rows as human-approved without evidence.
- [ ] Attempt permitted publication through available SDK/tools; record exact failures and ready-to-run recovery commands if blocked.

### Task 5: Integrated verification
- [ ] Independent review of runtime, telemetry and comparison changes.
- [ ] Run affected Python tests, frontend tests and lint; resolve regressions.
- [ ] Verify local comparison and UI in the browser, and live published surfaces when network access permits.
- [ ] Report completed work, links, validation, and any remaining external blockers accurately.
