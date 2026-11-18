# Task 2 independent review

Reviewer: `/root/review_benchmark` (assistant). Review date: 2026-10-15.

Scope: task2 brief, final refreshed non-Git snapshot diff, worker `task2-report.md`, frozen benchmark/scorer/agent/web-policy/observability implementation, and focused test sources. No paid requests or production writes. Existing passing tests were not rerun.

## Resolved findings

1. **P2 — crash recovery can fail on a torn partial checkpoint.** `agent.py` writes `partial.json` directly, while `benchmark.py` resume parses it without recovery handling. A crash during the write leaves the started item permanently unable to resume, rather than recording the promised explicit interrupted failure. Write atomic durable checkpoints and handle corrupt recovery evidence without losing the failure denominator.
2. **P2 — interrupted conversational edits can be scored using their prior-turn image.** Before the final request, only `agent.partial_generation` is cleared; the earlier `partial.json` remains. A crash before the final render recovers that earlier image and changes its prompt ID/request to the final edit. Although the generation-error status prevents an aggregate pass, the scene judge then produces final-request submetrics from the wrong turn. Keep prior evidence distinctly labeled, and require matching final-turn provenance before judging final pixels.
3. **P2 — existing default annotation queues do not receive the added criteria.** `create_review_queue` publishes added specs but reuses an existing `Bufo identity calibration v1` queue without updating its scorer references. The existing live identity queue therefore retains its original criteria. Create a separately named/versioned scene queue or explicitly migrate the schema while preserving prior annotations.
4. **P2 — native evaluation publication can mark incomplete scoring complete.** The installed Weave `Evaluation.get_eval_results` catches per-row exceptions and substitutes an empty score object; `evaluate()` can therefore return without raising when visual replay or scoring fails. `_publish` records completion without checking native score coverage or persisting evaluation-phase errors. For example, unreadable image evidence can produce an explicit local error but a failed native visual replay. Ensure each expected row has its saved scene score in the native evaluation before recording completed publication.

All four findings were resolved and independently rechecked: partial checkpoints use fsync and atomic replacement with damaged-file recovery; prior-turn checkpoints rotate before the final request and `prior_turn_failure` prevents wrong-image judging; scene annotations use a distinct queue; visual replay preserves unreadable-image error data, and native evaluation verifies every saved score before recording completion while persisting failures.

## Positive findings

- Benchmark definitions provide 100 heldout prompts, 12 train prompts, and four additional validation families; the first training families address concert and F1 presentation failures. Family crossing and duplicate IDs are rejected before selection.
- Shared generation settings, model identities, provider temperature differences, source/reference hashes, search policy, and runtime versions are persisted and checked for resume. Each agent owns its disabled-search policy, preserving concurrent UI behavior.
- Generation and scoring results checkpoint incrementally. Terminal failures remain in all-attempt pass denominators. Judge failures expose null metrics and errors rather than fabricated zero judgments.
- SceneQuality validates numeric metrics and the exact supplied checklist. The rubric uses reference identity and candidate pixels, explicitly rejects textual labels as evidence of visible action, and states that thresholds remain uncalibrated.
- Native Weave scorer feedback reuses the saved judgment; local output includes linked images and raw results. Agent trajectory capture and final media ordering remain in place.

## Verification

The worker report records 74 focused/regression tests passing plus Ruff checks; this reviewer inspected the added failure-path tests rather than repeating the suite. Final loaded-code fingerprints and explicit prior-turn-failure denominator reporting were also inspected. One new integration check executed the actual installed Weave evaluation locally with `SceneQuality`, `visual_generation`, a cached missing-image result, and the coverage-validation override. It passed with the scene scorer retained in the summary. Weave was uninitialized and no network/judge call was made.

## Final verdict

**Spec: PASS. Quality: PASS.** No remaining important findings in the frozen Task 2 implementation. The existing evaluation entrypoints remain unchanged; the new benchmark scorer provides the requested controlled evaluation API without altering existing workflows. The implementation is ready for the root-owned bounded live pilot. This verdict covers implementation and offline evidence, not model quality, scorer calibration, or completion of a full heldout run.
