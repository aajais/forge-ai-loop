# Task 6 showcase evidence review

Reviewer: `codex/review_loop_runtime`. Independent read-only production review of `task6showcase-report.md`, `task6showcase-diff.txt`, current report code/tests, and `live-results.md` against saved local summaries and receipts. No remote calls or runtime changes.

## Spec: PASS

The four optional receipts are included verbatim under their intended phases. Missing receipts remain null. They do not change phase status, overwrite the explicitly selected RL state, replace computed benchmark metrics, or invent another evaluated arm. Existing HTML escaping and HTTPS link rendering apply to the new receipt fields. The actual four-arm visualization receipt is absent and remains absent/null in the report inputs.

The factual claims in `live-results.md` match the inspected evidence:

| Claim | Local evidence checked |
| --- | --- |
| 300 attempts, 292 PNGs, eight generation failures; passes 1/71/1 | `heldout-baselines/summary.json`, `heldout-baseline-visualization.json`, frozen benchmark model settings |
| 300 published feedback records and three native evaluations of 100 rows | `heldout-baselines/weave-publication.json`; 300 feedback entries, all three evaluation references, no saved publication errors |
| SFT job deployed, exact Gemma base and immutable v1 adapter | `forge-state.json`, frozen heldout model settings, `tuned-smoke/verification.json` |
| 10 train and four validation targets with assistant provenance | `pilot/curation-summary.json`, Forge `entry_counts` |
| Fifteen-image blinded spot-check, passes 0/5 baseline, 0/5 SFT, 3/5 Sol, agreement 15/15 | `heldout-visual-audit/summary.json`, `judge-agreement.json`, `audit-notes.md` |
| Agent Lens counts 404 total, 390 intent sentiment, 280 failure severity | `agent-lens-status.json`, including project-wide scope and recorded timestamp |
| Proxy rendered PNG, native tool-before-text/PNG ordering, extra gateway turns disclosed | Proxy smoke verification, trace verification, and message-order-check receipts |
| RL failed, zero completed recovery updates, no new RL checkpoint | `rl-recovery-failure.json` and `rl-live-checkpoint.json`, whose recorded checkpoint list contains steps 1 and 0 |
| Annotation queue link | `pilot/review-queue.json` matches the documented queue ID |

The document appropriately distinguishes provisional automated scores, assistant reviews, unverified user ARIA curation, successful serving, and actual quality improvement. It does not claim Sol parity or a successful RL arm. The stated 263 full-suite tests refer to the latest completed full regression in the prior task report; this small change has its own focused validation.

## Quality: PASS

No blocking findings. The diff is small and additive. Tests cover exact receipt preservation, selected RL state preservation, actual URL rendering, missing receipts, and prevention of receipt-only completion claims. Existing benchmark validation continues to bind displayed summary metrics to saved rows and frozen configuration.

Independent verification:

```text
.venv/bin/pytest -q tests/test_forge_cli.py
20 passed in 2.82s

.venv/bin/ruff check src/bufogen/loop_report.py tests/test_forge_cli.py
All checks passed!
```

Approved for root to regenerate the showcase. This review verified local evidence and report behavior; it did not re-query remote services, rerun training, or perform browser visual QA.
