# Final showcase evidence receipts

Scoped changes only: `src/bufogen/loop_report.py` and `tests/test_forge_cli.py`. Original snapshots under `task6showcase-before/`. Frozen generation/scoring configuration, routes, training states and evidence receipts are unchanged.

The report now reads four optional local receipts verbatim:

- Observe: `agent-lens-status.json` as `agent_lens_status`.
- Improve: `rl-recovery-failure.json` as separate `rl_recovery_failure`, leaving the explicitly selected `rl_state` untouched.
- Evaluate: `heldout-baseline-visualization.json` and `heldout-visualization.json` as distinct optional evidence.

Missing files remain JSON null. Receipts do not determine or override phase status, imply another completed model arm, replace benchmark metrics or claim ARIA verification. Existing recursive HTTPS-link rendering makes actual receipt URLs clickable in the corresponding phase card. The four-arm receipt is currently absent, so no four-arm link/result is fabricated.

Validation: **20 focused CLI/report tests passed**; changed-file Ruff passed. Tests verify exact saved values, selected RL state preservation, no status inference from receipts alone, actual URLs in HTML, and null fields/no fabricated success for absent evidence. Actual evidence was also rendered into a separate `/private/tmp/bufo-final-showcase-check` directory, leaving the root's showcase output untouched. No full-suite repeat was needed for this narrow read-only report addition.
