# Task 5 independent-review corrections

Scoped edits: `src/bufogen/cli.py`, `src/bufogen/loop_report.py`, `tests/test_forge_cli.py`; pre-edit snapshots under `task5fix-before/`. No frozen generation runtime, prompt, scoring policy, dependencies or live service state changed.

All three reviewer reproducers were first encoded as tests and observed failing:

1. Accepted current-train review previously overrode a saved heldout split/family. Export now independently checks BOTH the current configured prompt contract and the generation's frozen benchmark metadata against the explicit review. Any conflict fails before an output target file. Missing fields in a present frozen contract also fail closed. A current YAML ID cannot launder saved heldout provenance.
2. `--forge-state` parser default is now unspecified. `loop-report` therefore defaults to `--loop-root/forge-state.json`, via the report API. A report only reads another state when `--forge-state` is explicitly provided; `FORGE_STATE_PATH` is intentionally ignored for reports. Forge import/submit/status retain their workflow default resolution from `FORGE_STATE_PATH` or the original repo path. CLI help explains the distinction.
3. Report now reuses `benchmark._validate_rows` on source rows, checking actual prompt/model/run/configuration/family/requirements/image identity. A saved summary requires full expected prompt × arm coverage and exact per-arm totals. Complete scored-row coverage must match original source rows, generation IDs, frozen scorer configuration and requirements. The report recomputes the summary from saved scores and rejects drift. Partial valid generation manifests without a summary remain `running`, with no inferred metrics.

Focused verification: **18 CLI/report tests passed**. Added coverage also rejects wrong totals, foreign arms and changed metric counts, while permitting a valid incomplete running manifest. Actual local showcase successfully regenerated from the original pilot plus ongoing heldout run with the stricter checks. Existing W&B SDK import emitted failed optional Sentry telemetry attempts in this restricted shell; the report code performs no service calls or mutations.

Recovery work is separately documented in `task6-recovery-report.md` / `task6-diff.txt`.

Final regression including separately scoped recovery work: **258 Python tests passed** in 5.39 seconds, **3 Node conversation tests passed**, whole-tree Ruff passed. Existing Weave/Pydantic warnings remain.
