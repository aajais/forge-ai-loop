# Task 1 independent review

Final spec compliance verdict: **pass within the agreed version-1 scope**. Final code quality verdict: **pass**. No unresolved important correctness findings in the refreshed task1 diff.

## Resolved review finding

**P1 — Cumulative history could bypass review and split qualification (resolved).** The initial implementation restored all previous assistant actions but qualified only the latest generation's review, prompt ID, family, and partition. A reviewed training follow-up could therefore include an unreviewed, rejected, or held-out earlier target. The final `src/bufogen/trajectories.py:capture_trajectory` records `prior_conversation` and adds a nonexportable diagnostic for any previous snapshot; `export_training_example` also explicitly checks that provenance. `src/bufogen/agent.py` passes the previous snapshot. Legacy ancestors already fail through unauthenticated-history gating. The new same-model follow-up regression covers the originally missed case.

This intentionally makes all external-history captures ineligible for version-1 SFT until per-turn review/split qualification exists. Normal conversation continuation remains available. Repairs within one generation call remain exportable because they are part of one SDK run sequence rather than an external conversation snapshot. Mixed models and changed system/tool/prompt snapshots are also recorded and rejected, including descendants.

## Spec and correctness assessment

Actual `RunResult.to_input_list()` output is captured across repair attempts; the installed SDK implementation preserves the submitted input prefix as the completeness check expects. Actual call IDs/results, compact tool arguments, frozen tool schemas and the real final reply are retained. Hidden reasoning and application-only PNG publication are excluded from model actions. Legacy manifests still deserialize, legacy continuation uses explicit application context without synthetic calls, and unsupported multimodal exports fail while continuation retains image attachments. PNG persistence/display paths remain intact.

Review decisions require source identifiers, distinct reviewer kind/identity, timezone-aware timestamps, evidence and explicit acceptance. Current-row held-out IDs/families and nontraining partitions are excluded, and the external-history guard closes ancestor leakage. Missing and negative reviews cannot be replaced by automatic scores. No model configuration change, manufactured review, fabricated action or claimed trained artifact was introduced.

The new module keeps capture, conversion, continuation and validation responsibilities localized. Curation uses the exporter and provides concrete skip diagnostics. No broad refactor is needed for this task.

## Deliberate integration limits

The existing `curate` CLI does not expose review import arguments; its orchestration generates fresh random generation IDs before importing static decisions. Reviewed-manifest import plus `select_examples()` works directly. Task 5 owns the export-existing route; the old CLI should not be described as a completed reviewed-curation workflow. Multimodal SFT and multi-turn SFT remain explicitly unsupported. Real provider capture and Forge upload still need the bounded live integration pilot.

## Validation evidence

Read the brief, refreshed supplied diff, final worker report, affected implementation/tests, relevant UI history/media paths, and installed SDK `to_input_list()`. Scoped recheck covered the resolved ancestor qualification issue, persistent mixed-settings/model exclusion and continued within-request repair eligibility. No implementation files were edited and passing suites were not rerun.

Worker report records **42 passed in 1.08s** for `tests/test_trajectories.py`, `tests/test_training_contract.py`, `tests/test_pipeline.py`, `tests/test_ui.py`, and `tests/test_learning.py`, plus Ruff passing for changed Python files. These are worker-supplied execution results; the independent reviewer inspected the implementation and coverage.
