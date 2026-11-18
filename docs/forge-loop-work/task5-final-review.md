# Task 5 final integration review

Reviewer: `codex/final_review`, 2026-10-15. Independent review of `task5-diff.txt`, current Task 5 implementation, approved plan, task report, runtime approval and relevant integration contracts. Production files and running jobs were not changed. Only this review artifact was written; reproductions used temporary directories. No live service calls were made.

## Verdict

**Spec: PASS. Quality: PASS.** Independent re-review of `task5fix-diff.txt` resolves all three original blockers. No new blocking issues were found in the scoped fixes. The original findings below are retained as review history and are now closed. The ongoing heldout benchmark and separately reviewed RL recovery work remain outside this Task 5 approval.

## Re-review and approval

The corrections were inspected independently and their regression tests rerun:

- Export now checks current prompt declarations and frozen generation metadata independently. A current train declaration cannot override a saved heldout split or conflicting family. Production rows with no benchmark metadata still use their explicit reviewed provenance.
- Reports leave the omitted `--forge-state` unset, so the report API resolves it beneath the selected loop root. Explicit overrides remain supported. Forge import/submit/status separately retain their workflow environment/default path behavior.
- Report rows now pass frozen benchmark identity/image validation. A saved summary requires exact prompt/arm coverage, corresponding scored rows bound to the source and judge, and metrics recomputed from those rows. Valid partial runs without summaries remain allowed.

Fresh verification:

```text
.venv/bin/pytest -q tests/test_forge_cli.py tests/test_forge.py tests/test_training_contract.py tests/test_benchmark.py
57 passed, 732 existing Weave/Pydantic deprecation warnings, 3.40s

.venv/bin/ruff check src/bufogen/cli.py src/bufogen/loop_report.py tests/test_forge_cli.py
All checks passed!
```

Read-only report construction against actual local evidence succeeded: pilot `scored`, 32/32 generations; heldout `running`, 149/300 generations at inspection. That count is a snapshot, not a completed evaluation claim. SHA-256 checks confirmed all five runtime-review approved files (`agent.py`, `models.py`, `ui.py`, `benchmark.py`, `config/models.yaml`) remain unchanged. No live calls, production edits, report regeneration or running-job interventions were performed by this reviewer.

## Original blockers — all resolved

### 1. P1 — Current prompt YAML can override a generation's frozen heldout partition

Location: `src/bufogen/cli.py:167–172` (`export_reviewed`).

`contracts.get(row.prompt_id) or row.benchmark_metadata` selects the current benchmark declaration whenever its ID exists, ignoring the row's saved split/family. If a benchmark changes partition assignments, or export is pointed at another benchmark with overlapping IDs, a generation originally reserved for heldout evaluation can be accepted as training data. The later gates inspect the current heldout list and the review's rewritten partition, so they do not recover the original restriction.

Offline reproduction: create an authentic captured fixture, use a current train prompt ID in the row and trajectory provenance, set frozen `benchmark_metadata` to `{"split":"heldout","family":"original-heldout-family"}`, and provide an accepted review with the current train partition/family. `export_reviewed` succeeds and emits `partition="train"`. Observed output: `Frozen partition: heldout; Exported partition: train`.

Required fix: independently validate the review against existing frozen row metadata, reject current-config conflicts, and ensure frozen heldout/test rows cannot qualify regardless of current YAML. Preserve the explicit review path for production rows that genuinely have no benchmark metadata. Add regression coverage for current-YAML drift and independent frozen family conflicts.

### 2. P2 — A different loop root silently inherits the original experiment's Forge deployment

Location: `src/bufogen/cli.py:98–102, 216–220`.

The shared parser assigns `--forge-state` the repository pilot state path, and `loop-report` always forwards it. Thus `--stage loop-report --loop-root runs/another-experiment` reads another experiment's dataset, fine-tune and deployed artifact instead of allowing `build_loop_report` to resolve `another-experiment/forge-state.json`. Missing training evidence can consequently appear deployed using genuine but unrelated IDs.

Offline reproduction: parse `--stage loop-report --loop-root <temporary>/unrelated-experiment`; its implicit Forge state remains `/Users/aayush.jaiswal/Desktop/cw-bufo-gen/runs/forge-loop/forge-state.json`.

Required fix: resolve the report's omitted state path under the selected loop root; retain explicit `--forge-state` overrides and define any intentional environment override. Keep existing default state behavior for Forge mutation/status commands. Test report dispatch with two distinct roots and a missing state.

### 3. P2 — Showcase pairs unrelated manifest rows with a valid benchmark summary

Location: `src/bufogen/loop_report.py:32–64` (`_benchmark`).

Summary IDs are checked against `benchmark.json`, but manifest row run/configuration IDs, prompt/model/arm identities and summary coverage are not checked. Unique row keys alone do not bind images to metrics. A copied, stale or truncated manifest can therefore produce a report marked `scored` with unrelated images and the original successful summary.

Offline reproduction: copy the actual pilot `benchmark.json` and `summary.json` into a temporary directory; retain one manifest row, changing its prompt, model, backend and metadata run ID to foreign values. `build_loop_report` accepts it and returns `{"accepted_foreign_row":"foreign-arm","status":"scored","generated":1,"summary_attempts":32}`.

Required fix: validate each manifest row against its frozen configuration (the existing `benchmark._validate_rows` already provides the core checks), reject unexpected keys, and verify summary arm/count coverage before accepting it. Continue allowing valid partial generation manifests without a summary. Add mismatch and valid-running-run tests.

## Passing integration observations

- CLI commands bind to the actual benchmark, Forge and annotation APIs. Import validates its source before client construction; submission remains distinct from polling.
- Explicit review decisions, source IDs, authentic tool trajectories and validation partitions are otherwise retained. Export does not regenerate reviewed examples.
- Proxy opt-in, direct model defaults, source fingerprint coverage and real-response model provenance remain covered by the separate runtime review and SDK/HTTP tests.
- Documentation distinguishes assistant reviews, limited pilot calibration, teacher/judge correlation, deployed SFT, optional RL and demonstrated improvement. It does not claim an RL checkpoint or Sol parity.
- Comparison groups keep different fingerprints separate and do not pool repeated arms. Failure denominators are preserved by the reviewed benchmark scorer; the report needs the additional input binding described above.

## Verification

```text
.venv/bin/pytest -q tests/test_forge_cli.py tests/test_forge.py tests/test_training_contract.py tests/test_benchmark.py tests/test_rl_contract.py
69 passed, 1,250 existing Weave/Pydantic deprecation warnings, 4.95s

node --test tests/chat_ui.test.cjs
3 passed

.venv/bin/ruff check src scripts tests
All checks passed!
```

These tests and the temporary reproductions used local fixtures/mocked transports. No successful test is presented as evidence of live training, deployed RL, completed heldout scoring or browser-rendered HTML. The subsequent scoped re-review and Task 5 approval are recorded above.
