# Task 6 dependency preflight and one-group recovery preparation

Scope: `pyproject.toml`, `uv.lock`, `src/bufogen/train_rl.py`, `tests/test_rl_recovery.py`; separate before snapshots under `task6dep-before`. Also prepared the requested one-off script `/private/tmp/bufo_recover_rl_group.py`. No generation/judge/optimizer calls or remote mutations were performed. The script's paid branch has NOT been run and needs independent review before root launches it.

## Dependency diagnosis and fix

ART 0.5.20's `model.log` calls `write_trajectory_groups_parquet`, which imports `pyarrow` inside the function. Installed ART metadata declares pyarrow under backend/tinker extras but not the basic serverless install. This explains the live attempt's failure after four scored captures but before optimizer submission.

Pinned `pyarrow==26.0.0` in this project's `rl` extra. The released wheel supports this Python 3.12 macOS arm64 runtime and satisfies ART's declared `pyarrow>=15.0.0` bound. Ran uv lock without upgrades and installed via the working simple Python subprocess path. Parsed old/new lock records and installed package inventories: **no existing version changes; only pyarrow 26.0.0 added**. Evidence: `task6dep-version-audit.json` and `task6dep-installed-before.json`. The editable project rebuilt with the same version. Existing agent/model/Weave/cwsandbox dependencies, source files, scorers and benchmark runtime were not changed.

`require_rl_logging_dependencies()` imports `pyarrow` and `pyarrow.parquet`, with an actionable `uv sync --locked --extra rl` error. `RLTrainer.train` invokes it before dataset access, model registration or rollout. Tests prove the missing dependency fails before dataset access and the real installed ART writer can persist/reload a genuine mocked-HTTP captured completion through Parquet.

## Narrow saved-group recovery script

Default source is the actual failed `runs/bufo-gemma-sft-rl-pilot-20261015-retry1` attempt. Default execution mode is local validation only; `--execute` is explicitly required for the paid path. No generic optimizer resume is implemented.

Validation requires that exact failure shape: SFT+RL, original Gemma, step 1, originally requested two updates, `ModuleNotFoundError`, and no training intent/completed-step/artifact fields. It selects exactly four scored captures with indices 0..3 at step 1 and exactly the four manifest generation IDs. The one failed partial capture is excluded from training but retained and hash-bound as source evidence. It validates source model/thread/prompt/split, frozen scorer and calibration, recomputes rewards, verifies actual captured student exchanges and requires unchanged reconstruction. Source state, manifest, every capture, PNG and SVG are SHA-256 bound. No choices/tool calls are generated or synthesized.

Local execution succeeded: 4 trajectories, 11 actual recorded student responses (2/4/2/3 choices per trajectory), rewards `[0.26066666666666666, 0.40656000000000003, 0.3728, 0.24928000000000006]`. Installed ART wrote `/private/tmp/bufo-rl-recovery-group.parquet`; receipt `/private/tmp/bufo-rl-recovery-validation.json` contains exact hashes and captured source identities. This is serialization verification, not a training result.

The paid branch exclusively creates a NEW durable output directory, writes the audited group as Parquet, resolves the actual deployed Forge/SFT artifact identity, registers the same verified storage identity and verifies the unchanged positive step-1 checkpoint/model ID/inference name before logging. It rechecks checkpoint/source hashes after logging, durably records optimizer submission intent, and invokes `backend.train` exactly once. It saves the actual returned step/artifact immediately before any metrics log or artifact lookup; resolves and saves the immutable version/ID/digest before Registry linking. Any later failure retains that evidence. Existing output directories are refused; there is no optimizer retry. State distinguishes `source_requested_steps=2`, `recovery_requested_updates=1` and actual completed updates. Root must inspect a failed submission's remote state rather than rerun it under another name blindly.

The script can be reviewed and locally validated with:

```bash
.venv/bin/python -c 'import runpy; runpy.run_path("/private/tmp/bufo_recover_rl_group.py", run_name="__main__")'
```

Do not add `--execute` until independent review is complete. Root retains ownership of the actual paid launch.

## Verification

- Recovery tests including dependency/real ART writer: **8 passed**.
- Whole-tree and recovery-script Ruff: passed.
- Full regression pending below.

Final full regression: **260 Python tests passed** in 5.48 seconds; existing Weave/Pydantic warnings remain. Dependency audit confirms only additive pyarrow. Scoped review diff: `task6dep-diff.txt`.
