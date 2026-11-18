# Task 4 — Scene-reward RL and provenance-qualified SFT continuation

Implementation complete; awaiting independent review. This worker submitted no training jobs, changed no remote routes, and created no remote artifacts.

## Changes and public interfaces

- `src/bufogen/train_rl.py`: uses released ART 0.5.20 `with art.Trajectory()` / `trajectory.exchanges`, and the now-required `TrainableModel(run_name=...)`. Student agents disable web search per instance; SceneQuality executes outside capture with `rescore=True`. Captured requests must use the exact student model and verified CoreWeave inference endpoint; non-chat protocols, mixed teacher/judge calls, missing choices, unknown tools and orphan/unresolved tool results are rejected. ART does not expose request URL in exchanges, so endpoint identity is checked on the configured client rather than falsely claimed as captured metadata.
- `src/bufogen/train_common.py`: `reward(row, *, scorer_configuration)` replaces legacy CLIP/VLM/length shaping. Valid bound scene judgments earn `similarity * constraint_coverage * mean(anatomy, composition, readability)`. Missing Bufo, invalid renders, generation errors, missing/malformed/nonfinite/out-of-range metrics, foreign generation IDs, requirements or scorer configurations earn zero. Coverage is recomputed from the exact checklist; an unchecked cached derived scalar is never used. Training always requests a new judgment.
- New `src/bufogen/rl_contract.py`: `build_calibration_report(scored_rows, reviews, *, scorer_configuration)`, `require_calibration(report, *, scorer_configuration)`, `environment_tasks(rows)`, `write_environment_tasks(rows, output_directory)`, and `load_environment_tasks(train_jsonl_path)`.
- `RLTrainer(dataset_ref, *, name, base_model, calibration_path=None, initial_checkpoint=None, forge_finetune_id=None, forge_project_alias="bufo-gen")`; `await trainer.train(steps=...)` remains the launch API. Missing calibration fails before model registration. Initial adapters require an actual deployed Forge job ID. The primary `base_model` remains original Gemma, separate from the optional initial adapter.

`write_environment_tasks` accepts full validated benchmark rows, writes `train.jsonl` plus metadata/source digest, selects only train scenarios and explicitly excludes conversational edits (counts recorded). RL v1 does not silently reinterpret an edit without its prior requests. The artifact loader rejects nontrain/conversational rows outright. Tasks contain prompt ID/text, exact requirements, family and explicit split; no teacher response messages. Legacy reference-image rows lacking requirements/family fail actionably. Publish the directory through `RegistryClient.push_dataset`; retain the returned real versioned reference.

Calibration joins exact generation ID, prompt ID, PNG checksum, requirements, split and scorer configuration. Reviews require provenance, timestamp, exact evidenced checklist, explicit visual pass and 0–4 metric labels. Visual pass is independent of SFT acceptance (a good image with unqualified prior assistant targets can still calibrate vision). Heldout images cannot qualify the reward. Reports retain observations, reviewer types/identities, raw scores, checklist disagreements, scalar absolute errors, counts, excluded reviews and judge errors. `require_calibration` rebuilds from embedded evidence rather than trusting `rl_ready`.

Readiness is deliberately limited pilot agreement: at least one positive and negative, zero judge errors, zero false positives and at least 80% overall agreement. Passing this policy is **not population calibration** (`population_calibrated=false`); the underlying rubric remains uncalibrated. Policy thresholds are explicit in the report. A new scorer configuration requires a new matching report; no Qwen-specific assumptions are used.

## Same-model SFT → RL

`resolve_sft_identity(initial_checkpoint=..., forge_job=..., api=..., entity=..., project=..., base_model=...)` is read-only. The trainer fetches the actual Forge job before calling it. It requires deployed status, exact base, real artifact and W&B run ID; resolves the supplied and deployed adapter references to the same immutable artifact ID/digest/version; and verifies its logged-by run/entity/project and known base metadata. The source artifact collection, not a guessed model/task UUID, becomes ART's durable `run_name`.

ART's documented registration uses `return_existing=True` for that provenance-qualified storage identity. The API has no lookup-only registration mode. `verify_registered_continuation(model, identity, api)` checks the returned run identity/base/storage, positive service checkpoint step and the exact ID/digest of that `stepN` artifact before any rollout or training. A missing/deleted underlying model could still be recreated by registration, but step zero then blocks continuation; it is never reported as SFT+RL. There is no experimental artifact-fork shortcut. Immutable SFT URI remains recorded for the A/B arm while the same training collection advances.

Base RL uses a separate fresh name and is labeled `base_rl`; an already-trained name is rejected. SFT continuation is labeled `sft_rl`. State under `runs/<name>/rl-training.json` records original base, immutable SFT identity, dataset, calibration, actual registration IDs, steps, returned artifact and failure type. This driver is not an automatic crash-resume scheduler: inspect actual service state after interrupted submissions instead of assuming replay is safe.

## Dependency verification

Pinned optional `openpipe-art==0.5.20` and intentionally updated `uv.lock`. Installed only after root confirmed the long-lived pilot had completed. ART requires OpenAI <3, so the resolver changed OpenAI 3.26.1 → 2.54.0 and Agents SDK 0.23.1 → 0.20.0. Existing app tests remain part of the full regression; root separately reports a real deployed-adapter agent/render smoke passed on these versions. No Weave/W&B/cwsandbox downgrade.

Direct `uv` execution in this environment panicked in macOS system-configuration; launching the same command through the approved simple Python subprocess succeeded:

```text
.venv/bin/python -c 'import subprocess; subprocess.run(["uv","sync","--locked","--extra","rl","--cache-dir","/private/tmp/bufo-uv-cache"],check=True)'
```

The wheel source was inspected for capture, register, checkpoint and train APIs. One integration test uses installed ART with actual OpenAI/httpx mocked HTTP transport, proving a student request is captured and a later judge request is not. It also constructs the actual new TrainableModel with its required run_name.

## Evidence and limitations

Current real Qwen calibration report: `docs/forge-loop-work/task4-calibration-current.json`. It joins all 17 existing reviewed images: 15 visually positive, 2 negative, 1 judge error, 2 false positives, observed agreement 13/17 (76.47%). It correctly blocks RL. This worker did not modify the scorer or reinterpret those human/assistant observations. Root is separately testing a Sol judge; its scores must pass the exact new scorer configuration through this gate before launch.

Changed tests: new `tests/test_rl_contract.py`, legacy reward assertion in `tests/test_learning.py`, and the obsolete reference-image/AutoTrajectory training test in `tests/test_rl_dataset.py` (ownership extension approved by root; original saved under task4-before). The replacement training-boundary test proves exact environment requirements, disabled research, judge outside capture, four student rollouts, positive scene rewards, actual checkpoint promotion and heldout rejection before any second training call.

Verification: focused `tests/test_rl_contract.py tests/test_rl_dataset.py tests/test_learning.py`: **30 passed**. Ruff passed all modified Python files. Full-suite result is recorded below when final run completes. Existing Weave/Pydantic deprecation warnings remain.

Final full regression after dependency changes: `.venv/bin/pytest -q` → **224 passed, 456 known deprecation warnings in 4.95s**. The 17-image calibration file is a timestamped work-stage snapshot, not qualification of the later expanded 32-image review set. Rebuild against all completed reviews before launch.

Independent review found that the metrics-only `model.log` call inherited ART's default `val` split. Fixed it to `split="train"`, matching the trajectory log. The training-boundary regression now asserts the metrics split and completed step; it failed with `KeyError: split` before the fix. Registry promotion remains a separate downstream operation: if it raises, workflow status is failed but the already-completed step and returned checkpoint artifact remain in `rl-training.json`; inspect those before considering any repeat training.
