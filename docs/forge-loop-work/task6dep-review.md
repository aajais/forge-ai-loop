# Task 6 dependency and saved-group recovery review

Reviewer: `codex/review_loop_runtime`. Reviewed the worker report/diff, dependency changes, `train_rl.py`, recovery tests, and `/private/tmp/bufo_recover_rl_group.py`. No remote calls or optimizer execution were performed. Only the script's default local-validation mode was run.

## Spec: PASS

- `pyarrow==26.0.0` is an additive pinned RL dependency. Independently comparing normalized installed package inventories and parsed lock records found no changed existing versions and only the new pyarrow package. The new import preflight runs before dataset access, registration, and rollout.
- Saved-group recovery requires the actual pre-training failure shape: SFT+RL, Gemma, step 1, originally two requested updates, `ModuleNotFoundError`, and absence of training-source-step, completed-step, or checkpoint fields. It does not treat an ambiguous optimizer failure as safe to retry.
- Exactly four scored capture records must cover exactly the four manifest generations and rollout indices. Model, conversation, prompt, train split, scorer configuration, requirements, calibration, and recomputed finite rewards are checked. The existing student capture validator checks model/protocol/tool/message constraints. Reconstructed exchanges must equal stored exchanges exactly.
- Failed partial captures are excluded from training and retained as hashed evidence. Source state, manifest, all captures, and PNG/SVG evidence receive hashes that are rechecked before submission. No model choices or tool calls are synthesized.
- The execute branch exclusively creates a fresh destination, writes the recovered group through the installed ART Parquet writer, and verifies the deployed immutable SFT identity, same remote training model, positive step-1 checkpoint, and expected inference name. It repeats the checkpoint/source checks after model logging.
- Optimizer submission intent is durable before the single `backend.train` call. There is no optimizer retry. The actual returned step/artifact/metrics are saved immediately, then the checkpoint is resolved to an immutable version with matching run provenance and saved before metrics logging or registry linking.
- State explicitly distinguishes the original two-update request, this recovery's one-update request, and actual completed updates. A failure retains submission/result evidence rather than claiming completion or replaying automatically.

## Quality: PASS

No blocking findings for this specific, single-launch recovery. The script is deliberately not a general concurrent recovery scheduler: a failed or uncertain optimizer submission requires inspection of remote state, not another invocation with a different output name. The exclusive directory and repeated checkpoint validation support the planned one-off workflow.

Independent validation against the genuine `runs/bufo-gemma-sft-rl-pilot-20261015-retry1` directory succeeded:

- Four genuine scored trajectories with 2, 4, 2, and 3 recorded student responses (11 total); one failed partial attempt excluded.
- Rewards recomputed as `0.26066666666666666`, `0.40656000000000003`, `0.3728`, and `0.24928000000000006`.
- Read the written Parquet back with pyarrow and reconstructed all four ART trajectories; every exchange and reward matched its source capture exactly.
- Local validation receipt: `/private/tmp/bufo-rl-recovery-validation.json`.

```text
.venv/bin/pytest -q tests/test_rl_recovery.py tests/test_rl_contract.py tests/test_rl_dataset.py
33 passed in 3.41s

.venv/bin/ruff check src/bufogen/train_rl.py tests/test_rl_recovery.py /private/tmp/bufo_recover_rl_group.py
All checks passed!
```

Existing Weave/Pydantic deprecation warnings remain. This review establishes local recovery integrity and code readiness; live checkpoint revalidation and the optimizer result remain execution-time evidence.

Approved recovery script SHA-256:

```text
73aa1eeed40f96d2ddbc1ae96faff15d60dc6c7150386936f6c2dab89bd89646
```

The root may proceed with the reviewed single paid update. `--execute` was not run by this reviewer.
