# Task 6 training-job receipt review

Reviewer: `codex/review_loop_runtime`. Independent read-only production review of `task6jobid-report.md`, `task6jobid-diff.txt`, modified trainer/tests, and installed ART 0.5.20 serverless backend/client source. No remote calls or runtime changes.

## Spec: PASS

No blocking findings.

- The hook uses the pinned backend's actual `training_jobs` resource instance. Installed ART calls `create(model_id=..., trajectory_groups=..., experimental_config=...)` with keyword arguments and receives a `TrainingJob` with a string `id`, matching the wrapper exactly.
- The installed backend starts event polling only after the awaited `create` returns. Consequently the wrapper's durable acknowledged-ID checkpoint completes before polling starts. It preserves and returns the original response object.
- Submission intent records the actual model ID and source step before invoking the SDK method. Connection, timeout, cancellation, or other create failures retain an explicit error receipt and no invented job ID. A response without a usable ID fails closed while retaining evidence of the ambiguous submission.
- A second create within the same optimizer call is refused. The hook delegates transport to the original SDK method and adds no retry. Inspection of the installed SDK confirms `Client.request` disables POST retries.
- Receipt restoration runs in `finally`; both an existing instance override and the original class descriptor are restored appropriately. Nested installation on the same resource is rejected.
- The existing exclusive-attempt-directory policy still prevents automatic replay after a failed training call. Polling failures retain the acknowledged ID in the outer failed state. The receipt intentionally represents the current or most recent update, rather than a full job-history ledger.

## Quality: PASS

The private seam is narrow, explicitly version-gated, and covered by a test using the installed SDK resource and response type. The test checks durable intent before the SDK POST boundary and acknowledged ID before simulated polling, without replacing the SDK's request payload construction. Other focused tests cover timeout, polling failure, descriptor restoration, trainer state integration, and existing replay refusal.

Independent verification:

```text
.venv/bin/pytest -q tests/test_rl_job_receipt.py tests/test_rl_recovery.py tests/test_rl_dataset.py
17 passed in 3.31s

.venv/bin/ruff check src/bufogen/train_rl.py tests/test_rl_job_receipt.py tests/test_rl_dataset.py tests/test_rl_recovery.py
All checks passed!
```

Existing Weave/Pydantic deprecation warnings remain. This approval covers future job-ID receipts. It does not repair the separate remote `Could not locate a history message in the rendered history` failure, prove that failure safe to resubmit, or retroactively instrument an already running process.

Reviewed SHA-256 hashes:

```text
src/bufogen/train_rl.py: f9854e6574d5da5aa1e637c8d1ff3fe732869c94a50eb9bdb4d819cb18490757
tests/test_rl_job_receipt.py: 444464378b9e95fa9914b9899aab93944111a3444c30121ce36525804453eb9a
```
