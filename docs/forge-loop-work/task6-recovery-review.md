# Task 6 bounded RL recovery review

Reviewer: `codex/review_loop_runtime`. Independent review of `task6-recovery-report.md`, `task6-diff.txt`, `train_rl.py`, and `test_rl_recovery.py`. No production edits, remote calls, or changes to the active Task 5 work.

## Spec: PASS

- The added retry loop surrounds only a student `agent.generate` call. It accepts OpenAI connection/timeout errors and HTTP 429/5xx, with at most three complete rollout attempts and 2/4-second backoffs. SDK request-level retries remain separate and bounded by the existing client configuration.
- Every attempt creates a new agent and ART capture with a distinct conversation ID. Failed attempts retain their actual exchanges, error type/status, and available partial-generation record. Captures from earlier attempts are not merged into a later successful trajectory.
- Only a returned capture proceeds through student-model validation and scoring into the optimizer group. Failed captures are never relabeled as successful; `captured` and `scored` remain separate states. A crash before completion leaves evidence marked `started`, not successful.
- Judge, capture-validation, model-log, optimizer, and registry operations are outside the new retry loop. A training timeout is recorded as failed with the attempted source step and is not automatically resubmitted.
- Attempt names must be a single directory component. Exclusive directory creation refuses an existing attempt before model registration. A fresh name still runs the existing remote identity and immutable SFT-checkpoint checks; it cannot bypass them to silently restart from base or a changed checkpoint.
- Error persistence records exception class and HTTP status, not exception messages, credentials, or request headers. ART's installed HTTP capture serializes model request/response bodies rather than authorization headers. Prompt and generated-content persistence is intentional audit data.

## Quality: PASS

No blocking findings. Atomic writes serialize before modifying the destination, fsync both file and containing directory, and fail rather than silently dropping unsupported values. The recovery tests exercise the installed ART capture mechanism with actual OpenAI SDK requests through mocked HTTP, including preserving an earlier real response after a later timeout. They also verify finite retry budgets, no retry for HTTP 400/401, empty failed captures, exactly one optimizer invocation on timeout, retained scored captures, and refusal to replay an existing attempt.

Independent local verification:

```text
.venv/bin/pytest -q tests/test_rl_recovery.py tests/test_rl_dataset.py tests/test_rl_contract.py
31 passed in 3.33s

.venv/bin/ruff check src/bufogen/train_rl.py tests/test_rl_recovery.py
All checks passed!
```

Tests emitted existing Weave/Pydantic deprecation warnings. This review approves the recovery implementation; it does not independently verify the live service checkpoint or claim a successful RL optimizer step. Root must retain its remote checkpoint evidence before launching a fresh named attempt.

Reviewed SHA-256 hashes:

```text
src/bufogen/train_rl.py: 8c64dee8fd5cd9e649c4db982a9b65e655cf7e7f395d77ca9cdcd0c9d265a2b0
tests/test_rl_recovery.py: 5a00f1202e7d013f8149b822d7bf9538cebeb69f71c46b242b124377360eb2f7
```
