# Exact Dynamo evidence for Gemma Serverless RL

Implemented October 15, 2026. No live inference, optimizer submission, or training
was launched during this implementation task.

## Changes

- `pyproject.toml` and `uv.lock` pin ART to the verified upstream commit
  `45a17cbd2a85cc0333172e652fc1211c6ae3f3da`. The distribution remains version
  `0.5.20`; runtime preflight now verifies its installed VCS repository and commit,
  not merely that ambiguous version. All installed and locked package versions
  are unchanged; only ART's source changed. See `rl-dynamo-dependencies.json`.
- `RLTrainer` uses the registered model's managed inference client, preserving
  ART's Dynamo extension and token settings through `with_options`. It supplies
  the project header, a 180-second timeout, and zero transport retries. The
  existing bounded whole-rollout retry remains responsible for transient errors.
- `BufoAgent` accepts `training_logprobs=False`. RL opts in; ordinary generation
  retains its existing settings and behavior.
- Every new captured student exchange must carry finished Dynamo engine prompt
  IDs, completion IDs, and finite aligned completion logprobs. Those arrays must
  agree exactly with ART's training representation. Missing or inconsistent
  evidence is rejected before visual judging, trajectory logging, or optimizer
  submission. Old incompatible captures are not repaired or reused.
- Full upstream `trajectory.tokenize(multi_history=True)` runs as a mandatory
  local preflight. Its histories and token counts are durably recorded alongside
  the capture. Exact engine evidence bypasses tokenizer reconstruction, so no
  Transformers/PyTorch dependency or model download is needed.
- The existing actual training-job receipt hook still persists the server's job
  ID immediately after acknowledged submission, and never repeats that POST.

## Evidence

The installed runtime successfully processed the untouched genuine two-exchange
probe from the research task: 103/39 and 107/6 prompt/completion token counts;
two histories with 142 and 113 total tokens. Neither `transformers` nor `torch`
was imported. Receipt:
`runs/forge-gap/rl-probe/installed-runtime-preflight.json`.

New tests drive the actual managed ART client through OpenAI Agents' chat model
using deterministic HTTP responses. They assert the project header, Dynamo
extension, logprobs request, exact serialized evidence, and successful upstream
tokenization. Negative cases cover missing evidence, unfinished responses,
invalid/empty token IDs, missing/misaligned/nonfinite logprobs, conflicting SDK
mapping, and the same-version-but-wrong-source installation. An integration case
proves unusable metadata stops the trainer before the judge or optimizer.

Earlier focused RL suites: 36 passed. New transport/preflight suite: 12 passed.
Final combined RL, agent pipeline, and reply-model suite: **55 passed**. Ruff
checks passed. Reviewable scoped runtime diff: `rl-dynamo-runtime.diff`.
Full project suite at initial verification: 285 passed, one unrelated model-label
assertion failed in `tests/test_forge_cli.py:314` (expected the removed phrase
`no Sol-parity claim`); parent notified to coordinate that label-owned test.

## Limits

This establishes client capture and local preprocessing compatibility. A new
genuine rollout group and an actual remote optimizer result are still necessary
to establish service training compatibility. It is not evidence of improved
image quality or parity with `gpt-6.1-sol`.

Before-edit snapshots: `docs/forge-gap-work/rl-dynamo-before/`. No Git repository
is available, so these snapshots delimit this task's runtime changes.
