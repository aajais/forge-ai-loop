# Dynamo RL implementation review

Verdict: **approved for one bounded live optimizer compatibility attempt**, using fresh genuine rollouts and the existing continuation checks. This approval is a code-review result, not evidence of a successful remote update or quality gain.

Read-only review compared the `rl-dynamo-before` snapshots with current `src/bufogen/train_rl.py`, the minimal `training_logprobs` change in `agent.py`, `pyproject.toml`, `uv.lock`, new tests, and `rl-research.md`. No runtime or experiment files were changed by this reviewer.

## Verified contract

- The dependency and lockfile pin official ART commit `45a17cbd2a85cc0333172e652fc1211c6ae3f3da`. Runtime preflight checks version **and** Git repository/revision through installed distribution metadata; it rejects the older PyPI package that happens to share version `0.5.20`. The reviewer ran the actual preflight successfully.
- The registered model's managed OpenAI client is injected into the existing agent, preserving its actual inference model/checkpoint request. `with_options` retains ART's default request extensions. The real SDK transport test confirms `nvext.extra_fields=[engine_data]`, exact-token flags, preserved-thinking request, project header, and `logprobs=True` survive the Agents SDK.
- The only ordinary-agent behavior addition is an opt-in `training_logprobs=False` constructor setting. Existing generation receives no new logprobs argument unless RL explicitly enables it. No prompt, tool, rendering, temperature, or output transformation was added by this diff.
- Every captured exchange must contain finished engine metadata, nonempty integer prompt/completion token IDs, finite aligned completion logprobs, and one sampled choice. ART's mapped training representation must equal the provider's recorded arrays exactly.
- The **entire captured trajectory** then runs through ART's real `tokenize(multi_history=True)` before the scene judge or optimizer. Distinct server-rendered histories are retained. Missing/mismatched metadata is rejected, never reconstructed or copied from another call.
- Raw captures are persisted before validation. Successful tokenization receipts are saved before scoring. Failed captures are not silently replaced with fabricated successful histories.
- Existing SFT artifact ID/digest, deployed Forge job, base-model, actual run/storage identity, positive step, and registered checkpoint checks remain unchanged. Existing exact request-model and allowed-tool capture checks remain in place. Web search remains disabled during student rollouts; scoring occurs outside the ART capture scope.
- The managed inference client has `max_retries=0`. The existing explicitly bounded whole-rollout retry policy remains unchanged and excludes scoring, token validation and optimizer submission. Inspection of the pinned serverless client confirms it disables retries for **POST** requests. The existing receipt wrapper permits one create per optimizer invocation, records submission intent, and stores the actual returned job ID before event polling.
- The lockfile changes only ART's package source, not other dependency versions. No GPU/backend extras were introduced.

## Independent verification

Reviewer ran:

```text
.venv/bin/pytest -q tests/test_rl_dynamo.py tests/test_rl_contract.py tests/test_rl_recovery.py tests/test_rl_job_receipt.py
43 passed
```

Existing Weave/Pydantic deprecation warnings remained. Tests exercise the actual ART/OpenAI/Agents transport through deterministic HTTP fixtures, reject missing/invalid/mismatched metadata, verify source pinning, and cover the preserved continuation/recovery/submission safeguards. Tests do not perform a live optimizer update.

## Limits and follow-through

The research evidence establishes same-checkpoint inference metadata and local preprocessing. A successful remote training result and actual immutable checkpoint are still required before claiming Serverless RL works end to end. The next attempt must use newly captured metadata-bearing trajectories; the previous failed captures cannot be repaired by changing this runtime. Do not automatically resubmit a job whose submission outcome is unknown.

The existing training loop logs post-update metrics before persisting `step_completed`. If that later logging fails, the remote checkpoint may already exist; inspect the persisted real job ID and remote checkpoints rather than treating a local `failed` status as proof no update happened. This is a pre-existing lifecycle limitation, not introduced by the Dynamo patch, and does not justify a duplicate optimizer call.
