# Conditional next step: contact-agent Serverless RL

This is a read-only implementation map, not an approval or claim that the current SFT candidate passed. **Do not begin RL unless the frozen candidate reaches at least 20/24 strict training-fit passes and the reward separately passes its review gate.** Neither condition is assumed here. No new generations, network requests, model registrations, or training submissions were made for this document.

The existing code is `src/bufogen/training/rl.py`. A prior project run completed one real serverless optimizer update from a Forge SFT checkpoint and served its result; that proves the existing continuation mechanism, not compatibility or quality improvement for these new contact trajectories. See [the preserved live result](../forge-gap-work/rl-live-results.md).

## Reusable code and the two immediate incompatibilities

| Code path | Current behavior | Contact experiment implication |
|---|---|---|
| [`train_rl.require_rl_runtime`](../../src/bufogen/training/rl.py) | Requires ART 0.5.20 from Git commit `45a17cbd2a85cc0333172e652fc1211c6ae3f3da`, not merely the same PyPI version. | Reuse this exact runtime gate and record source revision. Do not upgrade dependencies during the experiment. |
| `train_rl.student_rollout` | Accepts an `agent_factory`; captures `agent.generate(...)` inside `art.Trajectory()`, and saves raw exchanges and partial failures before returning. | The factory can construct the unchanged `contact_agent_v2.ContactAgent`. There is no need to call its private `_generate` directly. |
| `train_rl.validate_student_capture` | Requires one CoreWeave student/model and chat-completion exchanges, matched tool history, and tool names within `plan_scene`, `render_svg`, `search_web`. | This validator rejects `render_scene`. Add a separate contact-specific validator in the experimental module. |
| [`ContactAgent._generate`](contact_agent_v2.py) | Accepts an injected client, executes the real typed `render_scene`, saves scene/SVG/PNG, and captures actual SDK messages/tools through `capture_trajectory`. | These portable captures are useful audit evidence. They do not contain the exact training token/logprob evidence ART requires. Capture raw provider exchanges simultaneously. |
| `ContactAgent` constructor/model settings | Has no `training_logprobs` argument and does not set `logprobs=True`. | It cannot replace `BufoAgent` inside `RLTrainer.train` unchanged: that method passes `training_logprobs=True`. Use the separate adapter described below. |
| `train_rl.validate_training_tokens` | Requires every Dynamo exchange to be finished, with genuine prompt/completion token IDs and finite aligned completion logprobs; checks ART's mapping and `tokenize(multi_history=True)`. | Reuse unchanged before judging or optimizing. Do not reconstruct token IDs from visible text, SFT JSON, or another inference. |
| [`SceneQuality.score`](../../src/bufogen/evaluation/scorers/scene_quality.py) | Judges the actual final PNG against canonical Bufo identity, the prompt, and exact visual requirements. | Reuse outside ART's student-capture context. Compiler geometry diagnostics are not semantic reward. |
| [`train_common.reward`](../../src/bufogen/training/common.py) | Validates generation/rubric/checklist binding, then returns `similarity × requirement_coverage × mean(anatomy, composition, readability)`; invalid/error/missing/unusable cases receive zero. | A suitable existing dense reward, conditional on independent visual calibration. Partial checklist credit is intentional; it is not the strict evaluation pass rate. |

The SFT export guards in `trajectories._validate_messages`, its imported alias in `forge`, and `ContactTrainingContract` are **not the RL capture gate**. The current contact SFT adapter intentionally admits only one clean successful tool exchange. RL must retain genuine repair attempts and zero-reward task failures; applying that SFT filter would remove the examples RL needs.

## Smallest isolated adapter

Propose one new experimental `contact_rl.py` module and focused tests. Leave `src/bufogen/*`, `contact_agent_v2.py`, its instructions/schema, the compiler, and assets untouched. `RLTrainer.train` hardcodes its agent factory and validator, so it is not directly dependency-injectable. Reuse its existing standalone identity, token, capture, job-receipt, and checkpoint helpers in a small separate orchestration function rather than globally monkeypatching the production trainer.

1. Wrap the registered model's managed client in a narrow forwarding facade whose `chat.completions.create` adds **only** `logprobs=True`. Forward all other arguments and response objects untouched to ART's existing managed client. Do not construct a fresh raw OpenAI client: ART's proxy adds `return_token_ids`, `return_tokens_as_token_ids`, `nvext.extra_fields=["engine_data"]`, and `preserve_thinking`. The installed client does **not** enable logprobs itself. Retain the existing serving URL override to `api.inference.wandb.ai`, project header, zero transport retries, and explicit timeout.
2. Inject that facade into the unchanged `ContactAgent`; run public `generate` inside `art.Trajectory`. `ContactReplyModel` still delegates to the chat client; its PNG attachment affects tracing, not the student's sampled response. Renderer/telemetry operations are environment operations. Keep the judge and any teacher calls completely outside the capture context.
3. Validate the actual request model/provider/URL and exact frozen instructions/tool schema on every student exchange. Only `render_scene` is allowed. Verify real tool-call/result IDs and preserve the complete repair history. Separate protocol/evidence corruption from a model's invalid scene: a genuine model argument error with actual tool feedback is a task failure, not fabricated data. Do not silently remove malformed or unsuccessful attempts. Initially abort an optimizer group if an exception leaves an incomplete exchange rather than inventing missing tool results.
4. After a returned generation, bind final scene/SVG/PNG paths and hashes to its capture, record terminal status and exact requirements, and verify the PNG hash before and after scoring. `ContactAgent` snapshots do not populate the benchmark runner's final `png_sha256` automatically; the new adapter must do that. Save raw ART exchanges separately from portable SDK messages. Hash the new adapter, reused helper sources, compiler, agent, schema, scorer, reference assets, and runtime receipt.

Before any live rollout, a deterministic HTTP transport test must drive the actual contact tool call **and final reply** through this facade. Assert that the provider request contains the required metadata/logprob flags, captures only the student, preserves tool schema/history, and passes `validate_training_tokens`. Test an actual tool-validation-error/repair path too. This proposed facade is inferred to be compatible from the existing managed-client implementation and [`test_rl_dynamo.py`](../../tests/test_rl_dynamo.py); it has **not** yet been implemented or live-verified.

## Continuing the managed SFT checkpoint

The supported project path is **continuation in the actual deployed SFT model storage**, not importing an arbitrary adapter into a new ART name:

- `resolve_sft_identity` checks the deployed Forge job's base model, artifact ID/digest, immutable `:vN`, W&B run, entity/project, and actual artifact collection name.
- ART registers `TrainableModel(run_name=identity["storage_name"], ...)`; the pinned serverless backend uses `return_existing=True`.
- `verify_registered_continuation` requires the same run/storage/base identity, a positive service step, and that `:stepN` resolves to the exact frozen SFT artifact ID/digest **before any rollout**.
- Sample the current verified checkpoint through `get_inference_name(step=step)`. Preserve the immutable starting checkpoint and per-group step identity. Ensure no other job advances this same managed model concurrently; recheck the expected step before optimizing a collected group.
- Reuse `capture_training_job_receipt` to save the actual remote create acknowledgement/job ID once. Reuse `resolve_rl_checkpoint` to resolve the returned step to an immutable artifact version and digest. Optional Registry-link failure must not trigger another update.

The installed pinned `ServerlessBackend.train` supports CISPO/default or PPO, with the existing path using `learning_rate=5e-6` and `precalculate_logprobs=True`. It rejects custom `loss_fn_config` and `adam_params`. Registration does not forward a caller LoRA-rank setting. This proposal does not assume higher rank, a new optimizer implementation, teacher-token distillation, or arbitrary checkpoint forking. These statements describe the inspected pinned SDK and prior project proof, not an independent claim about every current service feature.

## Reward calibration is an additional gate

[`rl_contract.build_calibration_report` and `require_calibration`](../../src/bufogen/training/rl_contract.py) require explicit PNG-bound reviews, exact per-requirement boolean judgments, reviewer ID/evidence/timestamp, `identity`/`anatomy`/`composition`/`readability` ratings on 0–4, and genuine **train or validation** provenance. They rebuild the report, requiring positive and negative examples, no unusable judgments, zero false positives, and at least 80% agreement. This is labelled limited pilot agreement, not population calibration.

The completed neutral packets currently contain binary pass/fail, evidence, failed requirements, and PNG hashes. They lack the additional metric ratings and review fields. Do not invent those ratings from automated scores. Add a new immutable review supplement by inspecting the **same eligible images**, retaining the original decisions and receipt hashes.

The v2 development review is useful stress evidence but its split is `development`; the existing calibration helper will reject it. Do not relabel those images as training/validation. The validated 32-attempt publication currently exposes **two judge false positives** (one for each model) and one false negative. Those disagreements must remain visible even if a later eligible calibration cohort passes. All-attempt agreement alone does not satisfy the zero-false-positive gate.

Retest these exact, already-frozen counterexamples before accepting any proposed reward revision; verify their original PNG hashes and keep original and revised judgments side by side:

| Generation ID | Existing neutral image | Assistant-rejected requirement |
|---|---|---|
| `837a99b07aa9417f8f74f235ee500bcc` | [Image 26](../../runs/forge-contact-pilot/v2-development/visual-review/images/image-26.png) | A hand visibly grips the wrench handle |
| `b969c74630604b82b30846ef4fe78701` | [Image 17](../../runs/forge-contact-pilot/v2-development/visual-review/images/image-17.png) | A hand touches the laptop keyboard |

These are fixed development stress checks, separate from qualifying train/validation calibration. Require genuine new image inspection for any additional 0–4 ratings; binary reviews cannot supply those ratings by conversion. A revised reward should correctly reject both known failures before proceeding. This requires no new generation and must not consume the reserved confirmation prompts.

For the qualifying cohort, use a predeclared complete set of eligible training-fit/base-training outputs and their completed reviews, including genuine rendered failures and successes. Do not cherry-pick only agreeable cases or use the eight held-out diagnostics or the reserved 48 confirmation outputs. If eligible negative images or detailed ratings are absent, calibration remains blocked. Any reward/rubric adjustment needs its own version and recalibration; do not rewrite the frozen scorer or existing evaluations.

## Minimal future execution and accounting

Only after both gates pass, start with **one optimizer update**, using four fresh on-policy student rollouts on one predeclared training task, followed by independent evaluation before more updates. Keep the existing rank and initial `5e-6` recipe to test compatibility first. A group with equal rewards has no useful relative learning signal; record it and stop instead of rerolling until a favorable group appears.

Reuse `rl_contract.environment_tasks`/`load_environment_tasks` for train-only prompt/requirement tasks, not teacher response messages. Enforce the explicit disjoint diagnostic/reserved-family lists in the experiment receipt too. Sampling temperature may be 1.0 for exploration as in the existing RL runner; report that policy separately from frozen evaluation temperature.

Count **every planned rollout and every transport retry**. `student_rollout` saves partial captures and retries only connection errors/429/5xx, at most three attempts; it never retries the judge or optimizer. Completed, exact-token task failures remain zero-reward trajectories. Missing-token, interrupted, or infrastructure-failed captures remain failures in the audit denominator and must not be repaired or passed off as trainable successes. Its current terminal-exception behavior aborts the group, which is safer for the first pilot than filling the group with additional successful samples.

Report attempted/completed/token-valid/scored/trained counts separately, alongside all generation/judge errors, rewards, actual optimizer updates, immutable checkpoints, and post-update visual evaluation. Compare the RL checkpoint against its **same immutable SFT starting checkpoint under the unchanged contact-v2 evaluation harness**. Harness gains, SFT gains, and subsequent RL gains are separate claims. A successful optimizer job or rising training reward alone establishes none of them.
