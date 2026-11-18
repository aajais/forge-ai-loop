# Forge AI Loop: live evidence, 2026-10-15

The application, curation path, managed Forge SFT, deployed adapter and heldout evaluation have run against real services. The small SFT pilot did **not** achieve `gpt-6.1-sol`'s scene-generation quality. A corrected Serverless RL run subsequently completed one optimizer update and produced immutable v2, which served a real agent-generated PNG. Registry linking failed separately. No RL quality improvement has been measured; see the [current RL evidence](../forge-gap-work/rl-live-results.md).

## Compare the actual images

[W&B visual comparison](https://wandb.ai/wandb-smle/bufo-gen/runs/rqbi08g1) contains 300 attempt rows, including 292 rendered PNGs and eight generation failures. Each row retains the prompt, model, scene scores, failure evidence and trace link. The local [showcase](../../runs/forge-loop/showcase/index.html) joins the loop's datasets, training state, queues and native Weave evaluations.

| Model | Heldout attempts | Valid images | Strict scene passes |
| --- | ---: | ---: | ---: |
| `google/gemma-4-26B-A4B-it` | 100 | 95 | 1 |
| `gpt-6.1-sol` | 100 | 100 | 71 |
| `ft-google/gemma-4-26B-A4B-it`, SFT v1 | 100 | 97 | 1 |

All three arms use the same 100 heldout scenarios and frozen comparison policy. Generation failures remain in the denominator. All 300 attempts have published Weave feedback, and each arm has a native Weave evaluation. These are provisional `gpt-6.1-sol` judge results: every explicit visual requirement must pass, alongside the character, anatomy, composition and readability thresholds. They are not population-calibrated probabilities. The same model also supplies the teacher examples, which introduces potential correlated preferences.

A separate blinded assistant spot-check inspected 15 heldout PNGs: `google/gemma-4-26B-A4B-it` 0/5, SFT v1 0/5, `gpt-6.1-sol` 3/5, with 15/15 agreement with the frozen judge. This small check is supplementary evidence, not comprehensive human calibration. Heldout reviews were not accepted as training targets.

## What was trained

The user authorized assistant-reviewed pilot targets. Actual PNG inspection produced **10 eligible training trajectories and four separate validation trajectories**, with assistant provenance retained. Forge accepted the dataset and deployed the SFT job:

- Job: `100202f8-7034-4bc2-bd2f-2375dd43fbba`.
- [Training run](https://wandb.ai/wandb-smle/bufo-gen/runs/bufo-gemma-sol-pilot-20261015).
- Adapter: `wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v1`.
- Exact base: `google/gemma-4-26B-A4B-it`.

Real deployed inference rendered a PNG. That validates serving, not visual improvement. The full heldout comparison shows that this deliberately small pilot was insufficient to close the quality gap.

## Observe and curate

[Annotation queue](https://wandb.ai/wandb-smle/bufo-gen/weave/annotation-queues/01a1403d-e1af-7922-aa5f-a553c849a728) and [ARIA handoff](../../aria_prompts/04_forge_loop.md) are ready for user review. Assistant acceptance is recorded separately from human annotation-queue approvals.

A read-only Agent Lens query verified project-wide conversation insights: 404 conversation groups, 390 with intent sentiment and 280 with failure severity at the recorded timestamp. These counts include previous traffic and retries. Clustering completion and user ARIA curation have not been verified.

The Forge proxy agent smoke completed with a real rendered PNG. The native application turn places tools before the final text and PNG. The Forge proxy also emits separate gateway request turns; the trace verification documents this limitation rather than claiming a merged single turn.

## Earlier failures and subsequent successful RL update

Same-model SFT continuation was verified against the actual storage identity, artifact digest and step 1 checkpoint. Four genuine student trajectories were rendered, scored and saved. A single recovered optimizer submission then failed remotely with:

```text
ValueError: Could not locate a history message in the rendered history
```

At that time, the remote checkpoint query still returned only steps 1 and 0. The original failed attempts and exact error receipt remain under `runs/`. Do not replay a failed training attempt merely by deleting its state directory.

Local replay with untouched captures and the official tokenizer reproduced the failure. The [original diagnosis](gemma-rl-tokenization-diagnosis.md) and `runs/forge-loop/rl-tokenization-diagnosis/` preserve that evidence. Subsequent research identified upstream ART's exact Dynamo token support, absent from the PyPI package despite sharing version `0.5.20`. The project pinned the verified upstream commit, required genuine engine prompt/completion IDs and logprobs, and validated every new rollout locally before training. A separate attempt then hit HTTP 524 through the training gateway; routing inference through the verified serving endpoint resolved that path. No sampled outputs or rewards were rewritten to make training pass.

The corrected run submitted real job `20c27927-a11e-4148-a359-0090bcacc1ef` and completed step 2 from the same SFT storage at step 1. Its immutable checkpoint is `wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v2`. Registry linking failed because the organization for `wandb-smle` could not be resolved; the original training record retains that post-training error. The separate [verified recovery receipt](../../runs/forge-gap/rl-direct-result.json) confirms the completed update, artifact provenance, and subsequent successful agent serving/rendering check. An initial HTTP 500 serving failure is also preserved. **This establishes training and serving, not improved image quality or a fourth evaluated arm.**

Quality comparison remains a separate evaluation task. The larger teacher curriculum uses development families; the original heldout results must not become accepted training examples. See the [follow-up evaluation protocol](../forge-gap-work/evaluation-runbook.md).

## Verification and local UI

The earlier full-project checkpoint passed **265 Python tests, three Node chat tests, and Ruff**. Subsequent RL promotion-state verification passed **52 focused RL tests and Ruff**. Live checks include managed Forge deployment, deployed model inference, cwsandbox rendering, Forge proxy generation, PNG tracing, native Weave evaluation publication, and the corrected RL update with a served PNG. No browser-level visual test is claimed.

Restart the existing studio process to load the deployed adapter from `.env`:

```bash
uv run bufo-studio --reload
```

Model choices use exact model IDs. Fine-tuned checkpoints display as **`ft-google/gemma-4-26B-A4B-it`**, with checkpoint version and tuning method distinguishing SFT from SFT+RL. `OPEN_RL_MODEL` may reference the verified immutable v2 artifact; its availability in an already-running studio depends on that process's loaded configuration.
