# Serverless RL live result — October 15, 2026

**One real RL optimizer update completed, and its immutable checkpoint served a
valid agent-generated PNG. No quality improvement has been measured.**

| Evidence | Recorded result |
| --- | --- |
| Base model | `google/gemma-4-26B-A4B-it` |
| Displayed fine-tuned model | `ft-google/gemma-4-26B-A4B-it` |
| Initial checkpoint | `bufo-gemma-sol-pilot-20261015:v1`, SFT step 1 |
| Actual optimizer job | `20c27927-a11e-4148-a359-0090bcacc1ef` |
| Completed update | Step 1 → step 2, same registered storage and base model |
| Immutable RL checkpoint | `wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v2` |
| Artifact digest | `ab0a6a69bd55396972dc91a126edb125` |
| Registry linking | Failed: organization for `wandb-smle` could not be resolved |
| Agent serving | Successful retry produced a valid PNG and a Weave trace |

The [recovery receipt](../../runs/forge-gap/rl-direct-result.json) records the
completed backend update, actual training-job ID, artifact ID/digest, logging
run, original SFT checkpoint, and serving verification. The
[independent checkpoint lookup](../../runs/forge-gap/rl-checkpoint-verification.json)
confirms immutable v2. Four genuine scored student rollouts passed exact Dynamo
token validation and full local ART tokenization before the optimizer submission.

The [original training record](../../runs/bufo-gemma-dynamo-direct-rl-20261015/rl-training.json)
retains `status=failed` and `CommError` because optional Registry linking failed
**after** `completed_step=2` and the returned checkpoint had been saved. That
historical record was preserved. The separate recovery receipt distinguishes the
completed training from the failed Registry operation; no training was repeated
to recover it. The [state-handling fix](rl-promotion-state-fix.md) keeps those
outcomes separate in future runs.

The first full agent serving attempt returned HTTP 500 and produced no image;
its [error receipt](../../runs/forge-gap/rl-serving-smoke/error.json) remains intact.
The successful retry has a
[serving receipt](../../runs/forge-gap/rl-serving-smoke-retry1/receipt.json),
[rendered PNG](../../runs/forge-gap/rl-serving-smoke-retry1/e0540b75be4446b590f78b757904d7cd.png),
and [agent conversation trace](https://wandb.ai/wandb-smle/bufo-gen/weave/agents/conversations/open_rl%3Arl-serving-coffee-smoke%3Ae1ceb15587f1).
Its final reply was “Here you go—Bufo enjoying a coffee break!” This is a serving
and rendering check, not an image-quality evaluation.

Earlier failures remain separately recorded: the
[original history-tokenization failure](../../runs/bufo-gemma-sft-rl-recovered-20261015/rl-training.json)
and the [training-gateway HTTP 524 attempt](../../runs/bufo-gemma-dynamo-rl-20261015/rl-training.json).
The corrected path uses the verified ART Dynamo-support revision and sends
managed-client inference to `api.inference.wandb.ai`; see the
[research evidence](rl-research.md) and [routing diagnosis](rl-inference-routing.md).

The original 100-prompt comparison remains **1/100** strict passes for
`google/gemma-4-26B-A4B-it`, **71/100** for `gpt-6.1-sol`, and **1/100** for the
SFT v1 checkpoint. These results do not score RL v2. Neither this single update,
its training reward, nor its successful PNG establishes improvement or parity.
