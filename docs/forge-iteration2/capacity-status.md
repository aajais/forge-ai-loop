# Managed Forge capacity and experiment controls

Read-only recheck: 2026-10-16. No training, deployment, dataset upload, remote
configuration change, dependency change, or model-weight download was performed.

**The managed public interface still does not expose a verified rank-selection
field. Data and explicit learning-rate experiments can proceed now.** The user
subsequently clarified that higher rank likely is unsupported in serverless and
asked us to proceed with supported controls. There is no pending rank-setting
question or higher-rank job. Iteration 2 therefore improves the reviewed data
and uses explicit learning rate, epoch count and the existing managed service.

## Current evidence

The official [fine-tune creation schema](https://docs.coreweave.com/model-distillation/reference/management/fine-tunes/create-a-fine-tune.md)
was fetched successfully with a direct read-only Python HTTP request after the
browser retrieval failed. It still restricts `model_config.config` to
`batch_size`, `epochs`, `learning_rate`, `assistant_turns`, and `metric_logging`
with `additionalProperties: false`. `model_config.experimental_config` accepts
arbitrary JSON, but the schema supplies no rank field or semantics. A request
accepting an arbitrary key would not prove that the trainer applied it.

Current public [ART ServerlessBackend](https://github.com/OpenPipe/ART/blob/main/src/art/serverless/backend.py)
still registers the model using entity, project, name, base model and
return-existing only; it does not forward `model.lora_config`. The
[serverless client](https://github.com/OpenPipe/ART/blob/main/src/art/serverless/client.py)
does not add a rank or LoRA configuration argument to model creation or SFT.
The current [Megatron defaults](https://github.com/OpenPipe/ART/blob/main/src/art/megatron/lora_config.py)
still specify MoE rank 1, dense rank 8 and a trainer-process environment setting
`ART_MEGATRON_LORA_RANK`. Setting that variable on this laptop does not transmit
it through the managed API. Public source does not establish the deployed
service's exact revision or current operator overrides.

Authenticated GETs through the existing `ForgeClient` returned:

| Candidate | Forge job | Current status | Resolved artifact |
|---|---|---|---|
| 1 epoch | `ac29564f-fde2-45e8-a87a-9d22beaf2e20` | deployed | `wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-gap-e1-20261015:v1` |
| 3 epochs | `3d999308-e1b5-44ee-84ce-0eab850b456a` | deployed | `wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-gap-e3-20261015:v3` |

Both still report `google/gemma-4-26B-A4B-it`, 124 training examples, 23 validation
examples, batch size 2 and all assistant turns. Their configs contain only those
settings and their respective epoch counts. No new rank, optimizer, sequence
length or context fields were returned. These immutable prior jobs cannot tell
us whether an operator has changed the default for *future* jobs.

The earlier [actual adapter configuration and tensor-header audit](../forge-gap-work/adapter-capacity.md)
remains the capacity evidence: the selected 1-epoch artifact uses rank 1 per
adapted matrix, including each MoE expert, alpha 32 and attention plus MLP/MoE
targets. No second weight download was necessary.

## Controls usable now

The [official fine-tuning guide](https://docs.coreweave.com/model-distillation/studio/fine-tuning.md)
and creation schema support these controls:

| Control | Contract | Existing project support |
|---|---|---|
| Training rows | Reviewed dataset entries; optional deterministic training-only cap | Reviewed JSONL upload and immutable source hashes |
| Epochs | Positive integer | `ForgeClient.create_finetune(epochs=...)` |
| Batch size | `auto` or 1–32; documented auto resolves to 2 | `batch_size=...` |
| Learning rate | Positive scalar peak or nonnegative per-update list | Positive scalar `learning_rate=...`; list is not implemented in this wrapper |
| Assistant-turn loss | `all` or `last` | Workflow deliberately submits `all` |
| Rank | No verified public field | Requires documented service field or operator configuration |

For scalar learning rate the guide describes 10% linear warmup followed by
linear decay to zero. A list provides explicit values for each optimizer step.
The prior jobs' [actual W&B histories](../forge-gap-work/training-metrics-note.md)
show a peak of `2e-4`; this is measured behavior, not a guessed default.

The runnable project request is:

```python
client.create_finetune(
    dataset_id=reviewed_dataset_id,
    model_name=new_immutable_trial_name,
    base_model="google/gemma-4-26B-A4B-it",
    epochs=1,
    batch_size=2,
    learning_rate=2e-4,
)
```

This snippet documents the existing supported call; it was not executed during
this audit. The workflow also binds dataset revision, source hash and job ID.
Keep `assistant_turns="all"`: SVG-producing tool calls are earlier assistant
turns, so choosing `last` risks optimizing only the short final caption.

## Historical capacity hypothesis, not a scheduled trial

The earlier rank-32 proposal remains a hypothetical diagnostic only. It is not
part of the current managed iteration-2 experiment. The following checks would
apply if the product later documents a supported capacity control.

Before labeling a new job rank 32, obtain the exact setting and confirm that it
applies to newly created Gemma training models and their inference serving path.
Verify the exported `adapter_config.json` and bounded safetensors header after
training. Keep old checkpoints and their provenance unchanged.

A clean capacity experiment uses the same reviewed data, epochs, batch size,
explicit LR and evaluation policy, recording alpha, target modules and any
optimizer changes separately. Rank and alpha jointly determine effective LoRA
scaling: changing rank from 1 to 32 while retaining alpha 32 changes the usual
alpha/r multiplier from 32 to 1. Therefore a new higher-rank result is not a
pure parameter-count comparison unless scaling is also controlled or disclosed.
Do not invent an alpha override where the service has not documented one.

Data quality and capacity trials can be separate arms. If both data and rank
change at once, report the combined recipe's performance without attributing the
gain to either intervention. Prior confirmation prompts have already informed
diagnosis; they are development evidence for subsequent iterations. Freeze new
unseen families before claiming that a new candidate approaches
`gpt-6.1-sol` on a held-out comparison.

## Retrieval fingerprints

Successful current HTTP reads (SHA256 of complete response body):

| Source | SHA256 |
|---|---|
| Forge creation Markdown/OpenAPI | `bb2047864d78166424ade418b64e9941d8c9aaad5b7f7561aa5c25aba19c680d` |
| Forge fine-tuning guide Markdown | `b9ebba163838eb4d5a1e554cd9c0a45f5d3c0af13860010bc10683851c2cfde1` |
| ART main serverless backend | `9769ac83dc21cc191b8b35db7214ec13f0f894cdaeef70d2b87fbe8132dbb6a1` |
| ART main serverless client | `a6a5884dcc0e601f6c668b0d5a761d1d2ee32e8b873727ba4416bd4ac6cf8793` |
| ART main Megatron LoRA defaults | `98219149c1ae6ab3e2e101c8e915d708b4d321a059718a76cfbd83302d2d47a9` |

The authenticated read scope was exactly the two listed Forge jobs in
`wandb-smle/bufo-gen`, using local environment credentials without printing them.
No broad project scan or trial POST was used to probe undocumented controls.
