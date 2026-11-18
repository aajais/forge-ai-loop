# Deployed adapter capacity audit

Read-only inspection on 2026-10-15. No inference, training submission, runtime
change or weight-tensor download was performed.

The deployed `ft-google/gemma-4-26B-A4B-it` one-epoch candidate has a **rank-1
LoRA parameterization per adapted matrix, including each MoE expert**. This is
supported by the actual tensor shapes and the pinned ART conversion code, not
just the `r` field in a JSON file. It does **not** prove that rank is the cause of
the remaining quality gap, or that a larger rank will match `gpt-6.1-sol`.

## Artifact and job evidence

| Field | Observed value |
|---|---|
| Artifact | `wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-gap-e1-20261015:v1` |
| Artifact ID | `QXJ0aWZhY3Q6Mzk4OTk1MzY2OA==` |
| Artifact digest | `aa0221f0448ebb9955322b0e973b95b8` |
| Forge job | `ac29564f-fde2-45e8-a87a-9d22beaf2e20` |
| Observed job state | `deployed` |
| Base identity, artifact metadata and adapter config | `google/gemma-4-26B-A4B-it` |
| Returned job model config | `{"config":{"epochs":1,"batch_size":2,"assistant_turns":"all"}}` |
| Reviewed dataset counts | 124 train, 23 validation; dataset revision 2 |
| Artifact files | `adapter_config.json` (1,093 bytes), `adapter_model.safetensors` (61,907,664 bytes) |
| Adapter config SHA256 | `0e2c527ba2e4371b960ed8d74e07d43de002d296d97df6f93da8598090ff64f8` |

The logging run is `wandb-smle/bufo-gen/bufo-gemma-gap-e1-20261015`.
Its W&B config was empty. Neither the artifact metadata nor the returned Forge
config exposed a resolved optimizer, learning-rate schedule, or configurable
rank override. This audit concerns this specific artifact, not an assumption
about every Forge model or the three-epoch candidate.

## Actual adapter configuration and tensor shapes

The config records `r=1`, `lora_alpha=32`, `lora_dropout=0.0`,
`art_lora_format="vllm"`, `peft_type="LORA"`, and `peft_version="0.18.1"`.
`rank_pattern` and `alpha_pattern` are empty. `use_rslora`, `use_dora` and
`use_qalora` are false. Target modules are `k_proj`, `gate_proj`, `down_proj`,
`q_proj`, `up_proj`, `v_proj`, `experts`, and `o_proj`. Therefore, lack of
MLP/MoE targeting is not a supported explanation for this candidate's quality.

The safetensors header was read with two bounded HTTP Range requests:
`bytes=0-7` followed by `bytes=8-72143`. Both returned HTTP 206 with the expected
Content-Range. The reader was set to abort without consuming the body if the
server ignored Range. Total downloaded file metadata was 72,144 bytes;
**zero tensor payload bytes were read**. The JSON header is 72,136 bytes, with
SHA256 `f5f5f77ffb2b28b18bc6e26608da0b0e161d0b8e9661c1c48bf289036e69a4a6`.
It describes 540 BF16 tensors. Representative layer-0 suffixes follow; all use
the prefix `base_model.model.model.layers.0.`.

| Module | LoRA A shape | LoRA B shape |
|---|---|---|
| `mlp.down_proj` | `[1, 2112]` | `[2816, 1]` |
| `mlp.gate_proj` / `mlp.up_proj` | `[1, 2816]` | `[2112, 1]` |
| `self_attn.k_proj` | `[1, 2816]` | `[2048, 1]` |
| `self_attn.o_proj` | `[1, 4096]` | `[2816, 1]` |
| `moe.experts.base_layer` (packed gate/up) | `[128, 2816]` | `[1408, 128]` |
| `moe.experts` (packed down) | `[128, 704]` | `[2816, 128]` |

The MoE dimension 128 packs **128 experts times rank 1**, not rank 128 for each
expert. At the installed ART revision
`45a17cbd2a85cc0333172e652fc1211c6ae3f3da`:

- `art/megatron/model_support/handlers/gemma4.py:1802`,
  `_vllm_moe_config`, copies the config and adds `experts` to target modules;
  it does not rewrite the rank.
- The same file's `_to_vllm_lora_tensors` (line 1883) concatenates expert A
  factors along dimension 0 and packs expert B factors.
- `_from_vllm_lora_tensors` reads `rank = int(adapter_config["r"])`, computes
  `num_experts = gate_up_a.shape[0] // rank`, and slices each expert's A factor
  using `expert * rank : (expert + 1) * rank`.
- `art/megatron/model_support/internal_padding.py:36` stacks B factors and
  flattens the rank/expert dimensions; its inverse reshapes using the same
  separate `rank` and `num_experts` values.

Dense factors independently confirm a rank dimension of one. This establishes
the representational rank bound of each adapter update. Without reading tensor
values, it does not establish whether a particular learned update is nonzero or
has numeric rank exactly one. The pinned local exporter is evidence for the
format's semantics; the remote training service's exact source revision was not
returned by the inspected job.

## Supported controls versus unverified controls

Current official Forge documentation supports epochs, batch size, training all
or only the final assistant turn, and learning rate as either a positive peak
scalar or an explicit nonnegative per-step schedule. The documented scalar
policy uses 10% linear warmup then linear decay to zero. The creation API shows
`model_config.config.learning_rate` and an `experimental_config` object.
An unspecified experimental object is not a verified rank-control contract.
[Fine-tuning documentation](https://docs.coreweave.com/model-distillation/studio/fine-tuning),
[Create fine-tune API](https://docs.coreweave.com/model-distillation/reference/management/fine-tunes/create-a-fine-tune).

The local Forge wrapper currently submits epochs, batch size and assistant-turn
selection. The actual job's returned config contains these three controls only.
We should not label an unreturned default learning rate as known.

The W&B run summary does expose logged values:

| Logged summary key | Value |
|---|---:|
| `sft/loss/learning_rate` | `5.454545454545455e-05` |
| `train/loss/learning_rate` | `0.00010161290322580656` |
| `sft/loss/train` | `0.7102424502372742` |
| `train/loss/train` | `0.7174205111880456` |

These are summary metrics, not a resolved peak/schedule declaration. Their
aggregation semantics were not established in this audit. They do not identify
the optimizer, its betas/epsilon/weight decay, or the service's peak LR.

The pinned ART internal `TrainerRuntimeSpec` has `lora_rank`, `lora_alpha` and
`lora_target_modules` fields; its validator requires alpha 32 for that Megatron
implementation. This does not demonstrate that Forge accepts these fields in
`experimental_config`, or that rank is user-selectable for this base model.
Likewise, an SDK-side SFT default is not evidence of this remote job's default.

## Implications for the next experiment

The earlier [research review](research.md) remains directionally appropriate:
data coverage, objective masks, optimizer settings and capacity should be
distinguished. Research found that sufficient rank, appropriate LR and MLP/MoE
coverage can matter; rank-1 success in some RL tasks does not establish adequacy
for this SVG SFT task. Nor does the research justify multiplying an unknown
service default by ten. [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/),
[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673).

If validation still shows a material gap, the defensible next steps are:

1. Complete the frozen one-versus-three-epoch comparison. Inspect development
   failures and training fit before attributing poor scene composition to rank.
2. Obtain the service's actual resolved peak LR/schedule and optimizer identity.
   A controlled LR comparison is supported by the public interface; record an
   explicit schedule/config and keep data, rank, epochs and evaluation fixed.
3. Verify with the Forge service contract whether this exact Gemma base supports
   changing rank. Only after verification propose a matched capacity experiment;
   do not silently send guessed `lora_rank` fields or change alpha/modules.
4. Use development families for subsequent fixes. If confirmation outcomes guide
   a new experiment, freeze a fresh confirmation set before claiming generality.

This audit adds a concrete capacity hypothesis and rejects an attention-only
targeting hypothesis. It supplies no evidence that the gap has already closed.

## Follow-up: can we actually request a larger rank?

**No verified client-configurable higher-rank route was found for managed Forge
or ART ServerlessBackend.** This is a boundary of the inspected public contracts
and implementation, not a claim that the service operator cannot provide one.
The existing e1/e3 jobs and comparison remain unchanged.

The complete official Markdown/OpenAPI representation, rather than the collapsed
HTML field list, confirms that `model_config.config` accepts only `batch_size`,
`epochs`, `learning_rate`, `assistant_turns`, and `metric_logging`, with
`additionalProperties: false`. `experimental_config` accepts arbitrary JSON
syntactically but specifies no rank key or downstream meaning. Sending a guessed
rank there could be accepted without changing the training configuration.
[Published request schema](https://docs.coreweave.com/model-distillation/reference/management/fine-tunes/create-a-fine-tune.md).

The pinned ART source gives a stronger warning about an apparently plausible
alternative:

| Layer | Actual rank plumbing |
|---|---|
| `TrainableModel(lora_config={"rank": ...})` | The Python model accepts and stores this field. |
| `ServerlessBackend.register` | Calls `models.create` with entity, project, name, base model and return-existing only. It never forwards `model.lora_config`. |
| Serverless `Models.create` | Request schema/body has no rank or LoRA-config parameter. |
| Serverless SFT submission | Serializes batch size, learning rate, assistant-turn selection and optional metric logging. No rank. |
| Serverless `ExperimentalTrainingConfig` | Defines RL/loss/packing controls; no rank. The SFT endpoint does not accept this object through the inspected SDK. |
| `LocalBackend._get_service` | Passes `model.lora_config` into `get_model_config`; this is genuine configuration plumbing. |
| Megatron `DistributedModelService` | Reads `lora_config.rank` for identity-adapter creation and `TrainerRuntimeSpec.lora_rank`. |

These observations come from
[pinned ServerlessBackend source](https://github.com/OpenPipe/ART/blob/45a17cbd2a85cc0333172e652fc1211c6ae3f3da/src/art/serverless/backend.py),
[pinned serverless client](https://github.com/OpenPipe/ART/blob/45a17cbd2a85cc0333172e652fc1211c6ae3f3da/src/art/serverless/client.py),
and [pinned LocalBackend source](https://github.com/OpenPipe/ART/blob/45a17cbd2a85cc0333172e652fc1211c6ae3f3da/src/art/local/backend.py).
Read-only downloads of current public `main` for the serverless backend/client
also contained no rank-forwarding parameter at audit time. No unverified service
POST was used to probe this.

The rank-1 observation is consistent with the pinned
[`megatron/lora_config.py`](https://github.com/OpenPipe/ART/blob/45a17cbd2a85cc0333172e652fc1211c6ae3f3da/src/art/megatron/lora_config.py):
it sets the MoE default rank to 1, dense default to 8, and alpha to 32. This is
corroboration, not proof of the remote service's deployed source revision.
The internal `ART_MEGATRON_LORA_RANK` environment setting belongs to the trainer
process; setting it in our laptop's environment does not transmit it through the
managed API.

For a self-managed ART deployment the exact supported Python field is
`lora_config={"rank": desired_rank}`. The inspected source carries that field
through to the Megatron trainer. Choosing compatible GPU resources, model
runtime, maximum serving rank and adapter format would still require a separate
deployment validation. This is not a managed Forge fine-tune receipt, and it
must not be presented as one. Direct serverless `train_sft` remains a real SFT
path for the reviewed targets, but the inspected path does not solve rank
selection and does not itself create the Forge management job record.

To keep higher-rank training inside the requested Forge product workflow, the
missing requirement is a verified service-supported rank field or operator-side
configuration for `google/gemma-4-26B-A4B-it`, including allowed ranks, serving
limits, and how the resolved value appears in the job/artifact. Once supplied,
a new trial should verify its exported config and tensor header before calling
it a higher-capacity comparison. Until then, explicit learning-rate experiments
and better development data are the supported managed levers established here.
