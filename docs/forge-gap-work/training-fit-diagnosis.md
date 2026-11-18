# Training-fit diagnosis and next managed Forge experiment

Reviewed 2026-10-15. This is an **in-sample diagnostic**, not a new heldout
evaluation or a change to the selected candidate. No remote training or inference
was started for this diagnosis.

The selected `ft-google/gemma-4-26B-A4B-it` adapter has not learned reliable
action geometry even on these training prompts. A supported fallback experiment
is to test explicit lower learning rates on the same dataset and one-epoch
recipe. More epochs alone are not supported by the observed results. The user
has subsequently identified higher rank as available; verify that configuration
route before choosing between a rank-only comparison and the prepared LR sweep.
The inspected public managed API did not establish a rank control.

## Observed results

The selected artifact is
`wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-gap-e1-20261015:v1`.
The diagnostic run ID is `05de660317534953b1dd0440de08b6ad`.
Its [saved summary](../../runs/forge-gap/training-fit-selected/summary.json)
has SHA256 `b173d30cf2b8828831006ce75abee5dbdbb7743d7a2b302011f524460283eacd`.

| Measurement | Result |
|---|---:|
| Fixed training-family base prompts | 16 |
| Available, valid, scored generations | 16 |
| Generation errors / judge errors | 0 / 0 |
| Automated strict passes | 0/16 |
| Automated Bufo present | 16/16 |
| Mean per-image requirement coverage | 0.40625 |
| Satisfied individual requirements | 20/49 |
| Mean similarity / anatomy | 0.830625 / 0.72 |

Coverage is the mean of per-image fractions, not the pooled 20/49 fraction.
All 16 exact prompts have accepted targets in the reviewed training source.
One stochastic attempt per prompt does not establish an inability to reproduce
any target under every decoding condition. It does establish poor sampled fit
under the production-matched evaluation policy.

## Direct visual comparison with accepted targets

I inspected four student PNGs and their corresponding accepted `gpt-6.1-sol`
target PNGs, not just the captions or automated judgments. All eight PNG hashes
match their recorded generation/review hashes. This diagnostic sample is
unblinded and illustrative, not an independent estimate of pass rate.

| Training prompt | Selected adapter: visible issue | Accepted teacher: visible action |
|---|---|---|
| `office_coffee-base` | Blue mug floats above the extended hand with a clear gap. An identifiable open laptop is absent. | Fingers enter the mug handle; an open laptop rests on the desk. |
| `workshop_repair-base` | A gray bar is held well below the robot-like blue rectangle's bolt. An additional raised arm/hand complicates the anatomy. | A visible wrench jaw surrounds the robot bolt, with a connected hand gripping its handle. |
| `bakery-base` | The large apron covers the rolling pin and dough; the right hand is detached from the diagonal arm. | Both hands wrap around rolling-pin handles, with the pin resting across visible dough on the worktop. |
| `skate_park-base` | Blue clothing appears where the skateboard action should be; two long extra limbs extend off the image. No blue board is visible. | Both feet touch the same blue skateboard; wheels and the ramp make the action readable. |

The corresponding student generation IDs are
`d0f6b8a2c7c8442aab39a41c99c47055`,
`8b6cd4b13a3440aeb512fbc10bf72e89`,
`1d780b0f8d0f4344814ef783d45b19fa`, and
`115edea0dfc9440cbbedd090829a580a`.
Their teacher generation IDs, in the same order, are
`ae8be070f46543c7a6d57c3d77a00a5f`,
`87ec31893b1c45968d32f0064abd2750`,
`2b00e7b179514464a69b6b67ecca6d42`, and
`aa73d3cf9a894796afb03ebde7e7f3b8`.
Paths and hashes are retained in the
[diagnostic scored rows](../../runs/forge-gap/training-fit-selected/scored.jsonl)
and [accepted training records](../../runs/forge-gap/training/reviewed-examples.jsonl).

There is also a concrete judge-calibration issue: the bakery judgment credits a
rolling pin contacting dough near the bottom of the frame. These objects are
not visible in the saved PNG. The saved SVG places them before the later
Bufo/apron layer, which occludes them. The actual PNG hash matches the benchmark
hash; there is no evidence here of a stale file. The automated coverage should
therefore be treated as provisional and potentially optimistic. The bakery
strict failure remains correct. Existing scores should be preserved, with a
separate annotation recording this disagreement.

## What the evidence does and does not explain

For all four comparisons, the teacher target and sampled student trajectory
have byte-identical system instructions (7,314 characters) and identical tool
schemas. The target includes the `render_svg` tool arguments and a final reply;
the job uses `assistant_turns=all`. This reduces a simple missing-instruction or
missing-tool-schema explanation. It does not independently verify the remote
trainer's per-token loss masks, or prove every target received the intended
gradient.

The student tool arguments are substantially shorter than the corresponding
teacher arguments: office 1,812 versus 3,424 characters; workshop 2,352 versus
5,406; bakery 2,215 versus 4,712; skate park 1,763 versus 2,807. These are character
counts of serialized render arguments, not tokenizer lengths. In the inspected
images the shorter outputs omit contact structure and background detail.
Shortness itself is not a failure: a simpler SVG with the correct contacts would
be preferable to a longer incorrect one.

Saved actual W&B history provides additional evidence beyond earlier summary
snapshots:

| Job | Observed optimizer steps | Observed peak LR | Final LR | First / final logged batch loss |
|---|---:|---:|---:|---:|
| One epoch | 62 | `2e-4` | 0 | 1.7801 / 0.5915 |
| Three epochs | 186 | `2e-4` | 0 | 1.7231 / 0.5314 |

These values come from [e1 history](../../runs/forge-gap/training/e1/wandb-metrics.json)
and [e3 history](../../runs/forge-gap/training/e3/wandb-metrics.json).
The losses are different training batches, not fixed-corpus before/after
measurements. Falling token loss is compatible with poor free-running image
generation; it is not proof that spatial constraints were learned.

The completed validation comparison selected one epoch: it passed 2/24 versus
0/24 for the three-epoch adapter, which had seven generation failures. The base
passed 0/24 and `gpt-6.1-sol` passed 23/24. These are the recorded validation
results, not the training-fit diagnostic. Extra epochs at the same observed peak
LR did not close the gap.

Plausible explanations still include optimization settings, limited adapter
capacity, difficulty learning coordinate/contact relationships from the target
distribution, and compounding errors during generation. The inspected adapter
uses rank 1 per adapted matrix/expert, including MLP/MoE layers; see the
[capacity audit](adapter-capacity.md). That is evidence for investigating
capacity, not proof that rank alone explains failure. The generic Bufo identity
is strongly supported by the supplied character asset, so high identity scores
are weak evidence of learned scene composition.

## Next experiment: explicit LR, unchanged data and policy

The prepared LR fallback is two fresh managed Forge jobs from
`google/gemma-4-26B-A4B-it`, with peak learning rates **`2e-5` and `1e-4`**.
Compare against the existing `2e-4` one-epoch candidate as a prior control;
it is not a contemporaneous control replicate. Keep the exact reviewed 124 train /
23 validation dataset, batch size 2, one epoch, all assistant turns, character
assets, tools, generation policy and visual rubric fixed. Do not continue an
existing adapter or change prompts simultaneously. The exact source SHA256 is
`e8f2f4b23c30ae73bd8fd394ac340e0b6d299f3c277307b70ce1d2d3dc746984`.

Subsequent user steering says a higher-rank route is available. No LR jobs have
been submitted. Verify that route and its exact supported configuration before
choosing the next experiment; if available, a rank-only comparison with the
existing one-epoch `2e-4` recipe is an alternative that directly tests capacity.
Do not guess rank API fields or silently change an existing checkpoint.

Forge documents `model_config.config.learning_rate` as a positive peak scalar or
explicit per-step schedule. Its documented scalar policy uses linear warmup and
decay. This makes an explicit LR experiment feasible through the existing
managed product; record requested configuration, returned receipt and actual
logged schedule. [Forge fine-tuning](https://docs.coreweave.com/model-distillation/studio/fine-tuning),
[request schema](https://docs.coreweave.com/model-distillation/reference/management/fine-tunes/create-a-fine-tune.md).

The research motivation is conditional: controlled LoRA studies found learning
rate and capacity matter, and rank-1 optima could differ from larger-rank
settings. They swept learning rates rather than assuming an optimum. Those
experiments used other models/tasks and do not prescribe these numerical values
for Bufo. The proposed values are a project-specific bracket below the observed
peak, not a published recipe or guaranteed fix.
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/).

Predeclare the comparison before submission. Evaluate all candidates on the
same fixed 16 training-fit prompts and 24 development/validation prompts. Keep
all attempts, including generation failures, and inspect actual contact points
on a fixed sample with model identities hidden. Choose using development
strict passes, requirement coverage and generation failures, with a declared
tie-breaker. Training-fit improvement alone is insufficient to claim success.
If an apparent winner has only a one-case advantage, repeat the fixed
development comparison before treating it as a stable improvement.

This would be a new experiment and new selection receipt. It must not alter the
current selected adapter, frozen confirmation run or historical metrics. Any
confirmation set used to inform future fixes is no longer untouched evidence
for those fixes; reserve a fresh set before the next final claim.

## If LR changes still do not produce useful training fit

Do not respond by blindly increasing epochs or adding hundreds of equally
complex targets. First request service confirmation that SVG tool-argument
tokens receive the intended loss, and whether a supported higher-rank Forge
configuration exists. The public serverless path inspected in the capacity
audit does not forward the local `lora_config.rank` field.

The next dataset experiment can stay inside managed Forge: construct a small
contact-focused curriculum using authentic new teacher generations on training
families. Keep original input/trace provenance; select visually verified
demonstrations with one clear coordinate frame, few nested transforms, action
props painted in front of the body when appropriate, and explicit hand-to-prop
and prop-to-target contacts. Favor simpler successful geometry, not shorter
captions. Record rejected outputs rather than silently editing them into
purported teacher generations. Preserve the current dataset as the comparison
control and keep development/confirmation families out of the new training set.

A narrow memorization probe on accepted training scenes can test whether that
representation is learnable before scaling the dataset. Better fit there would
motivate expanding action coverage; poor fit would strengthen the case for a
training-contract or capacity limitation. Neither result establishes parity.
Student-failure-guided target collection is motivated by distribution-shift
research, but ordinary filtered correction SFT must not be labeled the actual
GKD objective: that method uses teacher distribution feedback on student
sequences. [On-policy distillation paper](https://arxiv.org/abs/2306.13649).
