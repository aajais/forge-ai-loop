# Next experiment after the training-fit diagnostic

**Latest decision:** the user can arrange higher adapter rank. Pause the
lower-LR job submissions described below and first verify the supported route
for rank 32. If available, change adapter capacity while retaining the
observed `2e-4` schedule policy, one epoch, batch size two, and the same source
data. Inspect the new exported adapter's actual configuration and tensor shapes
before labeling it a rank-32 comparison. Record alpha/r scaling and rsLoRA too;
if scaling changes, do not attribute the effect to rank alone. The optional LR support remains useful
for a later controlled comparison; no LR job has been submitted by this audit.

The selected one-epoch adapter failed all 16 base training prompts under the
frozen production agent policy (temperature 0.7), with requirement coverage
0.40625. These are in-sample diagnostic results, not heldout estimates. The
alternative LR comparison would change learning rate while keeping the same
reviewed data, model, one epoch, batch size two, and all-assistant-turn loss.
Do not increase epochs again as the first response to this evidence.

## What matches, and what remains unknown

Compared each of the 16 saved training-fit trajectories with its actual teacher
training example. All 16 have exactly equal system messages, first user
messages, and tool schemas. Both encode `tool_calls[].function.arguments` as
JSON strings; the teacher examples contain 17 render calls and the student
attempts contain 18. The application-level request format is therefore aligned.
The teacher's Responses API capture is exported to the same Chat Completions
message/tool schema used by the student. Teacher-private reasoning is not a
training target or a required serving input.

This establishes message/schema equality, not equality of the hosted trainer
and inference server's final token IDs. Training and serving can still differ
in template revision, tool-argument normalization, tool-choice generation
prefixes, or thinking defaults. Actual client code uses `tool_choice=required`
for initial agent requests and does not explicitly specify `enable_thinking`.
Its subsequent agent tool/final-reply handling is shared across evaluation
arms. The previous [loss audit](tool-loss-audit.md) verified SVG supervision
locally; the service has not exposed its masks or effective sequence limit.

The fit images fail mainly on hand/prop contact, object counts, missing details,
occlusion, and pointing direction. Their SVG tool arguments are much shorter:
median 1,767 characters (range 1,543–2,544), compared with teacher median 3,106
(range 2,518–4,885) across these matched examples' render calls. This supports
investigating omitted construction detail, but shorter SVG alone is not an
error metric and should not receive reward.

## Thinking support: model capability versus this endpoint

Google documents Gemma4 thinking via `enable_thinking=True` in the model's
chat template, and documents the empty thought-channel prefix used by larger
Gemma4 variants when thinking is disabled. This is model/template capability,
not a statement about CoreWeave endpoint defaults.
[Google thinking documentation](https://ai.google.dev/gemma/docs/capabilities/thinking).

The cached tokenizer defaults `enable_thinking=False`; pinned ART's local
helper also sets it false and preserves existing thinking text. CoreWeave's
public catalog confirms this exact Gemma model's tool and LoRA support, but
the documentation inspected did not establish a per-request thinking override
contract or default for this hosted adapter endpoint.
[CoreWeave model catalog](https://www.coreweave.com/products/coreweave-forge/serverless-inference).

Do not silently inject thinking into the frozen evaluation or attribute the
failures to thinking mode without evidence. A separate diagnostic can compare
actual Dynamo prompt token IDs from a minimal inference request with the local
template, using the same messages, tools, model artifact, and tool-choice
setting. Existing successful Dynamo metadata probes establish a route for this
check; they do not establish the current adapter's exact template. A request
override being accepted is not proof it changed template behavior.

## Small inference control before interpreting another adapter

Run the base model and selected `v1` adapter at temperature zero on four fixed
training examples: `archery_range-base`, `bakery-base`, `office_coffee-base`,
and `gpu_garden-base` (full IDs retain `gap-train-`). Keep instructions, tool
schemas, rendering, generation limit, and scoring unchanged; save a fresh
diagnostic protocol and all eight attempts. This tests whether decoding noise
substantially changes the observed failures. It is neither a new benchmark
selection nor evidence of generalization. If this looks materially different,
expand the same control to all 16 existing training-fit prompts before claiming
that training failed to fit them under every serving policy.

## Controlled managed learning-rate comparison

The actual previous histories peak at learning rate `2e-4`; the longer run
continues its schedule across epochs and performs worse on validation, with
seven generation failures. This makes a lower-rate experiment useful, but does
not prove the old rate caused the failures. The agreed follow-up candidates
use peaks **`2e-5` and `1e-4`**, each trained from
`google/gemma-4-26B-A4B-it` for **one epoch**, batch size **2**, with
`assistant_turns=all` and exactly the existing 124/23 split. Treat the completed
`2e-4` one-epoch candidate as the historical control, preserving any unexposed
seed/scheduling limitations.

Forge documents a positive scalar learning rate or a nonnegative per-step
schedule. Its scalar uses warmup followed by decay; the request nests this
under `model_config.config.learning_rate`.
[Fine-tuning controls](https://docs.coreweave.com/model-distillation/studio/fine-tuning),
[Create fine-tune schema](https://docs.coreweave.com/model-distillation/reference/management/fine-tunes/create-a-fine-tune).

The repository's Python workflow now exposes an optional LR argument; its CLI
has no `--learning-rate` flag. The scoped implementation adds optional
finite-positive `learning_rate` arguments to
`ForgeClient.create_finetune` and `forge_loop.submit_training`, includes the
value in the durable requested configuration, and preserves the exact old
payload when omitted. Its Python workflow call is:

```python
# Use a fresh initialized/imported workflow state for each candidate.
state = submit_training(
    state_path,
    client,
    epochs=1,
    batch_size=2,
    learning_rate=2e-5,  # second independent candidate: 1e-4
)
```

Before submission, bind each new state to the same source/upload hashes and
keep its unique model name and job receipt. After completion, verify the
returned config and actual logged learning-rate history. Inspect training-fit
performance first, then the existing validation set; keep the original
confirmation unchanged. Since that confirmation has now been inspected, a
new selected candidate needs a new untouched family holdout for a final
generalization claim. Do not promote solely because training loss fell.

## If lower rates do not improve fit

A capacity-controlled diagnostic is more informative than another epoch sweep.
The existing independent [adapter audit](adapter-capacity.md) establishes
rank-one-shaped exported factors. That is a plausible limitation, not proof
that rank alone caused geometry failures.

CoreWeave's October 5 post describes a programmable training API in limited
preview and illustrates `create_learner(model=model, lora=lora)`, explicit
optimizer control, and mixed SFT/RL updates. It is a possible route to an
explicitly configured capacity comparison. The illustrative blog does not
establish this account's access, Gemma4 support, allowed rank values, or a
working public client contract. Verify those before implementing or launching
a rank comparison; do not put a guessed rank field into Forge's generic
`experimental_config`.
[CoreWeave programmable training preview](https://coreweave.com/blog/closing-the-loop-between-inference-and-post-training).

With verified support, use a small fixed training-only memorization set and
compare rank one with one explicitly supported higher rank, holding examples,
step count, optimizer, template, and decoding fixed. If both fit, graduate to
the full reviewed data. If neither fits, inspect remote token/label alignment
and the learning objective before collecting more near-duplicate examples.
Keep this diagnostic separate from a claim of matching `gpt-6.1-sol`.

No inference, training job, or deployment was performed during this review.
The separately authorized LR workflow extension changed only the Forge client,
its durable submission wrapper, and focused tests. The original agent,
renderer, scorer, benchmark, and frozen evaluation evidence remain unchanged.
