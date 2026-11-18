# Gemma quality improvement implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for scoped implementation and independent review. Preserve existing work and saved experiment evidence.

**Goal:** Use exact model IDs in the product and improve `google/gemma-4-26B-A4B-it` toward `gpt-6.1-sol` on Bufo scene generation through real Forge training and controlled evaluation.

**Architecture:** Keep the existing generation/rendering/scoring workflow. Separate display names from immutable serving identifiers, expand the reviewed teacher curriculum, select training settings on validation data, and measure the selected adapter on a newly frozen confirmation set. Investigate the concrete ART/Gemma capture issue independently; do not let an unresolved RL integration prevent supported SFT experiments.

**Tech stack:** Existing Python/OpenAI Agents/cwsandbox, Weave/W&B, managed CoreWeave Forge SFT, ART Serverless RL.

**Spec:** User request on 2026-10-15: model IDs everywhere, `ft-` prefix for fine-tuned display, research papers/blogs/CoreWeave docs, continue until meaningful quality improvement is demonstrated or an evidenced external dependency prevents progress. Earlier authorization covers paid runs, assistant image review, and subagents.

## Global constraints

- Use `google/gemma-4-26B-A4B-it`, `gpt-6.1-sol`, and `ft-google/gemma-4-26B-A4B-it` for current displays. Preserve actual adapter URI/version separately.
- Never rename immutable model identifiers sent to inference or historical evidence to manufacture consistency.
- Train through Forge Model Distillation. Review actual rendered images before accepting teacher targets; keep assistant provenance explicit.
- Original 100 test cases remain a regression benchmark, not training data. Because their results informed this iteration, freeze a new family-disjoint confirmation suite before training and evaluate it only after validation selection.
- Keep model generation policy, tool budget, references and judge identical across comparison arms. Distinguish model improvement from additional inference-time search.
- No fabricated trajectories, sampled token IDs, log probabilities, labels, checkpoints or parity claims.
- Target proximity is a gap of at most five percentage points in strict pass rate on a matched comparison, with paired uncertainty reported; this is a target, not a promise or a claim supported by a tiny sample.

## Review focus

- Environment model overrides must propagate to labels, status, chat history, traces, plots and reports without changing routing IDs.
- Family/prompt/trajectory leakage across train, validation and confirmation must fail validation.
- Every accepted target must bind to its actual source PNG, model, tool actions, requirements and reviewer; judge-only filtering is not assistant review.
- Interrupted generation/training cannot silently duplicate accepted examples or paid jobs.
- Preserve failure denominators and avoid choosing candidates on the final confirmation set.

## Tasks

- [x] **1. Model display naming.** Centralize resolved IDs and `ft-` display labels, retain adapter metadata, update active UI/report/trace display paths. Test overrides and historical rendering; independent review.
- [x] **2. Research and data expansion.** Record primary-source evidence and platform support. Build approximately 128 varied teacher prompts plus separate validation and confirmation families. Generate actual `gpt-6.1-sol` tool trajectories, score them, inspect PNGs, and accept only verified targets. Aim for at least 80 accepted training examples; preserve every failure and rejected sample.
- [x] **3. Managed training comparison.** Import reviewed data with hashes and split checks. Train separate one- and three-epoch candidates from the exact base model using supported Forge configuration, all assistant turns, and durable job receipts. Compare on the validation set with the base model and teacher; choose without opening confirmation results.
- [ ] **4. Confirm improvement.** Run the selected immutable adapter and both reference models under one frozen policy on the new confirmation suite. Publish actual PNG comparisons, per-requirement metrics, strict passes and paired uncertainty. Update the UI's fine-tuned choice only to a verified actual adapter; retain old versions in evidence.
- [x] **5. Resolve RL compatibility where supported.** Test documented upstream exact-token metadata support using tiny genuine exchanges in isolation. Only update the runtime and submit a bounded optimizer job after a complete real trajectory passes local tokenization and independent review. Preserve SFT lineage and previous failures.
- [ ] **6. Review and handoff.** Run relevant tests, inspect generated images, verify real deployed inference and publish a truthful research/results report. If the gap remains, document the next falsifiable experiment rather than claiming completion.

## Initial evidence and rationale

The previous 10-example, one-epoch SFT run achieved 1/100 strict scene passes, as did the base model; `gpt-6.1-sol` achieved 71/100. Identity was usually recognizable. The dominant student failure was prompt fulfillment (94 baseline, 96 SFT rows flagged), suggesting that action/contact/composition coverage deserves targeted data rather than further character-only prompting.

- [ReST EM](https://arxiv.org/abs/2312.06585): iterated generation/filtering/SFT motivates quality-filtered data, but its math/code results do not establish visual-agent efficacy. This iteration uses teacher targets and explicit visual reviews, not a claim to implement the full paper.
- [Generalized knowledge distillation](https://arxiv.org/abs/2306.13649): student-state distribution mismatch motivates later correction data or on-policy work. Managed sequence SFT is not equivalent to its token-distribution objective.
- [Forge fine-tuning](https://docs.coreweave.com/model-distillation/studio/fine-tuning): supports dataset selection, epochs, batch size, learning-rate configuration and all-assistant-turn training; deployed artifacts must be evaluated before routing changes.

## Execution evidence

- Exact model IDs implemented and independently reviewed; existing W&B comparison relabeled without changing its results. Full suite after RL state fixes: 306 passed.
- Frozen curriculum: 128 train, 24 validation, 80 untouched confirmation prompts. All 152 teacher trajectories rendered and scored; 143 automatic provisional passes, zero judge errors.
- All 152 actual PNGs received explicit assistant review. A second contact-crop review of nine judge disagreements rejected one additional training target: final intended dataset is 124 train and 23 validation examples. Original blinded decisions are retained separately from adjudications.
- ART Dynamo exact-token metadata and inference routing fixes independently reviewed. Four real rollouts completed one optimizer update from the original SFT checkpoint; immutable v2 was verified and served a real agent PNG. Registry linking failed separately; original failure receipts remain intact. No RL quality improvement is claimed. See `docs/forge-gap-work/rl-live-results.md`.
- Managed one-epoch and three-epoch candidates deployed as immutable e1:v1 and e3:v3 artifacts. Fresh matched 24-prompt, four-arm validation finished and all 96 outcomes plus four native Weave evaluations were published. Passes: base 0/24, teacher 23/24, e1 2/24, e3 0/24; e3 had seven generation timeouts. The preregistered rule selected e1 before starting the 80-prompt confirmation run. No parity or generalization claim follows from selection.
- Actual e1 adapter config and tensor headers establish rank-1 parameterization. No public managed API rank override was verified; this is a capacity hypothesis, not an established explanation for quality.
- Confirmation generation and scoring are complete: base 2/80, teacher 65/80,
  selected e1 7/80. The teacher gap is 72.5 percentage points. Paired family
  bootstrap intervals and failures are preserved in `paired-results.json`.
  The fixed in-sample training-fit diagnostic is complete: 0/16 strict passes,
  40.625% scored requirement coverage, all 16 valid images. This does not support
  calling the quality objective complete.
- Confirmation publication completed: 240 feedback entries, three native Weave
  evaluations and no publication errors. The complete paired PNG comparison is
  W&B run `504b5fe7`; the local showcase and assembly hashes were refreshed.
- Optional explicit-LR workflow support and guarded follow-up runners are ready;
  no follow-up training jobs have been submitted. Full test suite after these
  additions and publication-recovery fixes: 364 passed. Higher-rank job
  configuration remains pending.

## Next controlled experiment — pending supported rank configuration

The user offered to increase adapter rank after seeing the rank-1 evidence.
Rank 32 was requested, together with the exact supported configuration route
and whether the change affects only new jobs. No follow-up training job has
been submitted. Existing immutable deployments must remain valid controls.

1. Reuse the exact reviewed 124-training/23-validation source and its upload hash.
2. Hold one epoch, batch size two, all assistant turns, generation tools and
   sampling policy fixed. Record the requested and actual peak LR (the existing
   control peaked at 2e-4). Do not combine a rank intervention with a LR sweep.
3. Verify new adapter config and tensor shapes: rank, alpha, target modules and
   rsLoRA flag. Alpha/r scaling is part of the comparison; if it changes, describe
   a capacity/configuration intervention rather than claiming rank alone caused
   the result. Do not send undocumented experimental fields.
4. Evaluate the fixed 16 training-fit prompts and 24 validation prompts. Preserve
   all attempts, actual PNGs, coverage and visual review disagreements. Selection
   may use validation only; training-fit is diagnostic.
5. If selecting a new checkpoint, freeze a fresh family-disjoint confirmation
   set before measuring final performance. The completed 80-prompt confirmation
   has informed this investigation and is now historical evidence.

Optional learning-rate support and separate guarded experiment runners are being
prepared while the rank route is clarified. Their existence is not evidence that
a job ran. Proposed lower-rate candidates (2e-5 and 1e-4) are an alternative
experiment, not simultaneous changes to the rank comparison.
