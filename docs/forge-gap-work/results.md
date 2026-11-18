# Forge Bufo improvement experiment — validation and confirmation results

Updated 2026-10-15. **The quality gap is not closed.** On the completed 80-prompt confirmation set, `gpt-6.1-sol` received 65 provisional strict passes, `google/gemma-4-26B-A4B-it` received two, and the selected one-epoch `ft-google/gemma-4-26B-A4B-it` received seven. The selected candidate's observed increase over the base is 6.25 percentage points, with a family-cluster 95% interval of [0, 13.75]; its remaining teacher gap is 72.5 points [57.5, 86.25]. This does not establish parity or a broad quality gain.

The validation rule selected the one-epoch checkpoint before confirmation started. The confirmation summary, paired analysis, and separate selected-checkpoint training-fit diagnostic are complete. Native confirmation publication finished with **240 feedback entries, three evaluations over 80 rows each, and zero publication errors**. The [complete W&B image comparison](https://wandb.ai/wandb-smle/bufo-gen/runs/504b5fe7) includes all 240 attempts and their actual PNGs where generation succeeded.

## What actually ran

The expanded curriculum generated 152 genuine `gpt-6.1-sol` trajectories: 128 training prompts across 16 scene families and 24 validation prompts across four other families. Assistant reviewers inspected actual PNGs against explicit requirements, preserving image hashes and visual evidence. Original reviews remain immutable. Seven explicit second visual adjudications changed one acceptance to rejection.

The final eligible source contains **124 training and 23 validation trajectories**, with five visual rejects and no additional trajectory-export failures waived. It includes authentic assistant/tool histories, not constructed text-only targets. Original heldout and new confirmation IDs and families were excluded. Source SHA256: `e8f2f4b23c30ae73bd8fd394ac340e0b6d299f3c277307b70ce1d2d3dc746984`.

The reviewed targets are published in the [W&B image-table run](https://wandb.ai/wandb-smle/bufo-gen/runs/ecbf90c5), artifact `wandb-smle/bufo-gen/bufo-gap-reviewed-targets:v0`, and a versioned Weave dataset. Receipts are in `runs/forge-gap/curriculum-publication.json` and `runs/forge-gap/training/source-receipt.json`.

Two independent managed Forge jobs imported identical source/upload bytes, trained from `google/gemma-4-26B-A4B-it` with batch size two and `assistant_turns=all`, and deployed real checkpoints. Requests omitted learning-rate overrides. Observed histories show a shared peak of 0.0002 with different-duration warmup/decay schedules; see [measured training schedules](training-metrics-note.md).

| Display model ID | Separate epoch metadata | Forge fine-tune ID | Immutable artifact suffix |
|---|---:|---|---|
| `ft-google/gemma-4-26B-A4B-it` | 1 | `ac29564f-fde2-45e8-a87a-9d22beaf2e20` | `bufo-gemma-gap-e1-20261015:v1` |
| `ft-google/gemma-4-26B-A4B-it` | 3 | `3d999308-e1b5-44ee-84ce-0eab850b456a` | `bufo-gemma-gap-e3-20261015:v3` |

Both artifact suffixes are under `wandb-artifact:///wandb-smle/bufo-gen/`. W&B display names use the model ID directly; epoch count, checkpoint, and dataset hash are separate metadata. Saved remote job IDs, storage names, and artifact versions remain unchanged.

## Completed validation: 24 prompts × four arms

All arms used the frozen generation environment and `bufo-scene-v2` rubric. The full comparison contains 96 scenario attempts, covering six variants in each of four validation families. These are not 96 independent scene families.

| Model ID | Epochs | Provisional strict passes / all attempts | Generation timeouts | Images scored | Mean requirement coverage, scored images only |
|---|---:|---:|---:|---:|---:|
| `gpt-6.1-sol` | — | 23/24 (95.83%) | 0 | 24 | 98.96% |
| `google/gemma-4-26B-A4B-it` | — | 0/24 (0%) | 0 | 24 | 47.63% |
| `ft-google/gemma-4-26B-A4B-it` | 1 | 2/24 (8.33%) | 0 | 24 | 49.78% |
| `ft-google/gemma-4-26B-A4B-it` | 3 | 0/24 (0%) | 7 | 17 | 47.45% |

The seven three-epoch failures are recorded as `APITimeoutError: generation failed`. They remain in the 24-attempt strict-pass denominator. Its requirement-coverage mean uses only the **17 scored images**, so that column must not be presented as its all-attempt success rate. There were no judge errors in this completed validation run.

The one-epoch candidate's observed strict-pass difference from the teacher is still **87.5 percentage points**. Its two additional passes relative to the base are a small validation observation, not evidence of broad or statistically established improvement.

The frozen ranking rule was: strict passes, then scored-only requirement coverage, then fewer generation errors, then one epoch. It selected `wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-gap-e1-20261015:v1`. The selection receipt is `runs/forge-gap/selection.json`, with receipt SHA256 field `b3addb54d81a6657d633cbb3f4fed44325fbcfee627861739ff0c23002b3e691`; it binds validation evidence and candidate state hashes. Its `confirmation_not_started` field records the state **at selection time**, not current run status.

Inspect the [W&B validation image comparison](https://wandb.ai/wandb-smle/bufo-gen/runs/62ac1cdf) and the [96-call Weave annotation queue](https://wandb.ai/wandb-smle/bufo-gen/weave/annotation-queues/01a1410a-3417-7acf-b900-5cde97449b28). The queue receipt is `runs/forge-gap/validation-candidates/review-queue.json`. These provide material for user/ARIA review and future curation; their availability does not imply human review is complete.

## Completed confirmation: 80 prompts × three arms

Confirmation used 16 separately frozen scene families with five prompts each. Only the checkpoint selected from validation was evaluated alongside the base and teacher; the three-epoch checkpoint was not selected or retuned using confirmation results.

| Model ID | Epochs | Provisional strict passes / all attempts | Generation failures | Images scored | Mean requirement coverage, scored images only |
|---|---:|---:|---:|---:|---:|
| `gpt-6.1-sol` | — | 65/80 (81.25%) | 0 | 80 | 95.23% |
| `google/gemma-4-26B-A4B-it` | — | 2/80 (2.50%) | 4 | 76 | 52.89% |
| `ft-google/gemma-4-26B-A4B-it` | 1 | 7/80 (8.75%) | 0 | 80 | 51.44% |

All 80 selected-checkpoint generations rendered valid images; the base's four generation failures remain in its 80-attempt pass denominator. There were zero judge errors. The selected checkpoint's **scored-only requirement coverage was lower than the base's** (51.44% versus 52.89%), and those means use different available-image denominators, 80 and 76. More strict passes and fewer generation failures therefore do not justify claiming better scene quality across the board.

| Paired all-attempt strict-pass comparison | Observed difference | Approximate 95% interval |
|---|---:|---:|
| Selected checkpoint minus base | +6.25 percentage points | [0, 13.75] points |
| Teacher minus selected checkpoint | 72.50 percentage points | [57.50, 86.25] points |

The analysis resampled **16 paired family clusters**, with 20,000 percentile-bootstrap replicates and seed `20261015`. It preserves pairing within each prompt and dependence among prompts from the same family. The improvement interval touches zero; it does not support an unqualified population-improvement claim. The teacher-gap interval remains far beyond the five-point parity target. This is still an approximate interval over purpose-built families, with one scenario result per prompt and no estimate of within-prompt sampling variability. Shared teacher/judge model bias is not accounted for by the bootstrap.

Final local evidence: `runs/forge-gap/confirmation-selected/summary.json` and `runs/forge-gap/paired-results.json`. The latter binds all confirmation files and the selection receipt, identifies the exact selected artifact, and records both failed five-point parity checks. Publication receipts are `confirmation-selected/weave-publication.json` and `confirmation-visualization.json`; the W&B evidence artifact is `wandb-smle/bufo-gen/bufo-gap-confirmation-evidence:v0`. Native evaluations replay the saved generation outcomes: their `model_latency` is replay time, while actual generation time remains in each row's `latency_s`.

## Visual checks and measurement limits

The automatic scorer uses **`gpt-6.1-sol`, the same model ID as the teacher**. Judgments may therefore share model preferences or errors with teacher generation; they are not an independent quality authority. The rubric is explicitly marked uncalibrated. Similarity alone is insufficient: a recognizable Bufo can still fail the requested action or contact geometry.

A fixed blind diagnostic sampled the base variant of each validation family, with all four arms retained: 16 attempts. Model identities and judge scores were hidden from its assistant reviewer. The teacher passed 4/4; the base and both fine-tuned candidates passed 0/4. All 13 available images showed recognizable, connected Bufos, but prop identity, grip, and contact requirements often failed. Three missing three-epoch generations remained failures. Four cases per arm cannot calibrate population performance or replace the full selection rule. See the [blind diagnostic](validation-blind-diagnostic.md) and local `runs/forge-gap/validation-blind-review/comparison.html`.

A separate **non-blind, post-score audit** inspected exactly the one-epoch candidate's two automatic passes. It found rough composition, unclear active scoop placement, and oversized/separated props. This was a deliberately selected qualitative audit, not a random sample, independent calibration, training-target acceptance, or score override. It illustrates that satisfying a checklist may still fall short of the full prompt and desired visual finish. Original observations and PNG hashes are in `runs/forge-gap/validation-positive-quality-audit.json`.

## SVG tool supervision: measured locally, hosted details still limited

The [tool-loss audit](tool-loss-audit.md) reconstructed the actual Forge upload and matched its saved source/upload hashes. All 147 accepted entries retain their full messages and tool schemas, including 151 `render_svg` calls with SVG arguments; both jobs requested `assistant_turns=all`. There is no upload evidence that training accidentally omitted the geometry and learned only the final short reply.

Using the pinned local ART preprocessing and cached Gemma4 tokenizer, the audit found **325,790 supervised SVG-position tokens, zero masked SVG positions, and zero missing SVG payloads**. Full sequences ranged from 3,799 to 6,563 tokens. This certifies the measured local masking behavior, not the hosted trainer's exact labels, template revision, or sequence policy. Remote job reads report 124 training and 23 validation examples but do not expose effective maximum sequence length, dropped examples, or supervised-token counts.

The actual 62/186 logged updates are inconsistent with a specific 4,096-token whole-example-dropping hypothesis under the inspected local implementation; they do not rule out every possible remote truncation or partial-drop policy. Falling token loss is thus real evidence of optimization, but not a measurement of hands gripping props or geometry satisfying a prompt.

The completed training-fit diagnostic generated fresh images from the selected one-epoch checkpoint on 16 training prompts. It produced **16 valid PNGs, zero generation or judge errors, zero provisional strict passes out of 16, and 40.625% mean requirement coverage**. Native Weave publication records 16 feedback entries and one evaluation over all 16 rows, with no publication errors. Evidence is in `runs/forge-gap/training-fit-selected/summary.json` and `weave-publication.json`. This is a diagnostic on training prompts, not another heldout benchmark. Poor scene fit on these prompts weakens a simple explanation based only on unseen-family generalization; it does not by itself isolate capacity, optimization, or metric failure.

## Research interpretation and next decision

Actual one-epoch adapter configuration and safetensor shapes establish a **rank-1 LoRA parameterization per adapted matrix**, including each MoE expert. Attention and MLP/MoE modules are targeted. Rank is a plausible capacity hypothesis, **not a demonstrated cause** of the remaining gap; neither more epochs nor a hypothetical larger rank guarantees teacher-level performance. A verified client-configurable higher-rank managed Forge route was not found in the inspected interfaces. See [adapter capacity evidence and supported controls](adapter-capacity.md).

Research supports testing distinct explanations rather than promising parity:

- [On-policy distillation / GKD](https://arxiv.org/html/2306.13649v3) motivates studying student-state errors. Authentic teacher correction demonstrations would be correction SFT unless the paper's distribution objective is actually implemented.
- [ReST-EM](https://arxiv.org/html/2312.06585v2) supports iterative generation and filtering as an experimental direction; these teacher-produced targets are not student self-training under that algorithm.
- [LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2) and [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) motivate controlled capacity, learning-rate, and target-module investigations, with task-dependent results.
- [CoreWeave managed fine-tuning](https://docs.coreweave.com/model-distillation/studio/fine-tuning) documents the product path used here. Unsupported rank fields or objectives must not be inferred from an unspecified experimental configuration object.

The user has offered to increase adapter rank. The next proposed experiment is rank 32 with the same reviewed data, one epoch, batch size two, and recorded learning-rate schedule. Its supported configuration route and whether it affects existing deployments are awaiting clarification; no new training job has been submitted. Actual rank, alpha/scaling and target-module metadata must be verified before interpreting the comparison. Optional learning-rate support and separate runners are being prepared for a later controlled test, not combined with this capacity intervention.

The image metric also needs review against full requested actions, including active contact and visual finish. The completed three-epoch result does not support blindly adding more epochs. If confirmation failures inform the next intervention, freeze a fresh confirmation set before evaluating it. The Forge observe–curate–train–evaluate workflow has produced auditable evidence; it has not produced the requested quality parity. Separate Serverless RL execution evidence is documented in [the RL live-results note](rl-live-results.md), not attributed to these SFT candidates.

The local showcase is `runs/forge-gap/showcase/index.html`. It binds the selected one-epoch Forge state and all 152 final adjudicated reviews, keeps the earlier Serverless RL pilot lineage separate, and labels the gallery as a partial image sample. Metrics remain provisional. Explicit validation and confirmation links point to the complete published W&B image tables.

Verification: the full local suite passed **364 tests** after adding optional LR
controls and guarded follow-up runners. Independent evidence review checked all
493 retained benchmark PNG hashes, 152 final reviews, and 104 displayed showcase
images. This validates the application and evidence handling, not model parity.
