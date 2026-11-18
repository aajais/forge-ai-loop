# Results history

Every experiment, in order, with its numbers and evidence. Strict pass means the judge confirmed
Bufo identity and every listed requirement. Failed generations count as failures.

## 1. Gemma pilot (2026-10-15)

Student `google/gemma-4-26B-A4B-it`, 10 assistant-reviewed SFT targets, 100 heldout prompts,
SceneQuality judge.

| Arm | Strict pass |
|---|---:|
| Base Gemma | 1/100 |
| SFT pilot (`bufo-gemma-sol-pilot-20261015:v1`) | 1/100 |
| gpt-6.1-sol | 71/100 |

Serverless RL continuation from the pilot completed one optimizer step
(`bufo-gemma-sol-pilot-20261015:v2`); no quality change was measured.
Evidence: [live results](../forge-loop-work/live-results.md),
[W&B image comparison](https://wandb.ai/wandb-smle/bufo-gen/runs/rqbi08g1),
[RL live results](../forge-gap-work/rl-live-results.md).

## 2. Gemma quality-gap experiment (2026-10-16)

124 reviewed teacher examples, 23 validation targets, one vs three epochs, 80 new confirmation
prompts, SceneQuality judge.

| Arm | Strict pass (80) |
|---|---:|
| Base Gemma | 2/80 |
| SFT 1 epoch (`bufo-gemma-gap-e1-20261015:v1`, selected) | 7/80 |
| gpt-6.1-sol | 65/80 |

Gain 6.25 points, 95% family bootstrap 0 to 13.75. Requirement coverage did not improve.
Evidence: [results](../forge-gap-work/results.md), [research](../forge-gap-work/research.md),
[runbook](../forge-gap-work/evaluation-runbook.md),
[confirmation comparison](https://wandb.ai/wandb-smle/bufo-gen/runs/504b5fe7).

## 3. Judge recalibration (Stage A, 2026-10-21/22)

The scene judge had used the teacher or the student family. Replaced by `gpt-6-astra` with the
teacher's PNG as reference, structured output, and three conservatively merged samples. Rubric
v1 failed the user's blinded labels (accepted implied contact and heavy-lidded eyes as closed);
v5 reached agreement 0.875 on a fresh blinded packet. Under v5, Sol run 2 passed 154/176 and
every existing Gemma student 0/48.
Evidence: [calibration v1](../distill/calibration-v1.md),
[design](../superpowers/specs/2026-10-21-svg-distillation-design.md).

## 4. Stage B: teacher data at scale (2026-10-23)

1,024 teacher prompts, 1,021 generations, 911 accepted by astra v5 (113 rejected). Forge SFT sweep
on Gemma: `sft_j1` (3 epochs) and `sft_j3` (1 epoch) both 0/48 validation; no winner selected.
Heldout confirmation base Gemma 1/80 vs Sol 60/80 (gap -73.8 points).
Evidence: [Stage B results](../distill/stage-b-results.md), [progress](../distill/stage-b-progress.md),
funnel `runs/distill/stage-b/export/funnel.json`.

## 5. Qwen students on the same data (2026-10-23/24, current)

Same 911 rows, student `Qwen/Qwen3.8-27B` (Forge LoRA rank 8). Validation: 48 prompts in 4
families, frozen runtime (thinking on, 20k tokens, temperature 0.7), astra v5 k=3.

| Arm | Checkpoint | Strict pass | Coverage | Gen. errors | Median latency |
|---|---|---:|---:|---:|---:|
| Base | Qwen/Qwen3.8-27B | 17/48 | 0.79 | 10 | 919 s |
| SFT 1 epoch | `bufo-stageb-sft-q3-20261023:v1` | 30/48 | 0.91 | 2 | 618 s |
| SFT 3 epochs | `bufo-stageb-sft-q1-20261023:v3` | 27/48 | 0.88 | 5 | 534 s |
| SFT + RL | `bufo-stageb-sft-q3-20261023:v6` | 30/48 | 0.89 | 3 | 455 s |
| Teacher (historical) | gpt-6.1-sol, 6k tokens | 39/48 | 0.95 | 0 | 43 s |

Paired base vs SFT 1 epoch: -0.271 (95% CI -0.333 to -0.208; negative favors SFT). Base vs SFT 3
epochs: -0.208 (-0.333 to -0.083). The teacher ran with a different output budget and client, so
no parity interval is reported. A Nemotron 3.5 Lightning student was also configured; see
`config/models.yaml`.

Evidence: `config/forge-demo-comparison.json` (pinned digests),
`runs/forge-demo/comparison/` (portable HTML with all 48 attempts per arm),
[Weave comparison](https://forge.coreweave.com/wandb/wandb-smle/bufo-gen/weave/compare-evaluations?view=comparison_2026-10-24_22-07-09-875),
[Forge demo report](https://forge.coreweave.com/wandb/wandb-smle/bufo-gen/reports/Bufo-Forge-AI-loop-demo--VmlldzoxODE3NjU5MQ),
[post-training changes](../forge-demo-work/post-training-changes-2026-10-26.md).

## Open items

- Blinded human confirmation of the Qwen validation result (queue `bufo review confirmation`).
- Heldout run for the selected Qwen checkpoint; validation was used for selection.
- The release candidate in `runs/forge-demo/release-candidate.json` is prepared, not promoted.
- Forge proxy smoke on the routed Qwen target has not been run.
