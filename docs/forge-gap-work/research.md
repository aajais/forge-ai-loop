# Research for closing the Bufo scene-generation gap

**Execution update, 2026-10-15:** The comparison proposed below is now complete;
see [actual results](results.md). The selected checkpoint passed 7/80 confirmation
prompts versus the teacher's 65/80 and failed all 16 fixed in-sample diagnostic
prompts. Local audits verified SVG supervision and matching application prompts;
hosted trainer internals remain partly unexposed. The adapter uses rank 1. The
user has offered to increase rank; the supported configuration route and scope
are pending clarification. A rank-32 experiment would preserve data, epochs and
learning rate, and verify the resulting tensor shapes before attribution.
Lower-learning-rate tooling is being prepared as a separate alternative; no
follow-up job has been submitted. Rank and learning rate must not change together
in a comparison described as a rank-only test.

CoreWeave's [October 2026 AI Loop article](https://coreweave.com/blog/closing-the-loop-between-inference-and-post-training)
also describes a programmable training API in limited preview. Its illustrative
LoRA learner does not establish that this account can use a configurable-rank
route. The existing managed [fine-tune API](https://docs.coreweave.com/model-distillation/reference/management/fine-tunes/create-a-fine-tune)
explicitly documents learning rate; it does not document a rank field.

Reviewed 2026-10-15. Scope: public primary research, the authors' own research
blog, current CoreWeave documentation, and a read-only inspection of
`build_curriculum.py` and the implementation plan. The curriculum script was not
executed; no generations, reviews, training jobs, or runtime changes were made
for this research task.

The next defensible experiment is **more diverse, visually verified teacher
trajectories, followed by matched one- and three-epoch Forge SFT candidates**.
This tests a specific explanation for the existing result: ten targets did not
provide enough coverage of actions, contact, spatial relationships and tool use.
It does not establish that data size is the only bottleneck, or guarantee that
`google/gemma-4-26B-A4B-it` will match `gpt-6.1-sol`.

## Evidence and practical implications

| Primary source | Finding relevant to this project | Supported application and limitation |
|---|---|---|
| Agarwal et al., **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** (ICLR 2024; originally GKD, arXiv:2306.13649) | Teacher-only sequence training can encounter a different state distribution from student inference. GKD uses student-generated sequences with teacher distribution feedback. | Start with authentic teacher trajectories; if validation failures persist, collect student-state recovery demonstrations on development families. That would be correction SFT unless we implement the actual distribution objective. Cross-provider text completions alone do not supply the aligned teacher token distributions required by GKD. [Paper](https://arxiv.org/html/2306.13649v3) |
| Singh et al., **Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models** (2023/2024; ReST-EM, arXiv:2312.06585) | Iterating generation, correctness filtering and supervised training improved their math/code tasks. | Quality-filtering and iteration are useful design principles. Our first batch uses `gpt-6.1-sol` teacher outputs and explicit assistant image reviews; it is not the paper's student self-training algorithm. Pixel judges are also less objective than executable correctness checks. Retain rejected images to diagnose coverage gaps. [Paper](https://arxiv.org/html/2312.06585v2) |
| Biderman et al., **LoRA Learns Less and Forgets Less** (TMLR 2024, arXiv:2405.09673) | Low-rank settings underperformed full fine-tuning in some math/code regimes, with better retention outside the target domain. Results depended on rank, target modules and optimization; instruction tuning and continued pretraining differed. | A poor adapter is not proof of a fixed student ceiling. Inspect the actual adapter/training configuration if more data and epochs fail. This paper does not identify the right rank for this model or SVG scenes. [Paper](https://arxiv.org/html/2405.09673v2) |
| Schulman and Thinking Machines Lab, **LoRA Without Regret** (author research blog, 2025-09-29) | Their controlled studies found LoRA can match full fine-tuning in suitable post-training regimes, with sufficient capacity, appropriate learning rates and target layers; MLP/MoE coverage mattered. Large batches sometimes hurt. | Keep the initial batch small and change only epoch count. If needed, investigate supported LR settings and actual target modules before blaming rank alone. Their empirical LR ratios are not a prescription to multiply this service's defaults by ten. Their study did not test this Bufo task. [Research blog](https://thinkingmachines.ai/blog/lora/) |
| **CoreWeave Model Distillation: Fine-tuning** (current official documentation; page last modified 2026-10-06) | Managed fine-tuning supports dataset/base selection, epochs, batch size, peak LR or explicit schedule, and training on all or only the last assistant turn. Deployment records an exact hosted artifact. | Use the supported managed SFT controls and preserve actual artifact/job receipts. Set all assistant turns so the training objective includes SVG-producing tool requests, not only the final short reply. Do not assume arbitrary distillation losses or LoRA module controls are available merely because experimental configuration exists. [CoreWeave docs](https://docs.coreweave.com/model-distillation/studio/fine-tuning) |

## Proposed first comparison

1. Freeze the 128 training prompts, 24 validation prompts and 80 confirmation
   prompts before target generation. Preserve original heldout families as a
   regression set. Keep generation settings, character assets, tools, renderer and
   visual rubric identical across comparison arms.
2. Generate genuine `gpt-6.1-sol` trajectories on training and validation prompts.
   Bind reviews to the actual PNG hash and requirements. Accept a training target
   only after visually checking action/contact details and trajectory eligibility.
   A passing model judge or a friendly final sentence is not acceptance.
3. Aim for at least 80 accepted training examples with coverage across all 16
   training families. Treat this as an experiment-size choice, not a
   research-established threshold. If one family repeatedly fails teacher review,
   record the gap rather than filling it with duplicated or invented targets.
4. Submit two independent candidates from `google/gemma-4-26B-A4B-it`, sharing
   the exact reviewed dataset: one epoch and three epochs, batch size two,
   all assistant turns, with the same documented default LR policy. Do not change
   learning rate, prompting and dataset simultaneously. Preserve requested and
   service-resolved settings, token counts and immutable adapter versions.
5. Select on validation strict scene passes, then requirement coverage, then
   generation failures as the recorded tie-breakers. Also examine family-level
   results, render validity, identity, anatomy and tool-repair frequency. Lower
   training loss alone is not the user outcome.
6. Run the selected immutable adapter, the base ID and `gpt-6.1-sol` on confirmation
   only after selection. Report all attempts, paired outcomes and uncertainty.
   The five-percentage-point parity target in the plan is a target; a small point
   estimate difference does not establish statistical equivalence.

These are project-specific experimental choices inferred from the observed
failure modes and the research, not published guarantees of improvement.

## Read-only curriculum review

The generator specifies 16 training families × eight variants, four validation
families × six variants, and 16 confirmation families × five variants. Its checks
exclude original heldout family names from every new split and prevent overlaps
between new splits. It generates prompt requirements only, which is appropriate:
review labels and answers must come from actual subsequent runs.

No obvious blocking action/constraint contradiction was found in the inspected
script. The skate-park rail exception avoids hiding the required foot/board
contact. The small-character variant is intentionally difficult; image review
must confirm readable contact rather than infer it from the caption.

There are two statistical limits to preserve in reporting. First, 128 variants
represent 16 training families, not 128 independent skills; report accepted
coverage by family and requirement type. Second, the 24 validation examples cover
only four families, so the selected epoch count may be unstable. Confirmation
uncertainty should account for within-family dependence (for example, paired
resampling of entire scene families), alongside the raw per-prompt counts.
Matching family strings is necessary but not sufficient for semantic separation;
retained provenance and manual inspection remain useful.

## If the first comparison still misses

Use validation failures to pick the next falsifiable change. A train/validation
gap suggests overfitting or narrow coverage; weak training-set performance calls
for checking objective masks, tool-argument inclusion, actual updates, LR and
adapter configuration. For recurring student-specific bad states, collect real
teacher recovery demonstrations without modifying recorded earlier turns. Run
reward-driven RL only after its exact-token capture and continuation contracts
work on real trajectories. Keep those changes separate from SFT so any observed
improvement has an identifiable cause. Freeze another confirmation set if prior
confirmation results are used to design the next training iteration.
