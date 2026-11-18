# Planned scene distillation: second improvement iteration

## Execution amendment: 2026-10-20

Steps 1–5 are complete. Development strict passes were base 1/24, reference
21/24, prior fine-tune 0/24, new one-epoch 0/24, and new three-epoch 0/24.
The frozen rule selected three epochs by its coverage tie-break; it then failed
all 16 exact contact-training prompts. A blinded assistant review of the
prespecified four cases per arm found reference 4/4 and all Gemma arms 0/4.

Defer step 6 and leave all 48 fresh confirmation outputs ungenerated.
The development result does not justify promoting either candidate. This
amendment preserves all original metrics and selection receipts; it does not
change the historical selection rule or relabel training evidence as held-out.

Run a smaller diagnostic before further training: on the fixed 16 training
contact requests, compare the selected adapter's ordinary generation with
continuation from the authentic successful teacher plan. Use temperature zero,
the same system/tools/render budget, and explicit mixed-source provenance for
the injected prefix. Inspect actual images as well as automated scores.
If correct plans still yield disconnected contacts, investigate an executable
shared geometry representation before scaling the corpus. Any harness gains
must be reported separately from fine-tuning gains.

The user wants the fine-tuned Gemma agent to approach `gpt-6.1-sol` quality and
explicitly authorizes improving the data and continuing real experiments.
Earlier paid-training and assistant-review authorization remains in effect.

## Evidence and intervention

The prior selected adapter passed 7/80 confirmation cases versus Sol's 65/80 and
failed 16/16 sampled training prompts. Its data contained 143 single-tool and
four two-tool examples, with no observable pre-render planning. Hidden teacher
reasoning is deliberately excluded from training captures. Typical failures
are disconnected grips, tools missing targets, and props covered by later layers.

Add a shared `plan_scene` tool producing a compact design artifact: main character
transform, contact anchors and paint order. It gives coordinate guidance, not a
semantic quality verdict. Keep the same tool available to all compared models.
Collect actual plan and render calls with the existing authenticated capture.
Never invent teacher actions or strip privileged context from training inputs.

The hypothesis is that explicit intermediate design supervision and graded
contact examples are more learnable than completed SVG imitation alone. This is
inspired by [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301), whose
results are on NLP tasks; this project does not reproduce its multi-task loss or
claim its results transfer automatically to visual composition.

## Execution

1. Implement and test the shared tool; freeze the updated agent version before
   collecting targets. Preserve original benchmark/data/checkpoint evidence.
2. Freeze 64 graded plain training requests in the existing 16 training families,
   plus 16 explicit correction requests sourced only from training-fit failures.
   Corrections include original SVG and labeled automatic criticism as user data;
   they do not pretend failed student assistant actions came from the teacher.
   Reuse 24 development prompts for fresh matched comparisons. Reserve 48 fresh
   confirmation prompts in 12 disjoint families before generating outputs.
3. Generate genuine teacher trajectories. Require actual planning before render,
   valid PNGs and explicit assistant image review for every accepted target.
   Review visible grips, prop-target contact, occlusion and character identity.
   Preserve rejected cases. Train only on reviewed training rows; validation
   targets remain in the validation partition.
4. Train independent managed Forge candidates at one and three epochs, batch two,
   all assistant turns, explicit peak LR 1e-4. Record actual adapter rank, scaling,
   schedules and immutable artifacts. The user confirmed higher rank is not
   supported for this serverless route; use supported controls. This combines data,
   agent and optimizer changes relative to the historical run, so attribute
   improvement only through the new matched comparisons.
5. Compare base, teacher, prior FT control and both candidates on the same updated
   agent and 24 development prompts. Select by strict passes, then requirement
   coverage, then fewer generation failures, then one epoch. Inspect a fixed
   blinded image sample. Separately check fit on training prompts.
6. Evaluate the selected immutable candidate with the base and teacher on the
   untouched 48-case confirmation set. Keep every failed attempt in denominators;
   publish actual paired PNGs, Weave scorers and uncertainty. Similarity alone or
   falling token loss is insufficient. The target remains a teacher gap within
   five percentage points; report honestly if unmet.

## Constraints

- Exact display IDs, with `ft-` prefix and checkpoint metadata separate.
- No hidden reasoning extraction, fabricated trajectories or automatic review labels.
- The provisional visual scorer can over-credit invisible contacts; manual visual
  review remains necessary. Scores never prevent the user from getting an image.
- No undocumented rank parameters, repeated ambiguous job submissions, or silent
  promotion of an unverified adapter. Existing deployed checkpoints remain intact.
- New agent/tool changes benefit all arms; measure base-versus-FT under that same
  agent to distinguish scaffolding gains from weight changes.
