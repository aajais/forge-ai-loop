# Next-method research: reducing the geometry learning burden

Research date: 2026-10-20. This document proposes experiments; it does not report
new training or evaluation results. No paid jobs were started for this research.

The current iteration's reported validation result is 0/24 strict passes for
the fine-tuned candidate, compared with 1/24 for the base model and 21/24 for
`gpt-6.1-sol`. The reported training-fit diagnostic is also 0/16. These small
diagnostics do not establish that Gemma cannot learn the task. They do make
another large, similar SVG-data collection difficult to justify before locating
the remaining failure.

**Recommendation: finish the teacher-plan ablation first. If correct plans do
not translate into correct rendered contacts, test a compact, executable contact
representation before expanding the supervised corpus.** Any benefit from a
shared renderer must be measured separately from fine-tuning.

## First decision: teacher-plan ablation

The existing `plan_scene` tool exposes geometry but does not enforce it in the
SVG. A successful plan call is therefore not evidence of a correct image. The
parent experiment is testing whether giving Gemma a correct teacher plan changes
the outcome. Its results are not yet assumed here.

Use the same diagnostic prompts, SVG generation settings, renderer and scoring
for each ablation. Preserve original attempts and label the supplied teacher plan
as diagnostic assistance. Do not present it as Gemma's own planning or as a
deployable Gemma-only result.

Interpret the result as follows:

| Observation | Next action | What it would not establish |
|---|---|---|
| Teacher plans substantially improve contacts and visual success | Focus data on learning compact semantic plans; inspect whether Gemma's own plans recover the benefit | That the teacher-assisted result is an FT improvement |
| Teacher plans are correct, but SVG contacts remain wrong | Prioritize an executable representation that makes plan and render share coordinates | That increasing rank or dataset size is useless |
| Both teacher plans and resulting scenes fail | Diagnose plan semantics, prompt ambiguity and renderer constraints before training | A student-capacity conclusion |
| Plans render correctly but the image still fails composition or prompt alignment | Target the remaining semantic/visual failure modes; contact enforcement alone is insufficient | That a numerical geometry score measures visual success |

Keep this diagnostic set separate from future confirmation data. Once its
examples influence design choices, they are development evidence.

## Evidence and limits

The relevant literature suggests useful design principles, but none demonstrates
Sol-level Bufo generation through rank-1 Gemma fine-tuning:

- [LLM-grounded Diffusion](https://arxiv.org/abs/2305.13655) separates language
  interpretation into captioned bounding boxes and layout-guided rendering.
  It demonstrates improvements without training the language model. This
  supports testing the representation, while making attribution to the harness
  especially important.
- [VisProg](https://arxiv.org/abs/2211.11559) expresses compositional visual tasks
  as executable modular programs. Its reported gains do not require
  task-specific fine-tuning. The applicable idea is an executable intermediate
  representation, rather than a plan the renderer can ignore.
- [SketchGraphs](https://arxiv.org/abs/2007.08506) represents CAD sketches as
  primitives connected by geometric constraints. Shared contact anchors are a
  plausible adaptation of that idea; CAD results do not prove emoji quality.
- [StarVector](https://arxiv.org/abs/2312.11556) learns SVG generation with
  semantic primitives and a two-million-example dataset. Its scale and
  multimodal architecture differ materially from this project's small adapter
  experiments. It is evidence that SVG generation is learnable, not evidence
  that a few hundred additional rows will close this gap.
- [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649) addresses
  teacher/student distribution mismatch using teacher feedback on
  student-generated sequences and distribution-level losses. Teacher-corrected
  examples can address related exposure problems, but ordinary relabelled SFT
  must not be called an implementation of GKD.

The proposed renderer experiment below is an engineering inference from these
sources and our observed failure modes.

## Two realistic options

| Option | Hypothesis | Principal risk |
|---|---|---|
| More contact-focused SVG demonstrations | Additional examples teach both scene decisions and their coordinate realization | Long outputs and independent coordinates retain the current difficulty; no demonstrated training-set fit |
| Shared executable contact representation | Gemma can learn semantic choices when transforms and attachment geometry are deterministic | The tool itself may explain all improvement; abstraction may restrict artistic scope |

If the teacher-plan ablation shows that plan-to-SVG translation is already
reliable, option A becomes more attractive. Otherwise, begin with the smallest
option-B test below rather than committing to a large redesign.

## Compact generic representation

Use a scene graph with arbitrary named nodes, local coordinate systems, named
anchors and explicit relationships. Keep it generic enough to express new props
and backgrounds without adding one preset per prompt:

```json
{
  "canvas": [512, 512],
  "nodes": [
    {"id": "bufo", "kind": "bufo_rig", "transform": [1, 0, 0, 1, 180, 120]},
    {
      "id": "prop",
      "kind": "group",
      "transform": [1, 0, 0, 1, 330, 270],
      "anchors": {"grip": [0, 20]},
      "children": [
        {"kind": "rect", "x": 0, "y": 0, "width": 28, "height": 40,
         "fill": "#f6b739", "radius": 4},
        {"kind": "path", "d": "M28 8 C48 8 48 30 28 30",
         "fill": "none", "stroke": "#f6b739", "stroke_width": 6}
      ]
    }
  ],
  "constraints": [
    {"kind": "attach", "source": "bufo.left_hand",
     "target": "prop.grip", "offset": [0, 0], "move": "source_chain"}
  ],
  "paint_order": ["bufo.body", "prop", "bufo.left_arm", "bufo.left_hand"]
}
```

This is an illustrative proposal, not an implemented or validated API. The
example's prop and anchor would still require visual review; an arbitrary anchor
name cannot establish that the hand actually holds a recognizable handle.

The narrowed initial implementation proposal is in
[contact-renderer-design.md](contact-renderer-design.md): one attachment per
object in an acyclic dependency graph, translation-only attachment resolution
with declared rotation/scale preserved, and hands connected to resolved grips.
The broader graph discussion here does not require a general constraint solver,
automatic surface-normal alignment, or simultaneous two-anchor fitting.

Design constraints:

- A canonical Bufo rig supplies one connected head/body and named joints. The
  model chooses pose, placement and requested interactions. The rig must remain
  the same for every model.
- Props use circles, rectangles, paths, polygons, text and groups, with reusable
  user-defined local anchors. Avoid complete scene presets such as
  `perfect_gardener`.
- An attachment explicitly specifies the movable node or limb chain; the
  compiler applies coordinate transforms and solves that declared contact.
  Reject impossible/cyclic constraints with actionable diagnostics.
- Layer order is explicit. Coincident coordinates do not override visibility,
  occlusion, limb count or prompt-alignment scoring.
- Preserve freeform capability through bounded SVG/path groups for novel
  objects, clothing and backgrounds. The existing SVG sanitizer still applies.
  Freeform subtrees may expose named anchors but do not automatically receive
  semantic validation.
- Record the proportion of output using freeform geometry, token count and
  unsupported requests. If most output remains long SVG, the abstraction has
  not actually reduced the learning burden.
- The renderer must not read the original user prompt and silently choose a
  correct prop, action or composition. Those remain the model's decisions.

This changes the application from unrestricted SVG synthesis to constrained,
extensible scene composition. Evaluate that scope explicitly; it is not a claim
of unrestricted drawing equivalence.

## Smallest staged test

1. **Renderer check, no training:** create 12 hand-authored engineering fixtures
   covering one contact, two-contact chains, foreground occlusion and multiple
   objects. Inspect full-resolution and Slack-size PNGs. Keep these separate
   from authentic teacher trajectories.
2. **Interface diagnostic:** use 16 prompts across four relationship families
   and four variations. Run base Gemma and Sol through both the existing SVG
   interface and the proposed interface: 64 generations. Keep attempt count,
   sampling settings and scoring fixed. Record latency, tokens, validity,
   contacts, anatomy and prompt alignment.
3. **Decision gate:** if Sol cannot reliably express the scenes, fix the
   interface. If base Gemma already matches Sol, report an agent/tool gain;
   do not create a fictitious training gain. If Sol succeeds while Gemma makes
   identifiable semantic/parameter errors, collect a small learning diagnostic.
4. **Learning diagnostic:** collect approximately 24 actually executed,
   visually reviewed Sol trajectories and reserve eight additional prompts
   for diagnostic generalization. Train one supported managed candidate.
   Evaluate the exact training prompts separately from held-out prompts.
   A proposed predeclared threshold is at least 20/24 correct training scenes
   before scaling the corpus. This is a diagnostic gate, not statistical proof.
5. **If training fit is still poor:** inspect supervised token coverage,
   immutable deployment identity, loss and supported optimization controls.
   Do not infer that merely adding more data will fix failed training-set fit.
6. **If learnable:** expand to a balanced corpus, for example 96 reviewed cases
   spanning four relations, six prop categories and four composition variations.
   Hold out combinations and whole scene families. Add explicitly labelled
   student mistakes as user-side correction context followed by reviewed
   teacher actions; never treat failed assistant actions as positive targets.

The existing visual rubric can remain a continuity metric. Use blinded visual
review for contacts and occlusion, and deterministic geometry as a separate
diagnostic. All requested generations, including errors, remain in the
denominator. Freeze fresh confirmation families before candidate selection.

## Attribution and model naming

For the new shared interface, compare:

- `google/gemma-4-26B-A4B-it`
- `ft-google/gemma-4-26B-A4B-it`
- `gpt-6.1-sol`

Record adapter URI, training recipe and harness version as separate metadata.
The fine-tuning effect is:

`FT Gemma with new renderer − base Gemma with new renderer`

The harness effect is:

`base Gemma with new renderer − base Gemma with old renderer`

Do not compare FT/new-renderer against base/old-renderer and attribute the
difference to training. Teacher-assisted diagnostic plans are a third
intervention and must be labelled separately.

## Supported Forge and ART boundary

The current official
[fine-tuning guide](https://docs.coreweave.com/model-distillation/studio/fine-tuning)
documents dataset caps, batch size, epochs, scalar peak learning rate or
per-step learning-rate schedules, and assistant-turn supervision. The
[fine-tune creation API](https://docs.coreweave.com/model-distillation/reference/management/fine-tunes/create-a-fine-tune)
provides no verified rank-selection field. Keep the supported serverless
configuration and inspect the exported adapter rather than inferring rank
from a local environment variable or an arbitrary experimental key.

The current public
[ART ServerlessBackend source](https://github.com/OpenPipe/ART/blob/main/src/art/serverless/backend.py)
supports `cispo` and `ppo` training losses. It rejects custom `loss_fn_config`
and `adam_params`, and does not forward `lora_config` during registration.
Full teacher-distribution GKD is therefore not an established drop-in recipe
through the reviewed interface. Main-branch source is not proof of the deployed
service revision.

[CoreWeave's Model Distillation product](https://coreweave.com/products/coreweave-forge/model-distillation)
describes managed SFT on task examples followed by comparison against the current
model. This proposed sequence fits that product without claiming a custom
distillation loss.

[CoreWeave's Serverless RL guidance](https://www.coreweave.com/products/coreweave-forge/serverless-rl)
recommends improving prompts, tools and context first, and starting with SFT
when curated examples demonstrate the behavior. Once the compact representation
is learnable and rewards are trustworthy, a small genuine ART RL trial is
reasonable. Combine semantic/visual outcomes with geometry diagnostics: a policy
must not earn success merely by declaring coincident anchors for the wrong or
invisible objects.

The browser successfully retrieved the public HTML documentation and source.
Direct local Python HTTP retrieval failed with DNS resolution errors during this
read-only research. No authenticated job state was queried or changed here.
