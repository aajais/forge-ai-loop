# Contact renderer pilot implementation plan

> **For agentic workers:** Use subagent-driven development for the assigned
> independent tasks. The user already authorized autonomous execution and
> subagents for this project.

**Goal:** Test whether executable contacts make Bufo scene composition reliable
enough to teach with managed Forge distillation.

**Architecture:** A separate experimental compiler accepts arbitrary SVG
fragments, named anchors, explicit object attachments, and hand targets.
It resolves translations and draws connected limbs. It never reads the prompt
to correct the model's semantic choices. Production defaults stay frozen.

**Tech stack:** Existing Python, Pydantic, OpenAI Agents SDK, sanitized SVG,
serverless sandbox renderer, W&B and Weave.

**Spec:** `docs/forge-iteration2/contact-renderer-design.md`.

## Evidence and constraints

The completed teacher-plan diagnostic passed 1/16 ordinary versus 6/16 assisted
at temperature zero, with all 32 attempts rendered and scored. Correct plans
help but do not reliably produce correct SVG contacts. This supports testing
the compiler; it does not establish a fine-tuning gain.

- All models get identical tools, instructions, assets and attempt budgets.
- Display exact model IDs; fine-tuned label is
  `ft-google/gemma-4-26B-A4B-it`, with immutable artifact metadata separate.
- Keep the 48 reserved confirmation outputs ungenerated during prototyping.
- No scene presets, prompt-dependent substitutions, hidden teacher fallback,
  or automatic semantic repair.
- Keep generic SVG fragments for novel props and backgrounds.
- Keep the existing XML/SVG validation and serverless rendering boundary.
- Fixed 512×512 canvas, one canonical Bufo for this initial diagnostic.
- Only translation is solved for object attachments; rotation/scale are
  model choices. At most one attachment per object; reject cycles.
- Engineering fixtures are never represented as teacher trajectories.
- No unsupported rank settings or new broad training job before learnability
  evidence. A lower SVG-token burden is part of acceptance.

## Review focus

Unknown/cyclic anchors must yield actionable errors, not silently misplaced
objects. Rotated/scaled props must retain their declared shape. Grips must be
visibly connected and unobscured. Arbitrary SVG must not bypass existing
validation or inject reserved character definitions. Evidence must distinguish
compiler gains from model-training gains.

## Task 1: isolated compiler and real rendered fixtures

Own `docs/forge-contact-pilot/scene_renderer.py`,
`docs/forge-contact-pilot/fixtures.py`, and
`tests/test_contact_renderer.py`. No production source changes.

- [x] Write meaningful interface tests for `compile_scene(scene: Scene) -> dict`:
  one transformed contact, a surface→tool-tip→hand chain, two hands on one prop,
  a foot/object attachment, unknown/cyclic references, unsafe fragments,
  unreachable hands, and reserved-ID collisions.
- [x] Implement typed bounded scene inputs, model-authored fragments/anchors,
  object DAG resolution, fixed canonical shoulder/foot anchors, connected arm
  curves and visible fingers. Return `svg`, solved transforms/anchors and
  geometry diagnostics, never a semantic pass score.
- [x] Namespace fragment-local IDs and references. Keep one canonical body.
  Apply default background→body→clothing→arms→props→fingers ordering;
  explicitly supplied foreground fragments may occlude and remain scored.
- [x] Preserve model rotation and scale, solve only translation. Define all hand
  angle and width units unambiguously. Reject nonfinite inputs and impossible
  endpoints; do not move the body or change the requested object automatically.
- [x] Render 12 varied hand-authored fixtures through the existing Renderer;
  save exact scene JSON, compiled SVG, PNG, and hashes. Include turned props,
  two-contact chains, crossed arms, clothing and deliberate foreground
  occlusion. Inspect full-resolution and 64px views. Fix compiler defects
  before any model comparison.

## Task 2: matched experimental agent comparison

Own separate experimental runner files under `docs/forge-contact-pilot/`.
Use the compiler above and existing backend, renderer, telemetry and scoring
utilities. Do not mutate the production agent's tool contract.

- [x] Implement `render_scene` as a real tool, with tool output before the final
  friendly text and PNG. Capture actual instructions/tool schema/actions and
  content hashes; retain exact scene input for follow-up edits.
- [x] Freeze 16 diagnostic prompts across four relationship classes with four
  variations each, excluding reserved confirmation families. Freeze
  instructions, tools, assets, scoring and budgets before inference.
- [ ] Run base Gemma and Sol on both existing SVG and new scene interfaces:
  64 attempts. Count all errors, score actual PNGs, record tokens and latency.
  Perform intervention-blinded visual review.
- [ ] If Sol cannot use the interface reliably, fix it and repeat under a new
  version. If base Gemma already closes the gap, report a tool improvement.
  Only proceed to training if successful teacher programs expose meaningful
  student semantic/parameter errors.

## Task 3: small managed learnability trial, conditional

- [ ] Collect 24 authentic, visually reviewed teacher trajectories; reserve
  eight additional diagnostic prompts. Never export fabricated fixture traces.
- [ ] Freeze one supported managed training recipe and immutable data hashes;
  use the established transport/lifecycle guards and record actual adapter
  configuration. Do not reuse stale candidate state.
- [ ] Evaluate exact training prompts separately from held-out diagnostics.
  Require at least 20/24 training-scene passes before scaling data. This is
  a diagnostic threshold, not proof of Sol parity.
- [ ] If fit succeeds, expand reviewed data, compare base/FT/Sol on the identical
  compiler, and then design the next serverless RL experiment with calibrated
  visual rewards. Keep compiler effect and FT effect separate throughout.

No production promotion or claim of closing the performance gap follows merely
from successful rendering, loss reduction, job completion, or this plan.

## Verification before pilot inference

All 477 tests passed. The compiler has 31 focused tests; the experimental agent
has seven real-SDK interface tests, and the pilot protocol has six tests.
Two earlier fixture versions are retained: visual review rejected unclear
crossed wrists, and code review found an ambiguous fragment-ID namespace.
The current twelve sandbox-rendered fixtures passed assistant engineering
review at 512px and 64px; the intentionally occluded control is accepted only
for demonstrating its declared occlusion behavior.

Compiler SHA256:
`42d96f9d2c588113ba9fb2d9238052cace6a0630fa78ce427af579401857a897`.
The live Gemma schema-only probe was accepted; its truncated 32-token output
is not an image-quality observation. The 64-attempt pilot is now running
under `runs/forge-contact-pilot/comparison/pilot-protocol.json`.
