# Conditional design: minimal contact-aware SVG renderer

Proposal only, 2026-10-20. Adoption depends on the pending teacher-plan diagnostic
and subsequent shared-interface evaluation. No implementation or result is
implied here. See [next-method-research.md](next-method-research.md) for controls.

**Scope: one geometry module, a fixture generator, and a separate experimental
agent hook.** Avoid a general constraint solver or a catalog of complete scenes.
The model authors arbitrary SVG fragments; the renderer shares their contact
coordinates and draws connected arms.

## Proposed tool contract

`render_scene` receives `bufo`, `objects`, `hands`, and optional `foreground`.
The initial canvas is 512×512.

- `bufo`: position and positive uniform scale for one canonical character rig.
- Each object: `id`, `svg`, local `anchors`, optional transform
  `[x, y, rotation_degrees, scale]`, and a layer (`background`, `clothing`,
  `prop`, or `foreground`).
- Each object may have **one attachment**: its local anchor, an already-resolved
  target anchor, and an optional world-space offset.
- Each hand: side, target anchor, bend direction and grip angle/width.

Illustrative attachment and hand references:

```json
{
  "attach": {"anchor": "tip", "to": "surface.contact", "offset": [0, 0]},
  "hand": {"side": "left", "to": "tool.grip", "bend": "outward",
           "grip_angle": 30, "grip_width": 10}
}
```

These are illustrative field values, not a complete implemented request.
`tool` and `surface` are model-authored fragments, not catalog items. Named
anchors are local `[x, y]` points. Omit identity transforms and empty fields to
keep generated arguments short. Reuse the prior scene JSON for conversational
edits instead of embedding another full SVG.

## Deterministic algorithm

1. Validate bounded SVG fragments, identifiers, finite coordinates and transforms.
2. Process objects in dependency order. Reject cycles, unknown targets or multiple
   attachments for one object.
3. Preserve each object's declared rotation and scale. Solve **translation only**
   so its chosen anchor coincides with the resolved target plus offset.
4. Resolve hand targets through those final transforms and draw connected arms
   from fixed rig shoulders, then hands/fingers around the declared grips.
5. Compose the ordered SVG and use the existing sanitizer and sandbox renderer.

A two-contact action is an acyclic chain: surface anchor → tool tip placement →
tool grip → hand endpoint. Both hands can reference different grips on the same
resolved prop. No simultaneous two-anchor solve, automatic rotation, scaling,
inverse pose search, reflection or shear is in scope.

For a tool meeting a surface, the model supplies the surface contact point and
tool rotation. Translation enforces point contact; it does not infer surface
normals or choose a semantically correct orientation. This supports arbitrary
target surfaces while leaving action understanding with the model.

## Rig and visibility

Use a reviewed canonical Bufo body/head asset with original arms removed or
separated; never draw new arms on top of existing arms. Each generated arm starts
at a fixed shoulder and ends at the chosen grip, with bounded curvature/bend.
Grip angle/width determines finger placement. Reject unreachable endpoints
instead of stretching limbs or silently moving the body.

Default order: background → body → clothing → arms → props → fingers.
Arbitrary clothing is a fragment in body-local coordinates; it is not
automatically deformed into sleeves. Explicit foreground fragments may occlude
anything. The renderer must not silently reorder them to satisfy a prompt.
Visual scoring still checks hidden contacts, anatomy and readability.

## Bounds, safety and errors

Apply versioned limits to scale, coordinates, fragment/path size and object count.
Check final transformed bounds and report contact residuals. These diagnostics
prove geometric facts only; a named `handle` anchor may be semantically wrong.
Reject unsupported transforms rather than approximating them.

Retain existing XML/SVG restrictions, including no entities, scripts, event
handlers or external references. Namespace permitted internal fragment IDs.
No new renderer may bypass current sanitization or sandboxed rasterization.
Return actionable errors for the normal agent repair loop. After its bounded
attempt policy, preserve failures; do not substitute a hidden template or Sol.

## Application parity and acceptance

The experimental hook supplies the same schema, rig, compiler, instructions and
retry limits to base Gemma, FT Gemma and Sol. Preserve production defaults and
old traces. Record component hashes and immutable adapter identity.

Keep PNG/SVG downloads, final text followed by PNG, emoji name and user-image
input. Trace the actual scene tool call before the final response. Preserve exact
scene revisions for follow-up edits; do not invent scene JSON for legacy SVGs.
The renderer never reads the user prompt to correct object/action choices.

Test hand-authored fixtures first, then shared-interface generations, then the
small training-fit experiment in the research note. Measure:

- base/new − base/old: interface effect;
- FT/new − base/new: fine-tuning effect;
- FT/new versus Sol/new: remaining quality gap.

Track freeform SVG token volume, total tokens, unsupported requests and every
failed generation. If novel props require per-prompt code or generated SVG
remains equally long, reject or revise the abstraction before scaling training.
