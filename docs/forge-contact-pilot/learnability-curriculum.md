# Contact-tool learnability curriculum

`learnability-prompts.json` proposes **24 training prompts and eight held-out diagnostic prompts**. These are requests and review criteria, not training targets. No teacher generations, reviews, or training jobs have been produced by this preparation step.

| Relation class | Train | Diagnostic | What must transfer |
|---|---:|---:|---|
| `single_grip` | 6 | 2 | Put a visible hand on the correct part of a novel prop. |
| `two_hand_grip` | 6 | 2 | Place both hands at distinct, appropriate contacts on one prop. |
| `tool_target_contact` | 6 | 2 | Connect hand to tool and tool to its requested target. |
| `supported_object` | 6 | 2 | Preserve hand contact while aligning an object with its support. |

All 32 task-specific families are distinct. Diagnostic families are tambourine holding, suitcase carrying, skateboard display, picture-frame display, pizza cutting, wig combing, clock steadying, and juice-carton steadying. They share relation classes with training but use different tasks and props. This is a small within-relation transfer diagnostic, not evidence of unrestricted scene-generation quality.

Each prompt asks for one recognizable canonical full-body Bufo, a plain pale background, and visible face, feet, and contacts. Requirements separate identity/background checks from individual visible object and contact checks. Colors make the requested parts distinguishable. Props include curved handles, open frames, irregular silhouettes, multiple grips, wheels, and target surfaces. The model must invent its own SVG fragments, local anchors, and transforms; there are no coordinates, fragment templates, or scene presets in the curriculum.

The existing 16 pilot prompts are not repeated. The reserved confirmation families remain excluded: `window_washing`, `gift_wrapping`, `pumpkin_carving`, `domino_setup`, `fishing_pier`, `mortar_grinding`, `camping_flashlight`, `ticket_inspection`, `vacuum_cleaning`, `rope_bridge`, `lemon_squeezing`, and `stamp_printing`. Their prompts and results must not inform replacements.

Collect one initial teacher attempt per training row under the frozen chosen contact-agent version. Preserve failures and rejected images. An authentic, valid tool trajectory alone does not establish visual correctness: each accepted target needs an explicit PNG-bound assistant review satisfying every visible requirement. The eight `heldout` rows are diagnostics only and must never enter training uploads or teacher-target selection.

If fewer than 24 initial training outputs qualify, pause corpus assembly and record the shortfall. Any later replacements need separately recorded collection attempts and review decisions, using training families only and without consulting diagnostic results. Do not silently reroll until each row succeeds or relabel rejected images.

The proposed first experiment remains a small fit test: 24 approved teacher trajectories, one fixed candidate recipe, and separate evaluation of the training requests and eight diagnostics. Keep unchanged base-model and teacher comparisons under the same compiler, tool schema, instructions, and review rubric. A shared-renderer improvement is a harness effect; only the adapted model's improvement over its own unchanged baseline under that same harness can be attributed to fine-tuning.
