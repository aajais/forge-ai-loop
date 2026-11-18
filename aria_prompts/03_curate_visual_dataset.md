# Curate reviewed Bufo targets from production traces

Work in `wandb-smle/bufo-gen`. Use actual `invoke_agent` conversations and their
`render_svg` / `render_png` tool outputs. Inspect final PNGs, not the assistant's
claim that it drew the requested scene. SVG is a tool argument; the final assistant
reply is short text followed by the rendered image.

Start with the actual queue linked in `runs/forge-loop/pilot/review-queue.json`
(the pilot queue ID is `01a1403d-e1af-7922-aa5f-a553c849a728`) and new production
traces. The pilot comparison run is
https://wandb.ai/wandb-smle/bufo-gen/runs/q08be3sp . It contains paired google/gemma-4-26B-A4B-it/gpt-6.1-sol
images, assistant reviews, original Qwen-v1 and later gpt-6.1-sol scene-v2 judgments. These
judgments use different configurations and must not be merged into one score.

Identify recognizable Bufo identity, connected anatomy, visible hand/prop contact,
relative positions, composition and small-size readability. Check each exact scene
requirement against pixels; labels and captions do not prove an action happened.
Record missing features and uncertainty, even when the renderer succeeded. Preserve
all failed generations and judge errors in the observational dataset.

Partition by prompt family before collecting targets. `config/forge_prompts.yaml`
is the authoritative scene benchmark: 12 train, 4 validation and 100 heldout tasks.
Keep heldout/test IDs and all their families out of training, validation, reward
calibration and targeted prompt development. Original-emoji source datasets have
independent partitions; retain those as well. Never reinterpret the pilot's
train/validation results as a heldout evaluation.

For each candidate that you actually inspect, export one explicit review JSONL row:

```json
{"schema_version":1,"source_generation_id":"ACTUAL_GENERATION_ID","prompt_id":"ACTUAL_PROMPT_ID","decision":"accept","reviewer_kind":"aria_reviewed","reviewer_id":"ACTUAL_REVIEWER","reviewed_at":"ISO_8601_WITH_TIMEZONE","source_ref":"ACTUAL_TRACE_URL","evidence":"Specific visible observations, including any limitations","partition":"train","prompt_family":"ACTUAL_FROZEN_FAMILY"}
```

Use `reject` for bad targets. `aria_reviewed` means an actual ARIA-assisted review;
never relabel an assistant-only review as human or ARIA approval. Automated scorer
pass/fail alone is not acceptance. Keep conflicting reviewer judgments separate
until the owner resolves them explicitly; do not silently overwrite them.

Also record `visual_quality_pass`, an exact evidenced requirement checklist, image
SHA-256, and 0–4 similarity/anatomy/composition/readability labels when calibrating
the scene judge. Follow the existing calibration review schema in the pilot review
files. Visual acceptance and trajectory eligibility are separate: a good edited
image may be ineligible for SFT because prior assistant turns lack reviewed lineage.

Return the actual review file or versioned artifact reference, source trace IDs,
reviewer provenance and counts by split/model/failure category. To build SFT data,
join those decisions to the existing manifest and export captured trajectories via
`bufogen --stage export-reviewed`. Do not regenerate examples after assigning IDs.
Only authentic captured tool calls, original IDs/results, frozen instructions and
successful final replies qualify. Legacy synthetic calls, image-input trajectories,
external prior conversations, rejected or unreviewed targets are excluded from v1.

Create separate datasets for (a) observed successes/failures with images and labels,
(b) accepted gpt-6.1-sol training/validation trajectories, and (c) train-only RL environment
tasks with exact requirements and no teacher answers. Register actual source
artifacts and trace references. Never invent repaired images, tools, labels or
artifact versions. This curation task does not submit training or change routing.
