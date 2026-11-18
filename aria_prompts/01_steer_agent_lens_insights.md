# Observe bufo failures in Agent Lens

Work in W&B entity `wandb-smle`, project `bufo-gen`. Use Agent Lens conversations
and their actual spans. Join the `visual_generation` Weave calls through the
`conversation_id` attribute / `output.generation.group_id`. Inspect `output.image`
against `output.references`; do not infer visual quality from SVG text alone.

Configure Insights categories for: generic frog instead of bufo, missing bufo,
wrong facial proportions, occluded or tiny character, missing requested objects,
wrong object counts, disconnected hands/props, wrong pose or expression, illegible
requested text, invalid SVG, and unsuccessful render self-correction.
Read `BufoIdentity` scorer feedback: bufo_present, similarity, prompt_alignment,
quality_failure, failure_modes and reason. Thresholds are provisional. Compare
these with the `Bufo identity calibration v1` annotation queue; human judgments
take priority when the automatic judge is too generous.

Analyze open_base (google/gemma-4-26B-A4B-it); compare with closed_quality_bar (gpt-6.1-sol) and open_tuned
(ft-google/gemma-4-26B-A4B-it) only when those runs actually exist on matching prompt IDs.
Separate prompt versions and character-asset versions. Current composition uses
a shared, original-derived vector character; this is a system improvement, not
evidence of learned model improvement. Do not use generic trace counts as
conversation counts. These probes are development data, not held-out evals.

Log `bufo-failure-clusters` as a W&B artifact of type `analysis` containing
`clusters.json` with this schema:

```json
{"clusters": [{"name": "category name", "theme": "theme label matching prompts", "size": 1,
 "conversation_ids": ["actual conversation ID"],
 "prompts": ["actual source prompt"], "evidence": "observed failure"}]}
```

Use only observed failures. Link image traces as well as source conversations and include the sample
window, backend/model IDs and actual sample sizes. If Insights is unavailable or
there is insufficient evidence, report that explicitly; do not fabricate clusters.
The next task consumes the versioned artifact, not local files.
