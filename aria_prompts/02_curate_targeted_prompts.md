# Curate prompts targeting observed failures

Work in `wandb-smle/bufo-gen`. Read the latest `bufo-failure-clusters`
artifact and the versioned `bufo-aria-prompts` artifact. Generate 30 diverse training
prompts covering the weakest observed categories. Preserve the green bufo identity
and use #2374FF / #0541E9 only on accessories, props, or backgrounds.

Log a `bufo-curated-prompts` artifact of type `prompts`. It must contain
`prompts.yaml`: a YAML list of objects with `id`, `prompt`, `theme`, and `cluster`.
Use IDs starting with `aria-train-`, never `p` or `heldout-`. Include simple and
complex scenes, requested text, and varied compositions. Theme values must match
the failure artifact's theme labels. Do not copy the held-out evaluation prompts.

Also include `provenance.json` with the exact source failure-cluster artifact
version, category coverage, and conversation IDs used. Register the failure-cluster
artifact as an input to the W&B run that logs the curated prompt artifact so the
lineage graph connects them. Return the full versioned artifact reference.

This task is the prompt-expansion step only. Use the companion visual-dataset
task to curate actual images and production traces. Do not turn failed SVGs into
positive supervised targets or launch training.
