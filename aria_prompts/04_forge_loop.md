# Inspect the Forge Bufo AI Loop and propose the next curation batch

Use the saved evidence in `runs/forge-loop/` and the local `loop-report.json`.
Follow Run → Observe → Curate → Improve → Evaluate:

1. Run: inspect real google/gemma-4-26B-A4B-it, gpt-6.1-sol, ft-google/gemma-4-26B-A4B-it and (when present) SFT+RL traces.
   Forge proxy traffic requests the `bufo-gen` alias with `wandb.entity` and stable
   `wandb.thread_id`; direct backend traces remain distinct. Do not infer the actual
   served model from an alias when no resolved model was recorded.
2. Observe: compare PNGs and visible requirement satisfaction. The initial 32-image
   pilot used Qwen scene-v1; the grounded scene-v2 gpt-6.1-sol judge has a separate
   configuration and 32 assistant-reviewed development observations. The saved
   agreement qualifies only a bounded RL pilot, not population calibration. gpt-6.1-sol
   generated the teacher targets and judges pixels, so correlated preference is a
   limitation. Seek independent human annotation-queue review.
3. Curate: identify failures in existing and new production images, then follow
   `03_curate_visual_dataset.md`. Keep accepted targets separate from negative
   reward examples and uncertain cases. Preserve family partitions and actual
   reviewer identity. Do not train on the heldout benchmark.
4. Improve: inspect actual `forge-state.json`, fine-tune ID, immutable deployed SFT
   artifact, and optional `rl-training.json`. A queued job is not a trained model.
   A completed optimizer step is not a promoted deployment. A served adapter alone
   does not prove SFT→RL continuation; require the recorded matching storage/run,
   immutable source artifact and positive initial checkpoint step.
5. Evaluate: compare only matching frozen comparison fingerprints, prompt families,
   scorer configurations and generation budgets. Keep failed generations and judge
   abstentions in all-attempt denominators. Report each model's count, scored-only
   metric means, uncertainty and exact immutable checkpoint. Never pool repeated
   runs or claim gpt-6.1-sol parity from a small pilot or training completion.

Return concrete failure clusters, representative trace/image links, eligible new
training families, proposed review priorities and the exact artifact lineage.
Distinguish observed facts from hypotheses. Do not submit jobs, publish replacement
models, change stable-alias routes or manufacture dataset/evaluation references.
