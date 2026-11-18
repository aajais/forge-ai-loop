# Second-iteration matched evaluation

Run from the repository root using the existing virtual environment. Evaluation
never submits training, changes `.env`, or promotes a deployment.

```sh
.venv/bin/python docs/forge-iteration2/evaluate.py --stage validation
.venv/bin/python docs/forge-iteration2/evaluate.py --stage select
.venv/bin/python docs/forge-iteration2/evaluate.py --stage fit
.venv/bin/python docs/forge-iteration2/evaluate.py --stage confirmation
.venv/bin/python docs/forge-iteration2/evaluate.py --stage summarize
```

`validation`, `fit`, and `confirmation` perform paid inference and visual scoring,
then publish native Weave evaluations. Selection and summarization are local.

Validation contains 24 prompts and five arms: `google/gemma-4-26B-A4B-it`,
`gpt-6.1-sol`, the prior selected fine-tune, and two new fine-tunes. All tuned arms
are labeled `ft-google/gemma-4-26B-A4B-it`; immutable artifact IDs and internal arm
names distinguish their checkpoints. The prior control cannot win selection.
The ranking is strict pass count, scored-only requirement coverage, fewer
failed generations, then one epoch. Every attempt stays in the pass denominator.

Before generation, the runner checks local reviewed-source receipts, candidate
recipes, live Forge job identity, and immutable W&B artifact identity. The new
recipes must be independent base initialization, one/three epochs, batch two,
all assistant turns, and peak learning rate 1e-4. No adapter rank is assumed.
Runtime, source, sampling, scorer configuration, curriculum, and selected
checkpoint identities are frozen. A changed protocol cannot silently resume.

Selection requires all 120 validation rows and their scores, including error
outcomes; it binds their file hashes and is created before confirmation starts.
Fit uses the 16 fixed plain-contact training prompts only if each has an accepted
training target. It cannot select a checkpoint and is not heldout evidence.
Confirmation evaluates base, teacher, and the chosen new checkpoint on the fresh
48 prompts in 12 families. The report includes paired family-bootstrap intervals
and the remaining teacher gap; a point estimate alone is not a parity claim.

Outputs are `validation-candidates`, `training-fit-selected`, and
`confirmation-selected` under `runs/forge-iteration2`, with `selection.json` and
`paired-results.json` alongside them. The local API
`validated_evidence(root, stage, curriculum_path=None)` returns frozen settings,
actual scored `BufoGeneration` objects, the recomputed summary, and protocol.
Publishers must use this API before visualizing a comparison.

The visual scorer remains provisional and can over-credit weak contact geometry.
Keep assistant image reviews distinct from human annotation-queue approvals.
Native Weave publication receipts must be inspected separately for publication
errors; local scoring completeness does not assert remote publication success.
