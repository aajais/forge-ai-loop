# Stage B results (draft; human-check numbers pending)

## Verdict

Not Sol level. No student reached a validation strict pass above 0 in the SFT sweep, so the confirmation compares base Gemma with Sol only. On the 80 heldout prompts base Gemma passes 1/80 and Sol 60/80; the paired difference (base minus Sol) is -0.7375 with a 95% family-cluster bootstrap interval of [-0.875, -0.5875], far below the -10 pp bound that defines "Sol level". Verdict: gap -73.8 pp [-87.5, -58.8]. There is no distilled student to put in that comparison: SFT selected no winner and RL was not run.

## Heldout confirmation (`bufo-astra-v5`, k=3, heldout references `distill-heldout-v1`)

| arm | prompts | judge errors | strict pass | requirement coverage (mean) | latency p50 / p90 (s) | output tokens (mean) | serving path |
|---|---|---|---|---|---|---|---|
| open_base (`google/gemma-4-26B-A4B-it`) | 80 | 0 | 1/80 (0.0125) | 0.467 | 14.9 / 30.9 | 1679 | wandb_inference |
| closed_quality_bar (Sol, Task 4 arm) | 80 | 0 | 60/80 (0.750) | 0.928 | 48.7 / 61.8 | 2102 | openai |
| SFT arms | none (no winner) | | | | | | |
| RL arm | not run: no selected SFT checkpoint | | | | | | |

Paired (16 families x 5 prompts, 20000 replicates, seed 20261015): `runs/distill/stage-b/confirmation/paired-open_base-vs-closed_quality_bar.json`. Per-family differences range from -0.2 (chalk_artist, golf_green) to -1.0 (bowling_alley, car_wash, drum_lesson, kite_flying, pancake_breakfast, science_fair, woodcarving). Sol's reasoning tokens average 280 per call; base reports none. Tool-call counts are not in the scoring summaries and are not reported here.

Generation: 80/80 base rows completed and valid, 0 generation errors (396 s). Judging: 480 judge calls (80 prompts x 3 samples x 2 arms), 0 judge errors, 0 low-confidence rows, 0 truncated calls (about 8 min per arm). The confirmation was run once; nothing was re-run or re-judged.

## Sweep (validation, 48 prompts) and selection

| label | epochs | strict pass | coverage | paired vs open_base | paired vs Sol |
|---|---|---|---|---|---|
| sft_j1 | 3 | 0/48 | 0.413 | 0.000 [0.000, 0.000] | -0.8125 [-0.875, -0.750] |
| sft_j3 | 1 | 0/48 (47 scored; 1 generation error) | 0.448 | 0.000 [0.000, 0.000] | -0.8125 [-0.875, -0.750] |

Selection rule (highest strict pass, ties by coverage, errors, tokens; the paired lower bound over open_base must exceed 0): no winner, `selected: null` (`runs/distill/stage-b/sweep/selection.json`). sft_j4 (half data) was not run. Adapter distinctness (`sweep/distinctness.json`): sft_j1 is distinct from base on output length (-410 tokens, permutation p = 0.0012, base-vs-base controls p > 0.05); sft_j3 is not separable from base by any local statistic (+70 tokens, p = 0.63), so its 0/48 is not certified as a trained-model result; neither student uses any of the top-30 element ids of Sol's training SVGs (share 0.000). The direct paid check of the served adapters was not run. Rank is 1 because the Forge API exposes no rank field.

## Data funnel

1024 Sol train prompts over 4 attempts of 256 (1021 completed generations, 1021 judged ok, 0 judge errors); 911 accepted (below the 1000 target, stopped at 4 attempts), 113 rejected (`runs/distill/stage-b/export/funnel.json`). The SFT dataset is `export/accepted.jsonl` (911 train rows, train split only).

## RL

Not run: no selected SFT checkpoint. The controller may add a from-base demonstration separately.

## Human check (judge-independent)

Pending user labels. The 40-item blinded packet is built (`runs/distill/stage-b/confirmation/packet`, items sha256 `dcfdcacfb410925d102856587ea5a0803851d169bd694304194e104bd8600a04`), mixed across the two arms (15 base items, 25 Sol items; the Sol arm has far more passing drawings to sample from), with the key under `runs/distill/stage-b/confirmation/key` and the Weave queue `bufo review confirmation` (https://wandb.ai/wandb-smle/bufo-gen/weave/annotation-queues/01a16a66-b6cf-78f7-92c5-fb927d54f303). Agreement, judge-pass / reviewer-missing count and by-arm agreement to be added after `scripts/review_agreement.py`. No rubric change is allowed in response.

## Weave evaluations and W&B records

- Evaluations: `bufo-astra-google-gemma-4-26B-A4B-it-v5-k3` (confirmation), `bufo-astra-gpt-6.1-sol-v5-k3` (heldout Sol arm; same name family as the data runs, distinguished by digest).
- W&B records (project `wandb-smle/bufo-gen`): `bufo-stage-b-sweep-selection`, `bufo-stage-b-sweep-distinctness`, `bufo-stage-b-confirmation-summary`, `bufo-stage-b-heldout-sol-summary`, `bufo-stage-b-confirmation-paired-open_base-vs-closed_quality_bar`, plus the earlier funnel, decisions, probe and checkpoint records (ledger `runs/distill/stage-b/publications.json`).

## Paid calls (approximate where the progress doc does not state a count)

| task | generations | judge calls |
|---|---|---|
| 3: augmented Sol references | 128 + 1 | none |
| 4: heldout Sol reference and Sol arm | 80 + 80 | none |
| 5: anchoring probe | none | 39 |
| 6: Sol data, 4 attempts | 1024 (1021 completed) | about 3063 (1021 x 3) |
| 7: SFT sweep evaluation | 48 + 48 (one error) per label | about 288 (48 x 3 x 2, less one unscored row) |
| 7: Forge jobs | 2 SFT jobs (sft_j1, sft_j3) | |
| 10: confirmation | 80 | 480 |

The earlier rows come from the progress doc; Task 6 and Task 7 judge totals are computed from prompt counts and k=3, not read from a ledger. No Forge job was created for RL.
