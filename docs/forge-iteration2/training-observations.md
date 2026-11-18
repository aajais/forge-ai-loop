# Iteration 2 training observations

## Completed runs: 2026-10-16 16:20 UTC

Both managed jobs are deployed at progress 100, with no reported training
failure, and both W&B runs are finished. Their histories verify the requested
peak learning rate of `1e-4` was used. Both remote configurations still specify
batch size 2 and all assistant turns, with 78 training / 23 validation examples
from dataset revision 2.

| Training metric | One epoch | Three epochs |
|---|---:|---:|
| Logged optimizer updates | 39 | 117 |
| Peak learning rate | 0.0001 | 0.0001 |
| Last learning rate | 0 | 0 |
| First batch training loss | 2.242584 | 1.880867 |
| Minimum batch training loss | 0.452276 | 0.301447 |
| Last batch training loss | 0.547893 | 0.551419 |

The three-epoch run logs `sft/gradient_step` from 1 to 39 **within each epoch**;
its maximum value of 39 must not be mistaken for the entire run's update count.
The three complete 39-update sequences occupy W&B `_step` ranges 0–38, 40–78,
and 80–118. The learning-rate schedule continues decaying across those epochs:
their observed peaks are `1e-4`, `7.333333333333334e-5`, and
`3.61904761904762e-5`. Batch losses fluctuate with the examples; the last batch
is not an epoch mean, validation loss, or evidence of improved image quality.

The completed artifacts are
`wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-iter2-e1-20261016:v1` and
`wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-iter2-e3-20261016:v3`.

Final evidence is separately preserved in
`runs/forge-iteration2/training/metrics-observations-final.json`. The earlier
`metrics-observations.json` snapshot remains unchanged. This refresh queried
the two exact run IDs linked below, scanned only the four named optimizer
metrics within W&B steps `[0, 200)`, and read the two exact managed job IDs.
It did not mutate jobs, deployment, or local lifecycle state.

## Initial running snapshot

Read-only checks on 2026-10-16 at approximately 16:11 UTC confirm that both
managed jobs accepted the requested recipe and the reviewed dataset counts.
The one-epoch job has completed and deployed; the three-epoch job had just
started at this snapshot. No lifecycle state files were changed by this check.

| Evidence | One epoch | Three epochs |
|---|---|---|
| Base model | `google/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it` |
| Fine-tune ID | `4b33269b-42a5-4fce-ad98-bfe89e0e6a8b` | `70cd7bf0-3899-473e-8965-38c57506eab4` |
| Remote model config | 1 epoch, batch 2, LR `1e-4`, all assistant turns | 3 epochs, batch 2, LR `1e-4`, all assistant turns |
| Remote example counts | 78 train / 23 validation | 78 train / 23 validation |
| Dataset revision | 2 | 2 |
| Forge status / progress | deployed / 100 | training / 0 |
| Observed optimizer-history rows | 36 | 0 at this snapshot |
| Maximum observed learning rate | `0.0001` | Not yet observable |

The one-epoch run's `sft/loss/learning_rate` reaches exactly `1e-4`, verifying
that the requested peak rate was honored rather than merely stored. Later
history shows decay to `8.571428571428573e-6`; that lower final observed rate is
not a configuration mismatch. Its observed gradient step reaches 36, with
training loss `0.6170859336853027` and gradient norm `1.6517289553283605`.
These are training signals, not evidence that image quality has improved.

The three-epoch job's remote training start is `2026-10-16T16:10:51.047Z`.
Its W&B run initially reported `finished` without any SFT metrics, while Forge
reported a live training job. That W&B state does not establish completion or
failure of training; use the managed job lifecycle and inspect optimizer
history again after it starts logging. Its actual optimizer learning rate
cannot yet be verified from this snapshot. Both W&B run configs were empty;
the recipe above comes from each managed job's `model_config.config`.

## Audit trail

The query used `wandb.Api().runs` on `wandb-smle/bufo-gen`, filtered to exactly
`bufo-gemma-iter2-e1-20261016` and `bufo-gemma-iter2-e3-20261016` via `displayName`,
with a four-run result bound. Follow-up reads used these exact run IDs and
`scan_history` for steps 0 through 199 with only `sft/gradient_step`,
`sft/loss/learning_rate`, `sft/loss/train`, and `sft/loss/grad_norm`. The 36 rows
are the available matching history rows in that bounded window, not an asserted
final optimizer-step total. Job reads used `ForgeClient.get_finetune` for the
two IDs above. No remote mutations were performed.

- [One-epoch training run](https://wandb.ai/wandb-smle/bufo-gen/runs/bufo-gemma-iter2-e1-20261016)
- [Three-epoch training run](https://wandb.ai/wandb-smle/bufo-gen/runs/bufo-gemma-iter2-e3-20261016)
- Local evidence: `runs/forge-iteration2/training/metrics-observations.json`.

The evidence file records separate initial and follow-up observation timestamps;
the job and run reads are sequential snapshots, not an atomic cross-service read.
