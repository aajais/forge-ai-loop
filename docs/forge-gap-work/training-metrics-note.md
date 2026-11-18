# Observed Forge training schedules

These values come from the frozen W&B history captures for the two deployed `ft-google/gemma-4-26B-A4B-it` candidates. They describe measured training behavior; they do not establish image quality or parity with `gpt-6.1-sol`.

| Separate checkpoint metadata | 1 epoch, `bufo-gemma-gap-e1-20261015:v1` | 3 epochs, `bufo-gemma-gap-e3-20261015:v3` |
|---|---:|---:|
| Logged optimizer-update rows | 62 | 186 |
| First logged learning rate | 0.0000333333333 | 0.0000111111111 |
| Maximum logged learning rate | 0.0002 | 0.0002 |
| Update ordinals at maximum | 6–7 | 18–19 |
| Final logged learning rate | 0 | 0 |
| First / last logged batch loss | 1.78014 / 0.59147 | 1.72314 / 0.53139 |
| Epoch-average training losses | 0.71742 | 0.76715, 0.55999, 0.50654 |

Both requests omitted learning-rate overrides and used the service default. The actual histories establish the same observed peak of **0.0002**. They do not establish an identical schedule at each update: the longer run warms up and decays over more updates.

The observed values match linear warmup followed by linear decay: the first candidate rises over six updates and decays after update seven; the second rises over eighteen updates and decays after update nineteen. This is a fit to the logged values, not a claim that Forge returned an explicit scheduler configuration. Both end at zero. The `sft/gradient_step` counter resets each epoch, so cumulative ordinals above come from ordered history rows. Each epoch has 62 logged update rows; separate epoch-aggregate rows occupy W&B `_step` 62, 125, and 188.

The three-epoch run's learning rate continues decreasing across epoch boundaries: the last/next rates are approximately 0.000148503/0.000147305 and 0.0000742515/0.0000730539. It does not restart warmup at each epoch.

Losses are teacher-trajectory training losses. No validation-loss metric was observed in these captures. Selection remains the preregistered generated-image validation comparison, followed by untouched confirmation prompts; neither falling training loss nor this diagnostic blind image sample changes that rule.

## Evidence

The authenticated local W&B SDK read exact bounded `Run.scan_history` results for `_step`, `sft/gradient_step`, `sft/loss/train`, and `sft/loss/learning_rate`, plus separate `training_step`/`train/loss/train` epoch rows. The queried `_step` range was `[0, 250)`, containing all 62 and 186 observed update rows. Current run-summary fields can lag the final history; the table uses history instead.

- [One-epoch W&B run](https://wandb.ai/wandb-smle/bufo-gen/runs/bufo-gemma-gap-e1-20261015): `runs/forge-gap/training/e1/wandb-metrics.json`, SHA256 `75188acfc5a5634ac99898f50c9bfe3c10c72c35d55a785edcf02e471304e815`.
- [Three-epoch W&B run](https://wandb.ai/wandb-smle/bufo-gen/runs/bufo-gemma-gap-e3-20261015): `runs/forge-gap/training/e3/wandb-metrics.json`, SHA256 `4cbcf22d102ebcbecac89485db313edd8851acd6d5a43bc0d02a4c4e2431f5f1`.

Both remote run display names were verified as `ft-google/gemma-4-26B-A4B-it`. Epoch count, base ID, checkpoint URI, and dataset hash are separate config/notes metadata. Resource IDs and model artifacts are unchanged.
