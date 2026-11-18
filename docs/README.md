# Documentation

Start with the [project README](../README.md). This folder holds the reference material, the
experiment protocols and results, and the working notes behind them.

## Reference

| Doc | What it covers |
|---|---|
| [Studio and traces](reference/studio.md) | Running the chat app, model selector, runtime policy, tools, trace contract, Forge proxy |
| [Benchmarks and judges](reference/benchmarks-and-judges.md) | Prompt sets, benchmark stages and flags, astra and scene judges, paired statistics, loop report |
| [Training](reference/training.md) | Exporting reviewed trajectories, Forge SFT, Serverless RL, legacy stages, restoring old runs |
| [Results history](reference/results-history.md) | Every experiment with numbers and evidence links |
| [Forge preflight](forge-preflight.md) | Forge API surface verified read-only on 2026-10-15 |

## Experiment write-ups (by stage of the loop)

| Stage | Docs |
|---|---|
| Observe | [Agent Lens status](forge-loop-work/agent-lens-status.md), [telemetry notes](forge-demo-work/telemetry-notes.md) |
| Curate | [augmented prompts](distill/augmented-prompts.md), [curriculum review](forge-gap-work/curriculum-review.md), [service dataset audit](forge-contact-pilot/service-dataset-fidelity-audit.md) |
| Improve | [Stage B progress](distill/stage-b-progress.md), [training fit diagnosis](forge-gap-work/training-fit-diagnosis.md), [adapter capacity](forge-gap-work/adapter-capacity.md), [RL Dynamo implementation](forge-gap-work/rl-dynamo-implementation.md) |
| Evaluate | [judge calibration](distill/calibration-v1.md), [evaluation runbook](forge-gap-work/evaluation-runbook.md), [Stage B results](distill/stage-b-results.md), [gap results](forge-gap-work/results.md), [RL live results](forge-gap-work/rl-live-results.md) |

Design: [SVG distillation to Sol level](superpowers/specs/2026-10-21-svg-distillation-design.md).
Implementation plans live in `superpowers/plans/`.

## Folders

- `distill/`: Stage A (judge calibration) and Stage B (teacher data at scale, SFT sweep).
- `forge-loop-work/`: the first Gemma pilot, task briefs, reports and reviews.
- `forge-gap-work/`: the 124-example Gemma experiment, RL work, and its evaluation scripts.
- `forge-iteration2/`, `forge-contact-pilot/`: second data experiment and the contact-renderer pilot.
- `forge-demo-work/`: demo preparation, telemetry and post-training changes.
- `images/`: screenshots and example renders used by the README.

Several `.py` files in these folders are imported by `tests/`; they are source code and part of
the experiment record, not prose. `implementation-progress.md`, the `task*-brief/report/review`
files and the `*-2026-10-26.json` receipts are working notes kept for provenance.
