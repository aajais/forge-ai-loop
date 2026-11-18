# Bufo · Forge AI Loop

Draw Bufo in new scenes, inspect failures, curate actual agent trajectories, improve
google/gemma-4-26B-A4B-it through Forge Model Distillation and Serverless RL, then evaluate immutable
checkpoints against gpt-6.1-sol on the same heldout tasks.

`bufo-studio` is the conversational drawing UI. `bufogen` runs the measured loop.
Generated PNGs are retained even when visual quality fails. Successful training is
not a claim that google/gemma-4-26B-A4B-it matches gpt-6.1-sol; the heldout results determine that.

## Repository layout

The Python package is organized by responsibility:

```text
src/bufogen/
  agents/        Agent loop, instructions, model adapters, web context
  app/           FastAPI chat application and static UI
  rendering/     SVG composition/validation, image input, PNG rendering
  evaluation/    Benchmarks, scorers, calibration, references, reports
  curation/      Reviews, annotation queues, trajectories, dataset export
  training/      SFT, RL, and shared training contracts
  integrations/  Forge, Registry, Sandboxes, W&B, Weave tracing
  workflows/     Pipeline, AI loop, stage records, demo preparation
  cli.py         bufogen command entrypoint
  generation.py  Shared generation records
  models.py      Backend configuration and model presentation
  runtime.py     Shared inference policy
  settings.py    Project paths and environment settings
  ui.py          Compatibility launcher for existing bufo-studio installs
```

Keep new code in its owning package and use absolute `bufogen.<package>` imports.
Package initializers stay lightweight; import APIs from their defining module.
The shared modules at the root are used across these packages.

- `scripts/`, `config/`, `assets/`, and `aria_prompts/`: runnable workflows, model and
  benchmark configuration, character assets, and ARIA demo prompts.
- `tests/`: offline Python and chat UI checks.
- `docs/` and `notebooks/`: experiment protocols, results, and supporting tools.
  Several Python modules under `docs/` are imported by the tests; they are source code.
- `runs/`, `artifacts/`, `data/`, `wandb/`, and `.art/`: local experiment data,
  publications, logs, and training state. These are excluded from version control.
- `demo/`: recording scripts and local video deliverables. Generated media and edit
  intermediates are excluded from version control.

Use `uv run bufogen` as the pipeline entrypoint. To reconstruct an old experiment
checkout, use `scripts/restore_experiment_runtime.py` with its original run directory;
the canonical evidence and frozen source files remain under `runs/`. New runs freeze
the reorganized paths; historical run hashes and saved snapshots are not rewritten.

For experiments recorded before the package reorganization, restoration needs the
old scaffold as well as the run's frozen files. The complete pre-refactor source
backup is recorded in [the source-layout receipt](docs/source-layout-2026-10-26.json):

```bash
uv run python scripts/restore_experiment_runtime.py RUN_DIR TARGET \
  --project-root /private/tmp/cw-bufo-gen-src-layout-20261026T174844Z/before
```

That backup contains source/configuration/assets, not credentials or environments.
Copy it to durable storage if you need long-term restoration: `/private/tmp` can
be cleared by the OS. The restore command rejects mixing the two source layouts.

The [cleanup receipt](docs/cleanup-2026-10-26.json) lists the source backups,
restored checkouts, and generated intermediates moved out of the project on
2026-10-26, including their temporary recovery location. The recovery folder is
outside the repository and may be cleared by OS temporary-file cleanup.

## Forge demo preparation

Build the pinned validation comparison and inspect the lineage of the existing training data:

```bash
uv run python -m bufogen.workflows.demo_comparison
uv run python -m bufogen.workflows.demo_handoff
open runs/forge-demo/comparison/index.html
uv run --with marimo marimo run notebooks/forge_bufo_curation.py
```

The portable comparison includes all 48 validation attempts per arm, images, saved
judge explanations, trace/evaluation links, failures, latency and paired family
intervals. `config/forge-demo-comparison.json` pins the source digests and exact
model versions. `sft_q3:v1` is the **one-epoch** Qwen checkpoint; `sft_q1:v3` is
the **three-epoch** checkpoint. Labels derive from checkpoint lineage rather than
arm names. Historical `gpt-6.1-sol` results used different runtime settings and are
shown as a descriptive reference, not a paired parity result.

The live studio always enables reasoning, including retries: Qwen and its adapters
send `enable_thinking: true`, and Sol uses `reasoning.effort: medium`. Studio requests
omit output-token limits and share a maximum of 100 agent turns across the initial
attempt and automatic repair. Provider output/context limits still apply, as do
the existing inference timeouts and 30-minute studio deadline. The studio ignores
`BUFO_DISABLE_THINKING`; `/api/status` exposes its effective runtime policy.

CLI and benchmark defaults remain separate: Qwen and its adapters use 20,000 output
tokens, thinking enabled and a 900-second request timeout unless explicitly
overridden. Controlled benchmarks ignore environment drift and freeze resolved
settings per arm. Existing frozen runs are not migrated; use a new directory when
changing the runtime. Live studio results now use a different output-budget policy
from those historical benchmarks.

To publish the prepared comparison to W&B (no generation or judge calls):

```bash
uv run python -m bufogen.workflows.demo_comparison --publish --entity wandb-smle --project bufo-gen
```

`runs/forge-demo/comparison/publication.json` records the real publication state and
is safe to retry. A failed receipt is not a published dashboard. On 2026-10-24 the
coding environment could not launch the W&B service or resolve its API host, and
the W&B connector rejected its own configured credentials. The local comparison
is usable; live publication and the Forge evaluation pages still require verification.

New agent generations retain native root trace/span IDs and runtime metadata.
Saved scorer publication attaches source feedback without running a new judge.
Native custom feedback does **not** populate Agent Scores columns in the installed
SDK. [Telemetry notes](docs/forge-demo-work/telemetry-notes.md) explain the boundary
and give a source-only backfill command. The application renderer is explicitly
marked as application-generated; Agent Lens classification still needs live verification.

`runs/forge-demo/curation-lineage.json` contains 911 accepted **judge-reviewed
distillation** examples with original conversation links and reviewer identity.
The notebook can filter them by family and reviewer. In Forge Notebooks, import
`notebooks/forge_bufo_curation.py` and upload that JSON in the file widget. This
does not turn the existing examples into human-reviewed production data. For a
new production failure, use Agent Lens to inspect the pixels, review a corrected
authentic trajectory, then use the existing `export-reviewed` command to publish
a new dataset version. Leave the blinded confirmation queue for its human reviewer.

`runs/forge-demo/release-candidate.json` records the pinned Qwen candidate, actual
training recipe, dataset digest, evaluation evidence and direct base-model rollback.
It is marked **prepared, not promoted**. It does not change Forge routing. Registry
targeting now uses the registry's organization rather than assuming the team owns
it; `WANDB_REGISTRY_ORGANIZATION` optionally selects the organization explicitly.
This follows the [W&B Registry linking API](https://docs.wandb.ai/ref/python/experiments/run/).

For the demo, follow: studio prompt → original Agent Lens conversation → reviewed
example in the notebook → dataset and completed Forge training job → saved image
comparison → pinned model in the studio. Present SFT improvement and the evaluation
decision on RL; do not claim Sol parity or a production promotion from these records.

## Install and run the studio

Requires Python 3.12+, uv, a W&B team API key with inference/sandbox/Forge access,
and an OpenAI API key for gpt-6.1-sol. Copy `.env.example` to `.env` and add credentials
locally. Never put secrets in prompts or artifacts.

```bash
uv sync --locked
uv run bufogen --stage doctor
uv run bufogen --stage doctor --live
uv run bufo-studio --reload
```

Open http://127.0.0.1:8765 . `--reload` restarts the server after Python edits;
refresh the browser after static HTML/CSS/JS edits. `--port 9000` changes the port.

The chat supports iterative drawing, concurrent conversations, switching models
per message, pasted screenshots and PNG/JPEG/WebP attachments (5 MB maximum),
PNG/SVG downloads and Slack-style emoji names. Tool calls precede the final short
assistant text and rendered PNG in Weave conversations. SVG stays in tool arguments.
New native traces record the system prompt and tool definitions on the agent root,
plus every model call from the first request onward. Chat spans retain their inputs,
tool requests, provider-returned reasoning, token usage, served model, finish reason,
and errors. The application `render_png` span nests inside the model-requested
`render_svg` tool; only final text replies carry the output image. Earlier stored
traces retain their original instrumentation.
Native agent, chat, and tool spans are labeled when they start. SDK workflow/turn
bookkeeping spans are suppressed to avoid `Unknown` blocks and duplicate agent
invocations; actual tool calls retain their IDs, arguments, results, and render errors.
The trace layout follows [CoreWeave's Agents-view OTel contract](https://docs.coreweave.com/products/wandb/weave/guides/tracking/trace-agents-otel#shape-spans-for-the-agents-view):
each user exchange is one root `invoke_agent` in its own trace, with `chat` and
`execute_tool` descendants sharing its conversation ID. Repairs stay inside that
exchange; follow-up messages get a new trace with the same conversation ID.
Agent version `bufo-native-v2` identifies this instrumentation. A warning naming
an SDK `Bufo ... turn N` as the trace root indicates the earlier instrumentation
or an interrupted earlier run. Restart the studio to load the current code and
inspect a fresh completed turn; previously ingested traces are not rewritten.

The **Model** selector is focused on the four Forge demo models:

| Model | Role | Backend / checkpoint |
|---|---|---|
| `Qwen/Qwen3.8-27B` | Base | `qwen_base` |
| `ft-Qwen/Qwen3.8-27B` | SFT, 1 epoch | `sft_q3`, evaluated `:v1` |
| `rl-Qwen/Qwen3.8-27B` | SFT + RL | `rl_q3`, evaluated `:v6` |
| `gpt-6.1-sol` | Teacher | `closed_quality_bar` |

Choose **Compare all 4 models**, or open
http://127.0.0.1:8765/static/compare.html, to submit one prompt to all available
demo models concurrently. Outputs appear vertically in fixed model order with
independent elapsed times, errors, images, checkpoint details, and trace links.
Failed or invalid results offer **Retry this model**, which starts a fresh request
for that model with the same prompt and preserves the other cards. Earlier attempt
trace links remain visible, including after reloading a saved comparison. A retry
can still fail; output-limit failures are identified when the model never rendered.
You can browse saved comparisons while a run continues in the background. Both
chat and comparison show a rendered preview while the final reply finishes;
request-ID status polling also recovers completed results if the POST reply stalls.
Each comparison starts fresh conversations; it does not reuse a prior model's
answer. The last 12 comparisons are retained in this browser's local storage,
and completed generations also appear in the normal chat history. Keep the page
open while requests run: after a reload, unfinished requests are marked
interrupted and are never automatically resubmitted. Check chat history for
results that finish on the server after a reload.

Other experimental backends remain configured for CLI workflows, and historical
chat responses keep their recorded model identity. Environment model overrides
still apply, including `RL_Q3_MODEL`; availability reflects configuration, not a
live serving-health check. Refresh the page to pick up the simplified selector.

Displayed IDs follow configuration overrides. Fine-tuned labels use
`OPEN_TUNED_BASE_MODEL` or `OPEN_RL_BASE_MODEL`; these must match the adapter's
actual base. The selector and responses show the immutable adapter version
separately, so SFT and SFT+RL checkpoints remain distinguishable. Serving requests
continue to use the exact artifact URI. Historical responses use recorded IDs;
Forge responses show their resolved serving model when it was captured.

The existing Forge project uses alias `bufo-gen`, project ID
`dd5ddbe2-b31c-43e1-ab39-b5488c78f3d8`, entity/project `wandb-smle/bufo-gen`.
Its initial route serves google/gemma-4-26B-A4B-it. Selecting `forge_proxy` sends Chat Completions to
`https://proxy.training.coreweave.com/v1`, with literal metadata keys
`wandb.entity` and `wandb.thread_id`. The latter stays stable across chat turns.
The requested alias and returned serving model names are recorded separately in
trajectory provenance. This application never automatically changes routing.
gpt-6.1-sol uses the Responses API directly; it is not sent through the Chat-only proxy.

The live proxy smoke confirmed tools appear before the final text and PNG in the
application's native agent turn. Its conversation also contains two gateway
request turns; stable conversation metadata does not merge them into that agent
span. Evidence: `runs/forge-loop/proxy-agent-smoke/message-order-check.json`.
The documented proxy controls provide no per-request W3C parent-span merge.
Project routing `trace_sampling_rate=0` could suppress gateway traces for identified
conversations, but would also remove that Forge capture; it was not applied.
The direct baseline remains the default. See [metadata and traces](https://docs.coreweave.com/model-distillation/proxy/metadata-and-traces)
and [model routing](https://docs.coreweave.com/model-distillation/proxy/model-routing).

All drawing arms use the same system instructions, canonical connected Bufo SVG
assets and examples. Reference PNGs are not silently appended to user prompts.
Explicit user image attachments remain supported. Optional web context uses a
separate bounded `search_web` tool. Controlled benchmarks/RL disable that tool's
network access per agent instance to keep comparisons reproducible.

## Run and observe a controlled benchmark

`config/forge_prompts.yaml` has 12 train, 4 validation and 100 heldout scenarios,
with disjoint families. Some tasks include prior user turns. Freeze the runtime
before generation; changing source/assets/settings/dependencies invalidates resume
and cross-run comparison. Keep a new directory for every changed configuration.

```bash
# Development pilot; paid generation, separate from heldout evaluation.
uv run bufogen --stage benchmark --split train validation \
  --backends open_base closed_quality_bar --output runs/my-pilot
uv run bufogen --stage benchmark-score --output runs/my-pilot --publish
uv run bufogen --stage scene-review-queue --output runs/my-pilot

# Final heldout A/B; use actual immutable deployed model references.
uv run bufogen --stage benchmark --split heldout \
  --backends open_base closed_quality_bar open_tuned --output runs/my-heldout
uv run bufogen --stage benchmark-score --output runs/my-heldout --publish
```

`--checkpoint LABEL=MODEL` runs an explicit served checkpoint, including an RL
artifact. Without `--backends` only the given checkpoints run (the default
`open_base`/`closed_quality_bar` arms are used only when neither flag is given),
and each LABEL needs an entry in `config/models.yaml`. `--limit`, `--concurrency` (default 2), `--max-tokens` (6000) and
`--max-turns` (100) bound a run. The agent also defaults to 100 turns in chat,
with a 30-minute request timeout. `--max-turns` also applies to `run` and `demo`.
`--benchmark-config` selects a different validated
family-partitioned YAML. Generation defaults to heldout; always set `--split`
explicitly for a development run. Scoring is a separate stage. `--publish` writes
native Weave scorer feedback and Evaluation objects; local evidence always remains.

Outputs include `benchmark.json`, frozen source/assets, `manifest.jsonl`, images,
`scored.jsonl`, `summary.json`, `comparison.html`, and actual publication/queue
receipts. Generation failures and judge abstentions remain in denominators.
Interrupted in-flight tasks are recorded as failures rather than silently replayed.

SceneQuality checks Bufo presence, similarity, exact scene requirements, connected
anatomy, composition and readability. Similarity is a VLM judgment, not an embedding
metric. `SCENE_JUDGE_BACKEND=open_base` uses `JUDGE_MODEL` (Qwen); selecting
`closed_quality_bar` uses gpt-6.1-sol. `SCENE_JUDGE_MODEL` is an optional scene-only override.
The rubric and provider settings are fingerprinted. Changing judges requires fresh
calibration and separate evaluations; never relabel earlier scores.

## Curate existing images and authentic trajectories

Use `aria_prompts/03_curate_visual_dataset.md` in ARIA. The real pilot annotation
queue is linked in `runs/forge-loop/pilot/review-queue.json`. The image comparison
run is linked in `runs/forge-loop/pilot-visualization.json`. Inspect pixels, preserve
reviewer identity, and export explicit accept/reject decisions keyed to generation
IDs. Assistant reviews are labeled assistant, never human queue approvals.

```bash
uv run bufogen --stage export-reviewed \
  --manifest runs/my-pilot/manifest.jsonl --reviews path/to/reviews.jsonl \
  --include-validation --output runs/my-reviewed
```

Repeat `--reviews` for nonoverlapping files. The export uses existing captures; it
does not regenerate targets. It writes `reviewed-examples.jsonl`, exclusion
`curation-diagnostics.json` and a curation summary. Explicit accepted reviews,
valid renders, true model tool calls and frozen instructions are required. Heldout
IDs/families, rejected/unreviewed rows, image-input trajectories and external prior
conversations are excluded. Good edited PNGs can remain in visual evaluation even
when their earlier assistant targets are not qualified for SFT v1.

Validation remains validation (`val` in Forge), never training. Automatic scorer
passes alone cannot qualify targets. Keep the source export immutable after import.

## Improve with managed Forge Model Distillation

The primary SFT path is the documented Forge service; no GPU image is required.
The existing project must already exist; these commands never create or reroute it.

```bash
uv run bufogen --stage forge-preflight \
  --project-id dd5ddbe2-b31c-43e1-ab39-b5488c78f3d8
uv run bufogen --stage forge-import \
  --source runs/my-reviewed/reviewed-examples.jsonl \
  --dataset-name my-reviewed-bufo --name my-gemma-sft \
  --project-id dd5ddbe2-b31c-43e1-ab39-b5488c78f3d8 \
  --forge-state runs/my-forge/state.json
uv run bufogen --stage forge-submit --forge-state runs/my-forge/state.json \
  --epochs 1 --batch-size 2
uv run bufogen --stage forge-status --forge-state runs/my-forge/state.json
```

Import validates locally, preflights eligibility, creates a dataset once, dry-runs
exact entries and uploads them. Submit trains all assistant turns, including actual
tool requests. Status only polls the existing job. Source hashes, remote IDs,
pending mutations and errors are durably recorded. An ambiguous write blocks
repetition until remote reconciliation; never delete the state to retry a job.
Catalog eligibility is not project-specific job acceptance.

Only a `deployed` response with a real `model_artifact_ref` enables the SFT arm.
Set `OPEN_TUNED_MODEL` to that immutable `:vN` URI in `.env`, then select **ft-google/gemma-4-26B-A4B-it** in the UI. The completed pilot's exact state is in
`runs/forge-loop/forge-state.json`; its artifact is
`wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v1`.
This is a small assistant-reviewed pilot, not demonstrated gpt-6.1-sol parity.

The completed 100-prompt heldout comparison found strict provisional scene passes
of **1/100 for baseline google/gemma-4-26B-A4B-it, 71/100 for gpt-6.1-sol, and 1/100 for ft-google/gemma-4-26B-A4B-it**. The
10-target SFT pilot did not improve this pass rate. See the
[actual W&B image comparison](https://wandb.ai/wandb-smle/bufo-gen/runs/rqbi08g1)
and [live results and limitations](docs/forge-loop-work/live-results.md).

The quality-gap follow-up uses **124 reviewed training examples and 23 validation
targets** from actual `gpt-6.1-sol` trajectories. It compares one and three epochs
on identical data, selects on 24 validation prompts, and reserves 80 new prompts
for confirmation. See the [research rationale](docs/forge-gap-work/research.md),
[evaluation protocol](docs/forge-gap-work/evaluation-runbook.md), and
[reviewed image dataset](https://wandb.ai/wandb-smle/bufo-gen/runs/ecbf90c5).
The completed confirmation found **2/80 strict passes for
`google/gemma-4-26B-A4B-it`, 65/80 for `gpt-6.1-sol`, and 7/80 for the selected
one-epoch `ft-google/gemma-4-26B-A4B-it`**. Its observed gain is 6.25 percentage
points (family-bootstrap 95% interval: 0–13.75); the teacher gap remains 72.5
points. Requirement coverage did not improve. The provisional visual scorer is
not independently calibrated, and the gap is not closed.

See the [results and limitations](docs/forge-gap-work/results.md),
[local image showcase](runs/forge-gap/showcase/index.html), and
[W&B confirmation comparison](https://wandb.ai/wandb-smle/bufo-gen/runs/504b5fe7).
The candidate receipts are `runs/forge-gap/training/e1/forge-state.json` and
`runs/forge-gap/training/e3/forge-state.json`; completed selection and paired
results are in `runs/forge-gap/selection.json` and `paired-results.json`.
The selected checkpoint is
`wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-gap-e1-20261015:v1`.
The existing running UI still uses the original pilot until its configured
`OPEN_TUNED_MODEL` is changed and the process restarted. Selection is experimental,
not a claim of production readiness or teacher parity.

For current progress, inspect those candidate receipts,
`runs/forge-loop/heldout-baselines/` for the original heldout run,
and `runs/forge-gap/rl-direct-result.json` for the verified RL result. The original
RL training record retains a post-training Registry error; the recovery receipt
separately confirms completed training and serving. Earlier failed attempts remain
evidence. Regenerate
the loop report with the intended `--rl-state` after new results arrive; no final
heldout improvement or gpt-6.1-sol-parity result is implied by the SFT deployment.

## Continue the same SFT model with Serverless RL

```bash
uv sync --locked --extra rl
uv run bufogen --stage train-rl \
  --dataset-ref wandb-smle/bufo-gen/bufo-rl-environments-20261015:v0 \
  --name my-gemma-sft-rl --steps 1 \
  --calibration runs/forge-loop/calibration/rl-calibration.json \
  --initial-checkpoint wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v1 \
  --forge-finetune-id 100202f8-7034-4bc2-bd2f-2375dd43fbba
```

This documents the original v1-to-v2 continuation. That model storage is now at
step 2; do not replay the v1 continuation against it. Another training experiment
needs its own verified SFT deployment and current checkpoint.

Use a train-only environment-task artifact with prompt text, exact requirements and
family/split, not the SFT message dataset or the old reference-image dataset.
`rl_contract.write_environment_tasks` builds it from the benchmark, explicitly
excluding heldout/validation and unsupported conversational training tasks.

ART is pinned to upstream commit `45a17cbd2a85cc0333172e652fc1211c6ae3f3da`
(package version `0.5.20`); the PyPI release with that same version lacks the
required Dynamo token support. ART captures only student actions; search is disabled and the pixel judge
runs outside capture. Reward is identity × requirement coverage × mean(anatomy,
composition, readability), with zero for missing Bufo, invalid output or unbound
scores. Calibration must match the exact active scene judge and explicit review
observations. The 32-image assistant-reviewed pilot qualifies only limited pilot
agreement; it does not establish population calibration. gpt-6.1-sol is both teacher and
judge, so correlated preference remains possible; independent human review matters.

Continuation verifies the actual deployed job, immutable artifact identity, source
run/storage, original google/gemma-4-26B-A4B-it base and positive checkpoint step before rollout. It
never silently starts fresh base RL and calls it continuation. Omitting both initial
checkpoint and fine-tune ID selects explicitly labeled base RL. Inspect
`runs/<name>/rl-training.json` for actual status, completed steps and artifact.
This is not an automatic crash-resume scheduler. After interruptions, inspect remote
state before another launch. Set `OPEN_RL_MODEL` only to an actual returned serving
artifact, preserving the SFT `:vN` separately for evaluation.

The corrected live continuation completed **one real optimizer update, from
step 1 to step 2**, producing
`wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v2`.
Its identity and digest were verified, and a subsequent full agent run rendered
a valid PNG. The initial serving attempt returned HTTP 500; that failure and the
successful retry are both retained. **No RL quality improvement has been measured.**

Registry linking failed because the organization for `wandb-smle` could not be
resolved. This did not undo training or prevent the verified checkpoint from
serving. The original run still records that post-training `CommError`; use the
[recovery receipt](runs/forge-gap/rl-direct-result.json) and
[live RL evidence](docs/forge-gap-work/rl-live-results.md) to interpret it.

Earlier history-tokenization and training-gateway timeout failures remain
unchanged. The successful path uses exact Dynamo token evidence, local ART
tokenization checks, and the verified inference endpoint. No failed captures
were repaired or reused to claim success.

## View the evidence-backed loop

```bash
uv run bufogen --stage loop-report --loop-root runs/forge-loop \
  --forge-state runs/forge-loop/forge-state.json \
  --rl-state runs/forge-gap/rl-direct-result.json \
  --output runs/forge-loop/showcase
open runs/forge-loop/showcase/index.html
```

The standalone HTML includes actual phase states, paired rendered images, native
queue/evaluation links, reviewer counts, calibration limits, training artifacts and
per-run metrics. `loop-report.json` retains machine-readable evidence. Re-run after
jobs/evals finish; it performs no network writes. `--benchmark-dir` can be repeated
to choose explicit runs. Different fingerprints, judges and pilot/heldout samples
stay separate. Repeated arms are never pooled. Use `--reviews`/`--calibration` for
explicit evidence paths and `--rl-state` for the actual current training run.

`aria_prompts/04_forge_loop.md` describes the complete curation/evaluation handoff.

## Rendering and development

Generated SVG is rendered in an isolated CoreWeave CPU sandbox using `cwsandbox`;
it is not executed on the host. The worker receives no credentials, has no network,
and uses a pinned resvg wheel, SVG allowlist and resource limits. A warm sandbox is
reused and stopped on shutdown/failure. Character asset provenance is in
`assets/bufo_style/sources.json` (original artwork from bufo.fun).

```bash
uv run bufogen --stage render-smoke --output runs/render-check
uv run bufogen --stage sandbox-status --sandbox-id ACTUAL_ID
uv run pytest -q
uv run ruff check src scripts tests
node --test tests/chat_ui.test.cjs
```

Legacy `run`, `demo`, `eval`, `eval-identity`, `review-queue`, `evaluate`, source-image
builders and GPU `train-sft` stages remain for compatibility. They are not the
managed Forge showcase path. The old `curate` generate-then-review flow does not
qualify existing reviewed IDs; use `export-reviewed`. Legacy CLIP metrics are not
used for the current RL reward. No mocked training result is presented as live.

Official references: [Model Distillation](https://docs.coreweave.com/model-distillation/quickstart),
[Sandboxes](https://docs.coreweave.com/products/sandboxes/get-started).
