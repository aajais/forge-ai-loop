# Task 5 — Forge CLI, opt-in routing and evidence-backed showcase

Implemented locally; no remote mutations, model calls, training submissions or routing changes by this worker. Root owns live verification, heldout evaluation and training. Runtime files are frozen for the root's heldout run.

## Changed files

- `src/bufogen/cli.py`
- `src/bufogen/models.py`
- `src/bufogen/agent.py` (only proxy request metadata and truthful alias/served-model provenance)
- `src/bufogen/ui.py` (Forge entity availability check only)
- `src/bufogen/benchmark.py` (root-authorized inclusion of `models.py` in frozen source hashes only)
- `config/models.yaml` (new optional Forge and RL entries; existing entries unchanged)
- new `src/bufogen/loop_report.py`
- new `tests/test_forge_cli.py`
- `tests/test_ui.py` (new optional model availability)
- `aria_prompts/03_curate_visual_dataset.md`
- new `aria_prompts/04_forge_loop.md`
- `README.md`

Original files have snapshots under `docs/forge-loop-work/task5-before/`. Additional benchmark/config/UI-test snapshots were captured before edits. No Git repository is present; no commits/worktrees were invented.

## CLI contracts

- `forge-preflight`: actual read-only `ForgeClient.preflight`; explicit alias/entity/project identity. Does not create a project or change routing.
- `benchmark`: `load_benchmark` → selected split → configured arms/explicit `--checkpoint LABEL=MODEL` → `run_benchmark`. Defaults heldout, 6,000 tokens, 8 turns, concurrency 2. Existing benchmark preserves exact settings, failures and interrupted rows. Different runtime requires a new output directory.
- `benchmark-score`: `score_benchmark`; `--publish` explicitly enables native Weave feedback/Evaluation publishing. Never regenerates examples.
- `scene-review-queue`: real SceneQuality queue API and persisted receipt.
- `export-reviewed`: requires existing `--manifest` and one or more explicit `--reviews`; validates unique decisions, source identity and frozen family/split agreement. Previously attached reviews not explicitly provided are not silently accepted. Filters heldout IDs/families, rejected/unreviewed/legacy/ineligible trajectories, then revalidates through Forge's actual row mapper. `--include-validation` preserves validation as validation. Writes `reviewed-examples.jsonl`, curation diagnostics and provenance/count summary; refuses to overwrite a different existing export. This fixes the handoff without using legacy generate-then-review curation.
- `forge-import`: local source and heldout validation before constructing a remote client; real preflight, initialize and resumable import APIs. Requires source, dataset name, model name, existing project ID and a durable state path. Does not submit training.
- `forge-submit`: actual idempotent submission API with epochs/batch size. `forge-status`: only polls an existing fine-tune ID. Both retain the underlying workflow's locking, ambiguity and immutable-source safeguards.
- `train-rl`: now passes explicit calibration, immutable initial SFT checkpoint, actual Forge job ID and alias to `RLTrainer`. Missing calibration or only half the continuation identity fails before training. Base RL remains explicitly selectable by omitting both continuation arguments.
- `loop-report`: builds local HTML/JSON only; see below.

README replaces obsolete CLIP/length-reward and automatic name-resume claims. The managed Forge path is primary; legacy custom GPU stages remain labeled legacy compatibility paths.

## Proxy and model selection

`forge_proxy` is an explicit optional backend using `FORGE_PROXY_MODEL` (default `bufo-gen`) and `FORGE_PROXY_BASE_URL` (default `https://proxy.training.coreweave.com/v1`). Provider `forge` selects Chat Completions. `ModelSettings.extra_body` carries literal metadata keys `wandb.entity` and `wandb.thread_id`; the thread is the same agent `group_id` across turns. UI availability checks W&B entity and shared credentials. Original direct models and default selection remain unchanged. Demo's historical automatic arm list excludes the new optional arms.

The proxy's requested alias remains `row.model` and `trajectory.provenance.requested_alias`. A response hook records actual nonempty `model` fields returned by successful non-streaming responses in `trajectory.provenance.resolved_serving_models`; an empty list means no resolution was observed, never a guessed raw model. Direct backends send no extra metadata and use their previous request path.

`open_rl` is a separate optional serverless backend controlled by `OPEN_RL_MODEL`, empty/disabled by default. The existing `OPEN_TUNED_MODEL` retains immutable SFT. Root will set the RL variable only when an actual returned serving artifact exists. Existing chat model switching, concurrent conversations, paste input and trace ordering are untouched.

## Local loop report

Public APIs:

```python
build_loop_report(root, *, benchmark_dirs=None, forge_state=None,
                  review_paths=None, calibration_path=None, rl_state=None) -> dict
write_loop_report(root, output, **same_options) -> Path  # index.html
```

Default benchmark discovery is direct child `*/benchmark.json` under the loop root. CLI `--benchmark-dir` is repeatable for explicit selection. Report reads frozen manifests/summaries and checks run/configuration/fingerprint identity. Different fingerprints are separate comparison groups; repeated arms are flagged and never pooled. Pilot/development and heldout runs retain per-run denominators and actual scorer configuration. Judge errors stay in total attempts. It does not infer parity or improvement from deployment.

The five phase cards show actual saved status, queue/evaluation references, source dataset/artifact IDs, pending mutations/errors, optional RL state and calibration evidence. Missing evidence is not started. Actual reviewer counts distinguish visual acceptance from SFT eligibility and retain reviewer kind. Calibration's limited 32-image development evidence, lack of population calibration and shared Sol teacher/judge bias are explicit. Original Qwen-v1 pilot summary is not relabeled as Sol-v2.

Native Evaluation object URLs are derived from actual published Weave refs, queues from their receipts, and the W&B image-table link from `pilot-visualization.json`. Forge links to the documented distillation entrypoint; no invented project deep link. HTML embeds paired PNGs for the first 12 prompts per run; complete per-run counts and rows remain in JSON. This is a static artifact regenerated on command, not a live dashboard. Root's ongoing heldout run appears as running with its actual completed count until saved scores exist.

Generated local artifact: `runs/forge-loop/showcase/index.html` and `loop-report.json`. Read-only verification found five phase cards, actual visualization URL and real Gemma 1/16 / Sol 15/16 assistant-review counts. PNGs are embedded, not inferred from SVG. No browser-control tool was available; HTML structure/content was inspected and parsed, not claimed browser-rendered.

## Verification

- Full Python suite before final integration test: **244 passed**; all **3 Node chat tests passed**.
- Added actual Agents SDK + mocked HTTP integration: two agent turns each execute render tool then final reply; every request carries the same proxy thread/entity metadata; actual returned model is recorded and alias retained. Final focused CLI suite: **11 passed**.
- Whole-tree `ruff check src scripts tests`: **All checks passed**.
- Final full suite result recorded below after completion.

Tests also cover heldout default selection, immutable extra checkpoint wiring, local export of real tool IDs/reviewer provenance, conflicting review partitions, duplicate decisions, invalid Forge import before client construction, status-only polling versus explicit submit, required RL calibration/continuation arguments, model source fingerprint coverage, incompatible/repeated-run reporting and mismatched-summary rejection. External transport is mocked; no mocked result is claimed as live success.

Final regression: `.venv/bin/pytest -q` → **245 passed**, 2,290 existing Weave/Pydantic deprecation warnings, 5.16 seconds. Review diff is `docs/forge-loop-work/task5-diff.txt`.


## Confirmed live proxy tracing limitation

Root's live proxy-agent smoke verified the native application turn has render tools before final assistant text and a real PNG content reference. `runs/forge-loop/proxy-agent-smoke/message-order-check.json` records tool positions 3/4, final assistant position 6, and three total conversation turns: one native agent turn plus two gateway request turns. The display ordering is correct within the application turn; gateway requests remain additional turns.

The official [metadata/traces](https://docs.coreweave.com/model-distillation/proxy/metadata-and-traces) and [routing](https://docs.coreweave.com/model-distillation/proxy/model-routing) contracts do not document per-request W3C parent-span merging. Routing-level sampling zero would remove gateway capture for identified conversations, so root did not apply it. Direct baseline defaults and all routes remain unchanged.

README now identifies this limitation and points separately to deployed SFT state, heldout evidence and the explicitly selected RL attempt state. This prose update makes no final improvement claim. No runtime, configuration or routing changes; no full test rerun for documentation alone. Official documentation was re-read and the saved live ordering receipt checked.
