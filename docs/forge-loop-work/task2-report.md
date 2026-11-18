# Task 2 — Frozen paired benchmark and scene evaluation

Status: implemented, focused/regression checks passing, independent root-assigned reviewer reported no remaining important findings. Code frozen for the root-owned live pilot. This worker made no paid API calls, Git operations, or external publications. Root separately reports a real positive/negative SceneQuality smoke; those results are not included in this worker's test count or represented as calibration.

## Files and prompt partition

New: `config/forge_prompts.yaml`, `src/bufogen/benchmark.py`, `src/bufogen/scorers/scene_quality.py`, `tests/test_benchmark.py`, `tests/test_scene_quality.py`.

Modified: `src/bufogen/agent.py`, `generation.py`, `web_context.py`, `observability.py`. Existing-file snapshots are in `docs/forge-loop-work/task2-before/src/bufogen/`. No edits to trajectories, curation, CLI, or existing eval entrypoints.

The static YAML defines **116 scenarios: 12 train, 4 validation, 100 heldout**. Concert performance and F1 presentation are the first two training scenarios; their conversational edits also remain in training. Heldout has 20 distinct scenario families, five scenarios each, including physical action/contact, attached props, multiple characters, occlusion, full-body framing, small scale, exact text, and an actual conversational edit. Four additional validation families are ice-cream serving, airport marshalling, florist arranging, and locksmith work. Family/split leakage and duplicate IDs are rejected. These are definition counts, not claims that 116 scenarios have run.

Each YAML row contains `id`, `family`, `split`, `prompt`, and unique nonempty string `requirements`. Optional `turns` is a list of prior user requests. These are genuinely executed with the same arm and authentic conversation history; final requirements explicitly describe the edited scene. Intermediate rendered files survive. A failed prior turn is identified as `failed_turn_id` and cannot be scored as the requested final edit.

## Public APIs

```python
from pathlib import Path
from bufogen.evaluation.benchmark import load_benchmark, run_benchmark, score_benchmark, comparison_fingerprint
from bufogen.models import load_backends

prompts = load_benchmark(Path("config/forge_prompts.yaml"))  # validate complete partitions first
pilot = [p for p in prompts if p["split"] == "train"][:2]
backends = load_backends()
# Supply the configured student and Sol teacher configs explicitly.
rows = await run_benchmark(
    pilot, [backends["open_base"], backends["closed_quality_bar"]],
    Path("runs/your-new-pilot"),
    limit=None, concurrency=2, temperature=0.7, max_tokens=6000, max_turns=8,
)
scored = await score_benchmark(Path("runs/your-new-pilot"), concurrency=2, publish=True)
```

The configured arm keys are `open_base`, `closed_quality_bar`, and optional `open_tuned`. `run_benchmark` accepts either a sequence of `BackendConfig` or a mapping of them. The model ID is never silently substituted. `limit` limits selected **scenarios per arm**, with one repetition explicitly frozen; conversational scenarios additionally execute their prior requests.

`score_benchmark(output_dir: Path, *, concurrency=2, publish=True) -> list[BufoGeneration]` requires a complete paired generation manifest. `publish=False` disables Weave feedback/evaluation publication, **not VLM inference**; scoring a real image still calls the configured judge. No CLIP dependency. `JUDGE_MODEL` is resolved when the scorer is instantiated and frozen into the generation configuration alongside the judge backend endpoint/provider. Current default is the existing project Qwen judge, with temperature 0 and token budgets 3200/6400 for the existing bounded structured-output retry.

`SceneQuality().score(output, rescore=False, saved_score=None)` accepts either a generation dict or visual trace envelope. `saved_score` supports validated native-feedback replay of a previously computed score. `scene_result(data, requirements, *, threshold=.65)` validates raw results. `create_scene_review_queue(rows, *, name="Bufo scene calibration v1")` lives in `observability.py` and creates a separate scene queue. Existing identity queues retain their default name and fields.

## Frozen conditions, resume, and concurrency

`benchmark.json` records schema version, random run ID, full configuration, configuration hash, and comparison fingerprint. Configuration includes selected prompt rows and hash; prompt version; complete nonsecret backend configs; actual per-provider sampling behavior; relevant generation/scoring source hashes; source/reference asset bytes under `frozen/`; Python and package versions; loaded instruction and public runtime code fingerprints; tool/repair budgets; renderer image; disabled web policy; scorer configuration; and concurrency.

All arms receive the same explicit 6000-token default, eight-turn-per-attempt default, two generation attempts, tools, character assets, and per-instance disabled web context. No environment mutation occurs while tasks run. OpenAI omits temperature while the other provider receives the requested value; this unavoidable provider sampling difference is recorded for every arm. The renderer is shared; the existing SandboxJob adapter serializes remote renderer execution with its lock while independent model calls run under a bounded semaphore.

Resume requires exact configuration equality, including selected scenarios/order, arms, model IDs/endpoints, code, assets, scorer, policy, and concurrency. Existing rows are checked against their frozen prompt, model, family, split, requirements, and image checksum. The advisory file lock rejects a second process or concurrent call using the same directory. Started items are checkpointed before model invocation. An item interrupted after start but before its result becomes an explicit terminal failure on resume; it is not silently issued again. Failure rows are also terminal. Use a new directory for a deliberate rerun.

`comparison_fingerprint(configuration)` excludes arm identity and concurrency, plus benchmark orchestration source, while retaining prompts, generation/scorer code, assets, runtime and policy. Compare this fingerprint before pairing a later tuned-only run with a baseline. Keep the selected prompts, ordering, provider-independent policy, scorer, and relevant implementation unchanged. The full configuration hash remains stricter for in-directory resume. Source changes inside agent/web code conservatively invalidate comparability; do not hide such edits through source normalization. The fingerprint does not claim sampling parity between providers; inspect each arm's recorded sampling settings.

## Output files and schemas

- `benchmark.json`: `{run_id, configuration_hash, comparison_fingerprint, configuration}` as described above.
- `frozen/<repository-relative-path>`: exact source, rubric, renderer and asset bytes recorded at run creation.
- `started.jsonl`: `{prompt_id, backend}` before each paired item begins.
- `manifest.jsonl`: one full `BufoGeneration.to_dict()` per completed/failed paired item. `benchmark_metadata` adds `run_id`, `configuration_hash`, `family`, `split`, `requirements`, `turns`, `status` (`completed` or `generation_error`), optional `failed_turn_id`, and `png_sha256` when pixels exist. Original error, model, latency, paths, native trace IDs, trajectory and other existing fields remain available.
- `images/<stable-prompt-and-arm-hash>/`: original PNG/SVG files, including intermediate renders. `partial.json` is an atomic, fsynced last-render checkpoint. At final-turn start the prior checkpoint rotates to `prior-turn.json`; corrupted checkpoints become `partial.damaged.json` and an explicit interrupted failure. A model error after a successful render preserves that image and its valid-render flag while generation status still records failure.
- `scored.jsonl`: one complete generation row per ID with `scores.scene_quality`, flushed/fsynced incrementally. Resume validates that saved generation fields match the original manifest and score identity/configuration/requirements match; no repeated judge calls for saved errors or successes.
- `summary.json`: run/configuration/comparison IDs, explicit selected scenario count/splits, and `arms`. Each arm includes `total`, `scored`, `judge_errors`, `missing_images`, `prior_turn_failures`, `generation_errors`, `invalid_renders`, `provisional_passes`, `provisional_pass_rate_all_attempts`, and `metric_means_scored_only`. Means explicitly use only successful judgments; pass rate uses **all attempts**, including failed generations/judges. A valid PNG alone does not turn a failed generation into a pass.
- `comparison.html`: small local view linking original pixels, prompt, metadata, raw score/error records, and summary. Raw SVG is never embedded.
- `weave-publication.json`: checkpointed `feedback` by generation ID (`call_id`), `evaluations` by arm (`ref`, `summary`, `scored_rows`), and sanitized `errors` by generation ID or `evaluation:<arm>`.

JSONL writers flush/fsync every record; a truncated final line can be repaired on resume, while malformed completed records fail visibly. Atomic JSON files use temporary-write/replace. External Weave publication cannot promise exactly-once delivery across a process crash after server acceptance and before local acknowledgement; already checkpointed publications are skipped. Generation requests are protected from automatic duplicate replay by the start ledger.

## Scene score and review contract

A successful result has `status="ok"`, `bufo_present`, numeric `similarity`, `anatomy`, `composition`, `readability`, exact `prompt_checks` with boolean `satisfied` and concrete `evidence`, `constraint_coverage`, `reason`, `quality_failure`, `failure_modes`, `rubric_version="bufo-scene-v1"`, `calibration_status="uncalibrated"`, and provisional `threshold=.65`. Results also retain raw judge JSON, exact requirements, scorer configuration, judge model, and generation ID.

Missing/extra/duplicate requirements, malformed booleans, nonfinite/out-of-range metrics, and missing visual evidence are rejected. The rubric explicitly says labels/captions do not prove visible actions, physical contact or pointing. Candidate PNGs are compared with the original Bufo identity reference. Invalid-render flags do not discard existing pixels; such pixels are judged, with `invalid_render` retained as a quality failure.

`judge_error`, `missing_image`, and `prior_turn_failure` results retain explicit status/failure modes and null unavailable metrics. Judge exceptions do not become valid zeros. Error messages retain exception type plus a generic explanation, avoiding provider payload/credential leakage. The existing recovered negative human annotation remains untouched; one negative point does not calibrate this rubric.

Native publication attaches saved scores as Weave scorer feedback to visual generation calls and creates native evaluations per arm. It replays saved judgments rather than paying for a second judge pass. Visual traces retain unreadable-image errors with `image=None`. The evaluation wrapper verifies exact row count and equality of every saved score before recording an evaluation as complete; exceptions silently converted to empty results by Weave are detected and persisted as publication errors.

Scene review adds anatomy, composition and readability (0–4), `requirements_satisfied`, and `acceptance` (`accept`, `reject`, `needs_review`) alongside existing identity/prompt/failure fields. These are review fields, not automatic training acceptance. Task1's explicit review provenance/export gates remain unchanged.

## Verification and review evidence

Red phase: both new suites failed collection because benchmark/scorer modules did not exist. Later regression for altered scored requirements failed because stale scored rows were initially accepted, then passed after source-row and score-binding validation. Loaded-code snapshot hardening exposed unstable marshal serialization during repeated execution; stable public bytecode-field hashing resolved the resume regression.

Final command:

```text
.venv/bin/pytest -q tests/test_benchmark.py tests/test_scene_quality.py tests/test_trajectories.py tests/test_training_contract.py tests/test_pipeline.py tests/test_ui.py tests/test_identity.py
74 passed in 2.53s
```

Known warnings are installed Weave/Pydantic deprecation warnings. Tests cover partitions/counts, exact requirements, invalid numerics, visible-action failure, invalid-image retention, judge exceptions, paired bounded concurrency, terminal generation failures, matching/mismatching resume, process-directory locking, interrupted/corrupt checkpoints, model-failure-after-render retention, stale-score rejection, native-eval coverage, safe corrupt-image replay, prior-turn exclusion, and cross-run fingerprints, plus existing trajectory/chat/UI/identity regressions.

```text
.venv/bin/ruff check --fix src/bufogen/benchmark.py src/bufogen/scorers/scene_quality.py src/bufogen/agent.py src/bufogen/generation.py src/bufogen/web_context.py src/bufogen/observability.py tests/test_benchmark.py tests/test_scene_quality.py
All checks passed!
```

Files were formatted with Ruff. Independent review fixes covered atomic/damaged checkpoints, prior-turn image attribution, separate queue criteria, and native evaluation coverage. Root communicated that the refresh found no remaining important code issues. Live pilot/full-run counts and remote publication evidence belong to the root-owned run artifacts, not the mocked checks above.
