# Frozen Experiment Runtimes + Latency Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every prior Forge experiment re-executable from its own saved snapshot, then add per-model-call instrumentation and a thinking toggle so the Gemma latency hypothesis can be tested one variable at a time.

**Architecture:** Stage 0 adds one restore script that rebuilds a full project tree per run from `runs/<exp>/<run>/frozen/` and proves it with that run's own tests and an offline re-score. Stage 1 wraps the two openai-agents model classes to record `finish_reason`, usage and timing per call, adds an opt-in `enable_thinking=false` body for CoreWeave models, threads one flag through agent → benchmark → CLI, and runs a three-arm paid experiment. No default behaviour changes in this plan.

**Tech Stack:** Python 3.12, uv, openai-agents 0.20.0, openai 2.54.0, httpx `MockTransport`, pytest (asyncio auto).

**Spec:** `/Users/aayush.jaiswal/.claude/plans/look-atthis-project-i-noble-scroll.md` (Stage 0 and Stage 1 sections). Stages 2+ get their own plans after the Stage 1 experiment.

## Global Constraints

- Runtime budgets stay: `max_turns=100`, studio request timeout 1800 s, no dollar cap.
- No shared module (`agent.py`, `models.py`, `benchmark.py`, `generation.py`, `cli.py`) may change until Task 3 has passed for every restored runtime.
- `BUFO_DISABLE_THINKING` defaults to **false**; `max_tokens` default stays **6000**; the outer retry loop at `agent.py:322-347` stays.
- The thinking body `{"chat_template_kwargs": {"enable_thinking": False}}` is sent only when `provider == "coreweave"`; never for `openai` or `forge`.
- Existing files under `runs/` are read-only evidence. Restored trees and copies go in new directories.
- The project is not a git repository yet (git init is Stage 7). Each "Checkpoint" step appends a dated entry to `docs/implementation-progress.md` instead of committing.
- No new paid calls in Stage 0. Stage 1 paid calls are limited to Task 7's 3 × 12 Gemma generations plus their scoring.

## Review Focus

1. A frozen file that is listed in `source_hashes` but missing from `frozen/` → the restore must fail naming the path, not silently fall back to today's copy.
2. A restored tree whose *unhashed* modules (`generation.py`, `settings.py`, `cli.py`, `telemetry.py`, `observability.py`) are newer than the frozen hashed ones → tests from inside the tree must reveal incompatibility; `RUNTIME.json` must list which files came from the current tree.
3. Re-scoring a restored run must never issue a judge request; any row lacking a cached score must raise before network use.
4. A Gemma response with `finish_reason="length"` and no tool call → the row must still be written with `truncated_calls >= 1`, not lost to an exception.
5. A forge-proxy or OpenAI request while thinking-disable is on → the request body must contain no `chat_template_kwargs`.

---

## File Structure

- Create `scripts/restore_experiment_runtime.py` — one responsibility: build and hash-verify a restored tree for one run dir.
- Create `tests/test_restore_runtime.py`.
- Create `src/bufogen/recording_model.py` — `ModelCall` record + the two recording model subclasses.
- Create `tests/test_recording_model.py`.
- Modify `src/bufogen/generation.py` (two fields), `src/bufogen/models.py` (one method), `src/bufogen/agent.py` (model construction, extra_body merge, row fields), `src/bufogen/benchmark.py` (`_configuration`, `run_benchmark` pass-through), `src/bufogen/cli.py` (flag), `src/bufogen/sandbox.py` (`renderer_wheel` cache).
- Modify `tests/test_forge_cli.py`, `tests/test_benchmark.py`, `tests/test_sandbox.py`.

---

### Task 1: Restore script with hash verification

**Files:**
- Create: `scripts/restore_experiment_runtime.py`
- Test: `tests/test_restore_runtime.py`

**Interfaces:**
- Produces: `restore(run_dir: Path, target: Path, *, project_root: Path) -> dict` (returns the `RUNTIME.json` dict); CLI `python scripts/restore_experiment_runtime.py RUN_DIR TARGET`. Raises `RestoreError(ValueError)` on any hash mismatch, missing frozen file, or non-empty `target`.
- `RUNTIME.json` keys: `run_dir, run_id, configuration_hash, restored: {path: sha256}` (from `frozen/`), `from_current_tree: [paths]`, `recorded_packages` (= `configuration.runtime.packages`), `installed_packages` (filled by Task 2, initially `null`), `package_differences` (initially `null`).
- `env.frozen` lines: `SCENE_JUDGE_BACKEND=<configuration.scorer.judge_backend>`, `SCENE_JUDGE_MODEL=<configuration.scorer.judge_model>`, `BUFO_WEB_SEARCH=false`, and `OPEN_MODEL/SOL_MODEL/OPEN_TUNED_MODEL/...` = each arm's `model` keyed by its `model_env`. Secrets never written.

- [ ] **Step 1: Write the failing tests**

Fixture `fake_project(tmp_path)`: a project root with `pyproject.toml`, `uv.lock`, `.python-version`, `src/bufogen/{agent.py,settings.py}`, `tests/test_x.py`, `docs/forge-x/a.py`, and a run dir whose `benchmark.json` has `configuration.source_hashes = {"src/bufogen/agent.py": sha("OLD")}` and `frozen/src/bufogen/agent.py` = `"OLD"` while the project's `agent.py` = `"NEW"`.

```python
def test_frozen_file_overrides_current_and_hashes_verify(fake_project, tmp_path):
    runtime = restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)
    assert (tmp_path / "t/src/bufogen/agent.py").read_text() == "OLD"
    assert runtime["restored"] == {"src/bufogen/agent.py": sha("OLD")}
    assert "src/bufogen/settings.py" in runtime["from_current_tree"]
    assert (tmp_path / "t/tests/test_x.py").exists() and (tmp_path / "t/docs/forge-x/a.py").exists()

def test_missing_frozen_file_fails_naming_path(fake_project, tmp_path):
    (fake_project.run / "frozen/src/bufogen/agent.py").unlink()
    with pytest.raises(RestoreError, match="src/bufogen/agent.py"):
        restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)

def test_tampered_frozen_file_fails(fake_project, tmp_path):
    (fake_project.run / "frozen/src/bufogen/agent.py").write_text("TAMPERED")
    with pytest.raises(RestoreError, match="sha256"):
        restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)

def test_refuses_nonempty_target(fake_project, tmp_path):
    (tmp_path / "t").mkdir(); (tmp_path / "t/x").write_text("x")
    with pytest.raises(RestoreError, match="empty"):
        restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)

def test_env_frozen_has_judge_and_arm_models_but_no_secrets(fake_project, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    restore(fake_project.run, tmp_path / "t", project_root=fake_project.root)
    env = (tmp_path / "t/env.frozen").read_text()
    assert "SCENE_JUDGE_BACKEND=closed_quality_bar" in env and "OPEN_MODEL=google/gemma-4-26B-A4B-it" in env
    assert "sk-secret" not in env and "API_KEY" not in env
```

Also include frozen `docs/` copies: if `frozen/docs/...` exists it overlays the current `docs/`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_restore_runtime.py -v`
Expected: FAIL (`ModuleNotFoundError` / import of `restore`). Import the script via `importlib.util.spec_from_file_location`, as `tests/test_contact_experiment.py:13-15` does.

- [ ] **Step 3: Implement `restore` in `scripts/restore_experiment_runtime.py`**

Copy order: (a) scaffolding from `project_root` — `pyproject.toml uv.lock .python-version src/ scripts/ config/ assets/ tests/ docs/` excluding `__pycache__`, `.venv`, `runs`, `wandb`, `.env`; (b) overlay every file under `run_dir/frozen/`; (c) for each `source_hashes` entry, require the frozen copy exists and its sha256 equals the recorded value, else `RestoreError`; (d) `from_current_tree` = every `src/**/*.py` path not in `source_hashes`. Write `RUNTIME.json`, `env.frozen`, and a `README.md` with the Task 2 commands. Do not touch `run_dir`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_restore_runtime.py -v`
Expected: 5 passed.

- [ ] **Step 5: Checkpoint** — append "Stage 0 Task 1: restore script + 5 tests passing" to `docs/implementation-progress.md`.

---

### Task 2: Build venvs and record package differences for the required runtimes

**Files:**
- Modify: `scripts/restore_experiment_runtime.py` (add `--sync` mode)
- Output (not source): `runs/restored/<run-slug>/` per run

**Interfaces:**
- Consumes: `restore(...)` from Task 1.
- Produces: `python scripts/restore_experiment_runtime.py RUN_DIR TARGET --sync` runs `uv sync --locked --extra rl` with `UV_PROJECT_ENVIRONMENT=TARGET/.venv` and `cwd=TARGET`, then fills `RUNTIME.json.installed_packages` from `TARGET/.venv/bin/python -m pip freeze` (or `uv pip freeze --python TARGET/.venv/bin/python`) and `package_differences` = `{name: {recorded, installed}}` for each name in `recorded_packages` that differs. When a `wandb/run-*/files/requirements.txt` exists whose run time precedes the run's first manifest row, also diff `openai-agents, weave, cwsandbox, openpipe-art, openai` against it.

- [ ] **Step 1: Restore and sync the required runs**

```bash
for r in forge-loop/heldout-baselines forge-gap/confirmation-selected \
         forge-iteration2/validation-candidates forge-contact-pilot/v2-development \
         forge-contact-pilot/comparison/svg forge-contact-pilot/comparison/contact \
         forge-contact-pilot/learnability/fit-baseline forge-contact-pilot/learnability/teacher; do
  uv run python scripts/restore_experiment_runtime.py runs/$r runs/restored/${r//\//__} --sync
done
```
Expected: 8 targets, each with `.venv/`, `RUNTIME.json`, `env.frozen`. Any `RestoreError` stops here and is reported verbatim; do not edit frozen files to make it pass.

- [ ] **Step 2: Confirm hashed imports load from the restored tree**

Run (per target): `cd runs/restored/<slug> && BUFO_ROOT=$PWD .venv/bin/python -c "import bufogen, bufogen.agent as a; print(a.__file__)"`
Expected: path inside the target, no ImportError.

- [ ] **Step 3: Checkpoint** — ledger entry listing each target and its `package_differences` (an empty dict is a result, not an omission).

---

### Task 3: Prove each restored runtime executes

**Files:**
- Create per target: `runs/restored/<slug>/VERIFICATION.md`

**Interfaces:**
- Consumes: Task 2 targets.

- [ ] **Step 1: Run the experiment tests from inside each tree**

```bash
cd runs/restored/<slug>
set -a; . ./env.frozen; set +a
BUFO_ROOT=$PWD .venv/bin/python -m pytest -q <tests>
```
`<tests>`: contact-pilot targets → `tests/test_contact_*.py`; iteration-2 → `tests/test_iteration2_*.py tests/test_plan_continuation.py`; forge-gap → `tests/test_gap_*.py tests/test_lr_evaluation.py`; forge-loop → `tests/test_benchmark.py tests/test_scene_quality.py`.
Expected: all pass. A failure is recorded verbatim with the failing test id and the file(s) in `from_current_tree` it touches.

- [ ] **Step 2: Offline re-score a copy of each benchmark run**

```bash
cp -R "$PROJECT/runs/<run>" ./rescore-copy        # PROJECT = /Users/aayush.jaiswal/Desktop/cw-bufo-gen
BUFO_ROOT=$PWD WANDB_API_KEY= OPENAI_API_KEY= .venv/bin/bufogen --stage benchmark-score --output ./rescore-copy
```
Credentials are blanked deliberately: a fully cached run should complete without constructing a judge client. If it fails only because a credential is read eagerly, record that and rerun with credentials present, confirming no judge request was made (no new Weave calls; `scored.jsonl` byte-identical). Expected: exits 0, prints `rescore-copy/comparison.html`, and `diff <(jq -S . rescore-copy/summary.json) <(jq -S . "$PROJECT/runs/<run>/summary.json")` is empty. A "Scoring configuration mismatch" or a missing-key error is a Stage 0 failure.

- [ ] **Step 3: Contact-pilot validation entry points**

From the contact-pilot targets run the offline validation modes of `docs/forge-contact-pilot/evaluate.py` and `docs/forge-contact-pilot/fit_compare.py` (their defaults are validation-only per `tests/test_contact_fit_compare.py::test_default_is_validation_only`). Expected: exit 0 against the frozen protocol.

- [ ] **Step 4: Write `VERIFICATION.md` per target** — commands, exit codes, test counts, summary diff result, package differences, and the list of current-tree files used.

- [ ] **Step 5: Gate + checkpoint** — ledger entry: "Stage 0 gate passed for N/8 runtimes" with failures listed. If any target fails, stop and report to the user before Task 4.

---

### Task 4: Per-call recording models

**Files:**
- Create: `src/bufogen/recording_model.py`
- Modify: `src/bufogen/generation.py` (fields after `trajectory`)
- Test: `tests/test_recording_model.py`

**Interfaces:**
- Produces:
  - `@dataclass ModelCall: index: int; finish_reason: str | None; input_tokens: int; output_tokens: int; reasoning_tokens: int | None; reasoning_chars: int; latency_s: float; served_model: str | None` with `to_dict()`.
  - `class RecordingChatCompletionsModel(OpenAIChatCompletionsModel)` and `class RecordingResponsesModel(OpenAIResponsesModel)`, both `__init__(self, *, model, openai_client, calls: list[ModelCall])`, overriding `_fetch_response` (signatures as installed, see Task notes) and appending one `ModelCall` per non-streamed call.
  - `BufoGeneration.model_calls: list[dict] = field(default_factory=list)` and `BufoGeneration.truncated_calls: int = 0`.

Mapping: Chat Completions → `finish_reason = choices[0].finish_reason`; `reasoning_tokens = usage.completion_tokens_details.reasoning_tokens` if present else `None`; `reasoning_chars = len(getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or "")` (vLLM fields, via `model_extra`). Responses → `finish_reason = "length"` when `status == "incomplete"` and `incomplete_details.reason == "max_output_tokens"`, else `"stop"`; `reasoning_tokens = usage.output_tokens_details.reasoning_tokens`. Streaming calls (`stream=True`) are passed through unrecorded.

- [ ] **Step 1: Write the failing tests** (httpx `MockTransport` into `AsyncOpenAI(http_client=...)`, as `tests/test_forge_cli.py:398-404`)

```python
async def test_chat_records_finish_reason_usage_and_reasoning():
    calls = []
    model = RecordingChatCompletionsModel(model="m", openai_client=client_returning(
        finish_reason="length", completion_tokens=6000,
        details={"reasoning_tokens": 4800}, message_extra={"reasoning_content": "x" * 42},
        model_name="served-gemma"), calls=calls)
    await model.get_response(**minimal_get_response_kwargs())
    assert calls[0].finish_reason == "length" and calls[0].output_tokens == 6000
    assert calls[0].reasoning_tokens == 4800 and calls[0].reasoning_chars == 42
    assert calls[0].served_model == "served-gemma" and calls[0].latency_s >= 0

async def test_chat_missing_details_gives_none_not_zero():
    ...  # no completion_tokens_details → reasoning_tokens is None, reasoning_chars == 0

async def test_responses_incomplete_maps_to_length():
    ...  # status="incomplete", incomplete_details={"reason":"max_output_tokens"} → "length"

def test_generation_roundtrip_with_new_fields_and_legacy_rows():
    row = BufoGeneration(**legacy_row_dict_without_new_fields())
    assert row.model_calls == [] and row.truncated_calls == 0
```
`minimal_get_response_kwargs()` builds `system_instructions="s", input="hi", model_settings=ModelSettings(), tools=[], output_schema=None, handoffs=[], tracing=ModelTracing.DISABLED, previous_response_id=None, conversation_id=None, prompt=None`.

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_recording_model.py -v` → FAIL (import error).

- [ ] **Step 3: Implement** — call `super()._fetch_response(...)` with all args, time it with `time.monotonic()`, append the record when the result is a `ChatCompletion` / `Response`. Add the two `BufoGeneration` fields.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_recording_model.py tests/test_core.py -v` → all pass.

- [ ] **Step 5: Checkpoint** — ledger entry.

---

### Task 5: Wire recording + opt-in thinking toggle into the agent

**Files:**
- Modify: `src/bufogen/models.py` (`BackendConfig`), `src/bufogen/agent.py:76-104` (constructor), `:282-308` (model + settings), `:318-372` (usage loop + row)
- Test: `tests/test_forge_cli.py` (extend the existing real-SDK test), new cases in `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 4 `ModelCall`, recording models, new `BufoGeneration` fields.
- Produces:
  - `BackendConfig.thinking_extra_body(self, disable: bool) -> dict | None` → `{"chat_template_kwargs": {"enable_thinking": False}}` iff `disable and self.provider == "coreweave"`, else `None`.
  - `BufoAgent.__init__(..., disable_thinking: bool | None = None)`; `None` resolves from env `BUFO_DISABLE_THINKING` in `{"1","true","yes","on"}` (default false). Stored as `self.disable_thinking: bool`.
  - `extra_body` = shallow merge of `request_extra_body(group_id)` and `thinking_extra_body(...)`, `None` when both are `None`.
  - Row: `model_calls = [c.to_dict() for c in calls]` across both attempts (indices continue), `truncated_calls = sum(c.finish_reason == "length")`. Log `warning("Model output truncated (%d calls)")` when `truncated_calls`.
  - Provenance gains `"disable_thinking": self.disable_thinking`. `ImageReplyModel` keeps wrapping the (now recording) inner model unchanged.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("backend_name,disable,expected", [
    ("open_base", True, {"chat_template_kwargs": {"enable_thinking": False}}),
    ("open_base", False, None),
    ("closed_quality_bar", True, None),
    ("forge_proxy", True, None),
])
def test_thinking_body_only_for_coreweave(backend_name, disable, expected):
    assert load_backends()[backend_name].thinking_extra_body(disable) == expected
```
In `tests/test_forge_cli.py`, add a sibling of `test_agent_forwards_proxy_thread_on_tool_and_final_requests` using `load_backends()["open_base"]` with `disable_thinking=True`: assert every captured request has `chat_template_kwargs == {"enable_thinking": False}`, `row.model_calls` has one entry per request with the mocked `finish_reason`, and `row.trajectory["provenance"]["disable_thinking"] is True`. Extend the existing forge-proxy test: assert no request contains `chat_template_kwargs` even with `monkeypatch.setenv("BUFO_DISABLE_THINKING", "true")`.
Truncation case: mock returns `finish_reason="length"` with plain text and no tool call on the first request, then a render tool call, then text → `row.truncated_calls == 1` and the row is written.

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_forge_cli.py -k "agent_forwards or thinking" tests/test_pipeline.py -v` → new cases FAIL.

- [ ] **Step 3: Implement** in `models.py` and `agent.py` per Interfaces. Replace `model_class(model=..., openai_client=client)` with the recording subclass, passing a per-generation `calls` list. Leave `max_tokens` defaults and the retry loop as they are.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_forge_cli.py tests/test_pipeline.py tests/test_composition.py tests/test_trajectories.py -v` → all pass (existing tests unchanged).

- [ ] **Step 5: Checkpoint** — ledger entry noting `agent.py`/`models.py` hashes changed, so new benchmark dirs get a new `prompt_version`-independent `source_hashes`.

---

### Task 6: Thread the flag through benchmark config and CLI; wheel cache

**Files:**
- Modify: `src/bufogen/benchmark.py` (`_configuration`, `run_benchmark` ~`:300-420`), `src/bufogen/cli.py:78-90, 248-258`, `src/bufogen/sandbox.py:63-94`
- Test: `tests/test_benchmark.py`, `tests/test_forge_cli.py`, `tests/test_sandbox.py`

**Interfaces:**
- Consumes: `BufoAgent(disable_thinking=...)` from Task 5.
- Produces:
  - `run_benchmark(..., disable_thinking: bool = False)` passes it to every `BufoAgent`; `_configuration(..., disable_thinking: bool)` writes `policy["disable_thinking"]`. Because `policy` is part of `configuration_hash` and `comparison_fingerprint`, arms (a)/(b)/(c) of Task 7 are distinct frozen configs that cannot resume into each other.
  - `cli.py`: `--disable-thinking` (`store_true`), forwarded for `benchmark`, `run`, `demo`.
  - `renderer_wheel()`: after a verified download write `<wheel>.sha256`; on entry, if a `*.whl` with a sidecar whose content equals the file's sha256 exists in `.cache/render`, return it without any HTTP request.

- [ ] **Step 1: Write the failing tests**

```python
def test_configuration_records_disable_thinking_and_changes_fingerprint():
    a = _configuration(prompts(), [backend()], temperature=0.7, max_tokens=6000, max_turns=100,
                       concurrency=2, disable_thinking=False)
    b = _configuration(prompts(), [backend()], temperature=0.7, max_tokens=6000, max_turns=100,
                       concurrency=2, disable_thinking=True)
    assert a["policy"]["disable_thinking"] is False and b["policy"]["disable_thinking"] is True
    assert comparison_fingerprint(a) != comparison_fingerprint(b)

async def test_cli_forwards_disable_thinking(tmp_path, monkeypatch):
    ...  # pattern of test_generation_cli_forwards_custom_turn_budget; assert calls[-1]["disable_thinking"] is True

async def test_cached_verified_wheel_skips_pypi(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "ROOT", tmp_path)
    wheel = tmp_path / ".cache/render/resvg_py-0.5.0-cp38-abi3-manylinux_x86_64.whl"
    wheel.parent.mkdir(parents=True); wheel.write_bytes(b"w")
    wheel.with_name(wheel.name + ".sha256").write_text(hashlib.sha256(b"w").hexdigest())
    monkeypatch.setattr(sandbox.httpx, "AsyncClient", ExplodingClient)  # raises on construct
    assert await sandbox.renderer_wheel() == wheel

async def test_mismatched_sidecar_falls_back_to_network(...): ...
```
Update existing `_configuration(...)` calls in `tests/test_benchmark.py:267-278` to pass `disable_thinking=False`.

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_benchmark.py tests/test_forge_cli.py tests/test_sandbox.py -v` → new cases FAIL.

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Run full suite** — `uv run pytest -q && uv run ruff check src scripts && node --test tests/chat_ui.test.cjs` → pass (ruff on `tests/` is out of scope here; it is Stage 7).

- [ ] **Step 5: Checkpoint** — ledger entry.

---

### Task 7: Three-arm latency experiment (paid) and write-up

**Files:**
- Output: `runs/latency-a-baseline/`, `runs/latency-b-nothink/`, `runs/latency-c-nothink-3000/`
- Modify: `docs/implementation-progress.md`
- Create: `scripts/latency_report.py` (reads the three manifests, prints the table)

**Interfaces:**
- Consumes: `--disable-thinking`, `model_calls`, `truncated_calls` from Tasks 4–6.
- `latency_report.py RUN_DIR...` prints per arm: n, generation failures, rows with `truncated_calls > 0`, calls with `finish_reason == "length"` / total calls, median and p90 `latency_s`, median output tokens per call, median `reasoning_tokens` (or "not reported"), median `reasoning_chars`, median per-call latency, strict pass count from `scored.jsonl`.

- [ ] **Step 1: Generate** (same 12 train prompts, one arm each, nothing else varied)

```bash
uv run bufogen --stage benchmark --split train --limit 12 --backends open_base \
  --concurrency 4 --output runs/latency-a-baseline
uv run bufogen --stage benchmark --split train --limit 12 --backends open_base \
  --concurrency 4 --disable-thinking --output runs/latency-b-nothink
uv run bufogen --stage benchmark --split train --limit 12 --backends open_base \
  --concurrency 4 --disable-thinking --max-tokens 3000 --output runs/latency-c-nothink-3000
```
Expected: each dir has 12 manifest rows; `jq '.model_calls|length' manifest.jsonl` > 0 for every non-failed row. Confirm all three `benchmark.json` files share `prompt_hash`.

- [ ] **Step 2: Score** — `uv run bufogen --stage benchmark-score --output <dir>` for each (no `--publish`).

- [ ] **Step 3: Report** — `uv run python scripts/latency_report.py runs/latency-*` and paste the table into `docs/implementation-progress.md` under "Stage 1 latency experiment — 2026-10-xx".

- [ ] **Step 4: State the conclusion as evidence allows.** Write one of: "thinking explains truncation" (b removes `length` finishes and latency drops), "thinking is not the cause" (a shows `length` without reasoning tokens/chars and b does not change it), or "inconclusive" with the reason. Write separately whether (c) changed pass rate versus (b). Do **not** change any default in this plan; list the proposed default changes (thinking, `max_tokens`, outer retry loop) as three separate follow-up changes, each needing its own measurement.

- [ ] **Step 5: Checkpoint** — ledger entry with run dir paths and the conclusion.

---

## Self-review notes

- Spec coverage: Stage 0 steps 1–5 → Tasks 1–3 (step 5, in-flight job continuing from restored runtime, is enforced by Task 3's gate and recorded in the ledger). Stage 1 steps 1–5 → Tasks 4–7.
- Type consistency: `ModelCall`, `model_calls: list[dict]`, `truncated_calls: int`, `thinking_extra_body(disable: bool)`, `disable_thinking: bool | None` used identically in Tasks 4–7.
- Not covered by this plan (intentionally): changing defaults, removing the retry loop, any Stage 2+ work.
