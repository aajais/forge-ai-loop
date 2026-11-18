import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from bufogen.cli import dispatch, parser
from bufogen.models import load_backends


def arguments(stage, *extra):
    return parser().parse_args(["--stage", stage, *map(str, extra)])


async def test_proxy_http_metadata_resolved_model_and_thread(monkeypatch):
    from agents import ModelSettings, OpenAIChatCompletionsModel
    from agents.models.interface import ModelTracing

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    monkeypatch.delenv("FORGE_PROXY_MODEL", raising=False)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "object": "chat.completion",
                "created": 0,
                "model": "google/gemma-4-26B-A4B-it",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class MockClient(original):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    backend = load_backends()["forge_proxy"]
    async with backend.client() as client:
        model = OpenAIChatCompletionsModel(model=backend.model, openai_client=client)
        for prompt in ("Draw Bufo", "Add a hat"):
            await model.get_response(
                system_instructions="Rules",
                input=prompt,
                model_settings=ModelSettings(
                    extra_body=backend.request_extra_body("conversation-1")
                ),
                tools=[],
                output_schema=None,
                handoffs=[],
                tracing=ModelTracing.DISABLED,
            )
        assert client.bufo_serving_models == ["google/gemma-4-26B-A4B-it"]
    assert [r["metadata"] for r in requests] == [
        {"wandb.entity": "test-team", "wandb.thread_id": "conversation-1"}
    ] * 2
    assert all(r["model"] == "bufo-gen" for r in requests)
    assert load_backends()["open_base"].request_extra_body("conversation-1") is None
    assert load_backends()["closed_quality_bar"].request_extra_body("conversation-1") is None


def test_fingerprint_covers_model_transport_source():
    from bufogen.evaluation.benchmark import _source_hashes

    assert "src/bufogen/models.py" in _source_hashes()


async def test_benchmark_dispatch_heldout_and_explicit_checkpoint(tmp_path, monkeypatch):
    import bufogen.evaluation.benchmark as benchmark

    calls = []

    async def run(prompts, backends, output, **kwargs):
        calls.append((prompts, backends, kwargs))

    monkeypatch.setattr(benchmark, "run_benchmark", run)
    await dispatch(
        arguments(
            "benchmark",
            "--output",
            tmp_path,
            "--backends",
            "open_base",
            "closed_quality_bar",
            "--checkpoint",
            "sft_j1=wandb-artifact:///e/p/m:v1",
        )
    )
    prompts, arms, settings = calls[0]
    assert len(prompts) == 100 and {p["split"] for p in prompts} == {"heldout"}
    assert set(arms) == {"open_base", "closed_quality_bar", "sft_j1"}
    assert arms["sft_j1"].model.endswith(":v1")
    assert settings["max_tokens"] is None and settings["max_turns"] == 100
    with pytest.raises(ValueError, match="unique"):
        await dispatch(
            arguments("benchmark", "--backends", "open_base", "--checkpoint", "open_base=something")
        )
    assert len(calls) == 1


async def test_benchmark_without_backends_runs_only_the_given_checkpoints(tmp_path, monkeypatch):
    import bufogen.evaluation.benchmark as benchmark

    calls = []

    async def run(prompts, backends, output, **kwargs):
        calls.append(backends)

    monkeypatch.setattr(benchmark, "run_benchmark", run)
    await dispatch(
        arguments(
            "benchmark",
            "--output",
            tmp_path,
            "--checkpoint",
            "rl_q3=wandb-artifact:///e/p/rl:v5",
            "--checkpoint",
            "sft_q3=wandb-artifact:///e/p/sft:v1",
        )
    )
    assert set(calls[0]) == {"rl_q3", "sft_q3"}
    await dispatch(arguments("benchmark", "--output", tmp_path))
    assert set(calls[1]) == {"open_base", "closed_quality_bar"}
    assert all(c.base_model == "Qwen/Qwen3.8-27B" for c in calls[0].values())
    assert calls[0]["rl_q3"].model.endswith("rl:v5")
    with pytest.raises(ValueError, match="models.yaml"):
        await dispatch(arguments("benchmark", "--checkpoint", "nope=wandb-artifact:///e/p/x:v1"))
    assert len(calls) == 2


@pytest.mark.parametrize("stage", ["run", "demo"])
async def test_generation_cli_forwards_custom_turn_budget(stage, tmp_path, monkeypatch):
    import bufogen.cli as cli

    calls = []

    async def pipeline(prompts, backends, output, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(cli, "run_pipeline", pipeline)
    await dispatch(
        arguments(stage, "--output", tmp_path, "--theme", "Bufo gardening", "--max-turns", "150")
    )
    assert calls == [{"max_turns": 150, "disable_thinking": None}]


@pytest.mark.parametrize("stage", ["run", "demo", "benchmark"])
async def test_cli_forwards_disable_thinking(stage, tmp_path, monkeypatch):
    import bufogen.cli as cli
    import bufogen.evaluation.benchmark as benchmark

    calls = []

    async def pipeline(prompts, backends, output, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(cli, "run_pipeline", pipeline)
    monkeypatch.setattr(benchmark, "run_benchmark", pipeline)
    extra = ["--theme", "Bufo gardening"] if stage != "benchmark" else []
    await dispatch(arguments(stage, "--output", tmp_path, *extra))
    # run/demo defer to BUFO_DISABLE_THINKING (None); benchmark always freezes an explicit bool.
    assert calls[-1]["disable_thinking"] is (False if stage == "benchmark" else None)
    await dispatch(arguments(stage, "--output", tmp_path, *extra, "--disable-thinking"))
    assert calls[-1]["disable_thinking"] is True


def reviewed_inputs(tmp_path):
    from test_training_contract import review
    from test_trajectories import captured_row

    from bufogen.generation import Manifest

    row = captured_row()
    row.backend = "closed_quality_bar"
    row.training_review = review(row, reviewer_kind="assistant")
    manifest = tmp_path / "manifest.jsonl"
    Manifest(manifest).append(row)
    decisions = tmp_path / "reviews.jsonl"
    decisions.write_text(json.dumps(row.training_review) + "\n")
    return row, manifest, decisions


async def test_export_existing_manifest_preserves_real_calls_and_review(tmp_path):
    row, manifest, reviews = reviewed_inputs(tmp_path)
    output = tmp_path / "export"
    await dispatch(
        arguments(
            "export-reviewed", "--manifest", manifest, "--reviews", reviews, "--output", output
        )
    )
    data = json.loads((output / "reviewed-examples.jsonl").read_text())
    assert data["source_generation_id"] == row.id
    assert data["training_review"]["reviewer_kind"] == "assistant"
    assert data["messages"][2]["tool_calls"][0]["id"] == "actual-call-1"
    with pytest.raises(ValueError, match="Duplicate"):
        await dispatch(
            arguments(
                "export-reviewed",
                "--manifest",
                manifest,
                "--reviews",
                reviews,
                "--reviews",
                reviews,
                "--output",
                output,
            )
        )


async def test_export_rejects_partition_relabel_and_unreviewed(tmp_path):
    row, manifest, reviews = reviewed_inputs(tmp_path)
    decision = json.loads(reviews.read_text())
    decision.update(prompt_family="concert_stage", partition="heldout")
    row.prompt_id = "train-concert_stage-01"
    decision["prompt_id"] = row.prompt_id
    manifest.write_text(json.dumps(row.to_dict()) + "\n")
    reviews.write_text(json.dumps(decision) + "\n")
    with pytest.raises(ValueError, match="conflicts"):
        await dispatch(
            arguments(
                "export-reviewed",
                "--manifest",
                manifest,
                "--reviews",
                reviews,
                "--output",
                tmp_path / "out",
            )
        )
    with pytest.raises(ValueError, match="--reviews"):
        await dispatch(arguments("export-reviewed", "--manifest", manifest))


async def test_invalid_forge_import_never_constructs_client(tmp_path, monkeypatch):
    import bufogen.cli as cli

    monkeypatch.setattr(
        cli, "forge_client", lambda _: pytest.fail("Client constructed before local validation")
    )
    source = tmp_path / "invalid.jsonl"
    source.write_text("{}\n")
    with pytest.raises(ValueError):
        await dispatch(
            arguments(
                "forge-import",
                "--source",
                source,
                "--dataset-name",
                "data",
                "--name",
                "model",
                "--project-id",
                "project",
            )
        )


async def test_forge_status_only_polls_and_submit_has_explicit_config(tmp_path, monkeypatch):
    import bufogen.cli as cli
    import bufogen.workflows.forge_loop as loop

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    calls = []
    monkeypatch.setattr(cli, "forge_client", lambda _: Client())
    monkeypatch.setattr(
        loop,
        "resume_training",
        lambda path, client: calls.append(("poll", path)) or {"status": "training"},
    )
    monkeypatch.setattr(
        loop,
        "submit_training",
        lambda path, client, **kw: calls.append(("submit", kw)) or {"status": "queued"},
    )
    await dispatch(arguments("forge-status", "--forge-state", tmp_path / "state.json"))
    assert calls == [("poll", tmp_path / "state.json")]
    await dispatch(
        arguments("forge-submit", "--forge-state", tmp_path / "state.json", "--epochs", 2)
    )
    assert calls[-1] == ("submit", {"epochs": 2, "batch_size": 2})


async def test_rl_cli_requires_calibration_and_paired_continuation(tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        await dispatch(arguments("train-rl", "--dataset-ref", "e/p/d:v0", "--name", "run"))
    with pytest.raises(ValueError, match="both"):
        await dispatch(
            arguments(
                "train-rl",
                "--dataset-ref",
                "e/p/d:v0",
                "--name",
                "run",
                "--calibration",
                tmp_path / "calibration.json",
                "--initial-checkpoint",
                "e/p/m:v1",
            )
        )


def benchmark_fixture(path, *, fingerprint, backend="student", run_id="one", split="heldout"):
    from test_trajectories import captured_row

    from bufogen.evaluation.benchmark import _summary
    from bufogen.evaluation.scorers.scene_quality import failure_result

    path.mkdir()
    row = replace(captured_row(), backend=backend)
    identity = {
        "run_id": run_id,
        "configuration_hash": "hash",
        "comparison_fingerprint": fingerprint,
    }
    row.benchmark_metadata = {
        "run_id": run_id,
        "configuration_hash": "hash",
        "family": "family",
        "split": split,
        "requirements": ["Draw Bufo"],
        "status": "completed",
    }
    settings = {
        **identity,
        "configuration": {
            "prompts": [
                {
                    "id": "p",
                    "prompt": row.prompt,
                    "family": "family",
                    "split": split,
                    "requirements": ["Draw Bufo"],
                }
            ],
            "arms": [{"name": backend, "model": row.model}],
            "scorer": {"judge_model": "qwen"},
        },
    }
    (path / "benchmark.json").write_text(json.dumps(settings))
    (path / "manifest.jsonl").write_text(json.dumps(row.to_dict()) + "\n")
    row.scores["scene_quality"] = failure_result("judge_error") | {
        "generation_id": row.id,
        "requirements": ["Draw Bufo"],
        "scorer_configuration": settings["configuration"]["scorer"],
    }
    (path / "scored.jsonl").write_text(json.dumps(row.to_dict()) + "\n")
    (path / "summary.json").write_text(json.dumps(_summary([row], settings)))
    return path


def test_report_separates_fingerprints_and_never_pools_duplicate_arms(tmp_path):
    from bufogen.workflows.loop_report import build_loop_report, write_loop_report

    one = benchmark_fixture(tmp_path / "one", fingerprint="same")
    two = benchmark_fixture(
        tmp_path / "two", fingerprint="different", backend="teacher", run_id="two", split="train"
    )
    three = benchmark_fixture(tmp_path / "three", fingerprint="same", run_id="three")
    report = build_loop_report(tmp_path, benchmark_dirs=[one, two, three])
    assert len(report["comparison_groups"]) == 2
    assert report["comparison_groups"][0]["status"] == "ambiguous_repeated_arms"
    assert report["benchmarks"][0]["arms"]["student"]["total"] == 1
    assert report["benchmarks"][1]["split_kind"] == "pilot/development"
    assert report["phases"]["Improve"]["status"] == "not_started"
    assert report["phases"]["Curate"]["status"] == "not_started"
    html_path = write_loop_report(tmp_path, tmp_path / "report", benchmark_dirs=[one, two])
    text = html_path.read_text()
    assert "0/1" in text and "qwen" in text and "no gpt-6.1-sol parity claim" in text


def test_report_rejects_mismatched_summary_and_conflicting_reviews(tmp_path):
    from bufogen.workflows.loop_report import build_loop_report

    one = benchmark_fixture(tmp_path / "one", fingerprint="same")
    summary = json.loads((one / "summary.json").read_text())
    summary["comparison_fingerprint"] = "wrong"
    (one / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Summary"):
        build_loop_report(tmp_path, benchmark_dirs=[one])


@pytest.mark.parametrize("tool_rounds", [1, 12])
async def test_agent_forwards_proxy_thread_on_tool_and_final_requests(
    tmp_path, monkeypatch, tool_rounds
):
    from agents import trace

    import bufogen.integrations.observability as observability
    import bufogen.integrations.telemetry as telemetry
    from bufogen.agents.agent import BufoAgent
    from bufogen.rendering.renderer import RenderResult

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    monkeypatch.setenv("BUFO_DISABLE_THINKING", "true")
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(observability, "record_generation", lambda row: None)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        turn = len(requests)
        is_tool_turn = turn % (tool_rounds + 1) != 0
        message = {"role": "assistant", "content": "Here is Bufo!"}
        if is_tool_turn:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"render-{turn}",
                        "type": "function",
                        "function": {
                            "name": "render_svg",
                            "arguments": json.dumps({"svg": "<svg/>", "emoji_name": ":bufo-test:"}),
                        },
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": f"chat-{turn}",
                "object": "chat.completion",
                "created": 0,
                "model": "resolved-gemma",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if is_tool_turn else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class MockClient(original):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    class Renderer:
        async def render(self, svg):
            return RenderResult(b"png", True, None)

    backend = load_backends()["forge_proxy"]
    async with backend.client() as client:
        agent = BufoAgent(backend, Renderer(), tmp_path, client=client, web_search_enabled=False)
        with trace("local test", disabled=True):
            first = await agent._generate("Draw Bufo", group_id="stable-thread")
            second = await agent._generate(
                "Add a hat", group_id="stable-thread", history=[first.to_dict()]
            )
    assert first.valid and second.valid and len(requests) == 2 * (tool_rounds + 1)
    assert all(
        r["metadata"] == {"wandb.entity": "test-team", "wandb.thread_id": "stable-thread"}
        for r in requests
    )
    assert all("chat_template_kwargs" not in r for r in requests)
    # OpenAI reasoning targets behind the proxy reject max_tokens.
    assert all("max_tokens" not in r for r in requests)
    assert all(r["max_completion_tokens"] == agent.max_tokens for r in requests)
    assert len(first.model_calls) == tool_rounds + 1
    assert second.trajectory["provenance"]["requested_alias"] == backend.model
    assert second.trajectory["provenance"]["resolved_serving_models"] == ["resolved-gemma"]


def _chat_response(turn, tool, finish_reason):
    message = {"role": "assistant", "content": "Here is Bufo!"}
    if tool:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"render-{turn}",
                    "type": "function",
                    "function": {
                        "name": "render_svg",
                        "arguments": json.dumps({"svg": "<svg/>", "emoji_name": ":bufo-test:"}),
                    },
                }
            ],
        }
    return httpx.Response(
        200,
        json={
            "id": f"chat-{turn}",
            "object": "chat.completion",
            "created": 0,
            "model": "served-model",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


async def _run_open_base(tmp_path, monkeypatch, script, **agent_kwargs):
    from agents import trace
    from openai import AsyncOpenAI

    import bufogen.integrations.observability as observability
    import bufogen.integrations.telemetry as telemetry
    from bufogen.agents.agent import BufoAgent
    from bufogen.rendering.renderer import RenderResult

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(observability, "record_generation", lambda row: None)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        tool, finish_reason = script[len(requests) - 1]
        return _chat_response(len(requests), tool, finish_reason)

    class Renderer:
        async def render(self, svg):
            return RenderResult(b"png", True, None)

    backend = load_backends()["open_base"]
    client = AsyncOpenAI(
        api_key="test-key",
        base_url="http://mock.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    agent = BufoAgent(
        backend, Renderer(), tmp_path, client=client, web_search_enabled=False, **agent_kwargs
    )
    with trace("local test", disabled=True):
        row = await agent._generate("Draw Bufo", group_id="thread-1")
    return row, requests


async def test_agent_open_base_thinking_off_and_records_each_call(tmp_path, monkeypatch):
    row, requests = await _run_open_base(
        tmp_path,
        monkeypatch,
        [(True, "tool_calls"), (False, "stop")],
        disable_thinking=True,
    )
    assert row.valid and len(requests) == 2
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in requests)
    assert [c["finish_reason"] for c in row.model_calls] == ["tool_calls", "stop"]
    assert [c["index"] for c in row.model_calls] == [0, 1]
    assert row.truncated_calls == 0
    assert row.trajectory["provenance"]["disable_thinking"] is True


async def test_agent_open_base_default_sends_no_thinking_body(tmp_path, monkeypatch):
    monkeypatch.delenv("BUFO_DISABLE_THINKING", raising=False)
    row, requests = await _run_open_base(
        tmp_path, monkeypatch, [(True, "tool_calls"), (False, "stop")]
    )
    assert all("chat_template_kwargs" not in r for r in requests)
    assert row.trajectory["provenance"]["disable_thinking"] is False


async def test_agent_counts_truncated_calls_across_attempts(tmp_path, monkeypatch):
    row, requests = await _run_open_base(
        tmp_path,
        monkeypatch,
        [(False, "length"), (True, "tool_calls"), (False, "stop")],
    )
    assert len(requests) == 3
    assert row.truncated_calls == 1
    assert [c["finish_reason"] for c in row.model_calls] == ["length", "tool_calls", "stop"]
    assert (tmp_path / f"{row.id}.svg").exists()


async def test_current_train_contract_cannot_launder_frozen_heldout_generation(tmp_path):
    row, manifest, reviews = reviewed_inputs(tmp_path)
    decision = json.loads(reviews.read_text())
    row.prompt_id = "train-concert_stage-01"
    row.trajectory["provenance"]["prompt_id"] = row.prompt_id
    row.benchmark_metadata = {"split": "heldout", "family": "original-heldout-family"}
    decision.update(prompt_id=row.prompt_id, partition="train", prompt_family="concert_stage")
    manifest.write_text(json.dumps(row.to_dict()) + "\n")
    reviews.write_text(json.dumps(decision) + "\n")
    with pytest.raises(ValueError, match="conflicts"):
        await dispatch(
            arguments(
                "export-reviewed",
                "--manifest",
                manifest,
                "--reviews",
                reviews,
                "--output",
                tmp_path / "export",
            )
        )
    assert not (tmp_path / "export/reviewed-examples.jsonl").exists()


async def test_report_default_state_follows_selected_root_even_with_env(tmp_path, monkeypatch):
    root = tmp_path / "other"
    root.mkdir()
    (root / "forge-state.json").write_text(json.dumps({"status": "local-only"}))
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"status": "wrong-project"}))
    monkeypatch.setenv("FORGE_STATE_PATH", str(stale))
    output = tmp_path / "report"
    await dispatch(arguments("loop-report", "--loop-root", root, "--output", output))
    report = json.loads((output / "loop-report.json").read_text())
    assert report["phases"]["Improve"]["status"] == "local-only"
    await dispatch(
        arguments("loop-report", "--loop-root", root, "--forge-state", stale, "--output", output)
    )
    assert (
        json.loads((output / "loop-report.json").read_text())["phases"]["Improve"]["status"]
        == "wrong-project"
    )


def test_report_rejects_foreign_rows_under_genuine_summary_identity(tmp_path):
    from bufogen.workflows.loop_report import build_loop_report

    directory = benchmark_fixture(tmp_path / "run", fingerprint="same")
    row = json.loads((directory / "manifest.jsonl").read_text())
    row.update(prompt_id="foreign-prompt", backend="foreign-arm", model="foreign-model")
    row["benchmark_metadata"] = {"run_id": "foreign-run"}
    (directory / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="frozen"):
        build_loop_report(tmp_path, benchmark_dirs=[directory])


@pytest.mark.parametrize("mutation", ["total", "arm", "score"])
def test_report_rejects_summary_coverage_or_metric_drift(tmp_path, mutation):
    from bufogen.workflows.loop_report import build_loop_report

    directory = benchmark_fixture(tmp_path / "run", fingerprint="same")
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    if mutation == "total":
        summary["arms"]["student"]["total"] = 32
    elif mutation == "arm":
        summary["arms"]["foreign"] = summary["arms"].pop("student")
    else:
        summary["arms"]["student"]["provisional_passes"] = 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Summary"):
        build_loop_report(tmp_path, benchmark_dirs=[directory])


def test_valid_incomplete_manifest_remains_running_without_summary(tmp_path):
    from bufogen.workflows.loop_report import build_loop_report

    directory = benchmark_fixture(tmp_path / "run", fingerprint="same")
    (directory / "summary.json").unlink()
    settings = json.loads((directory / "benchmark.json").read_text())
    settings["configuration"]["prompts"].append(
        {**settings["configuration"]["prompts"][0], "id": "second"}
    )
    (directory / "benchmark.json").write_text(json.dumps(settings))
    run = build_loop_report(tmp_path, benchmark_dirs=[directory])["benchmarks"][0]
    assert run["status"] == "running" and run["generated"] == 1 and run["expected_generations"] == 2
    assert run["arms"] == {}


def test_report_optional_live_receipts_preserve_selected_rl_and_link_actual_urls(tmp_path):
    from bufogen.workflows.loop_report import build_loop_report, write_loop_report

    baseline = {
        "run_url": "https://wandb.ai/team/project/runs/actual-baseline",
        "scope": "100 heldout prompts per arm",
        "images": 300,
    }
    additional = {"run_url": "https://wandb.ai/team/project/runs/actual-additional"}
    lens = {"scope": "all project conversations", "conversations": 404, "aria_verified": False}
    failed = {"status": "failed", "actual_completed_updates": 0, "new_rl_checkpoint": None}
    for name, value in [
        ("heldout-baseline-visualization.json", baseline),
        ("heldout-visualization.json", additional),
        ("agent-lens-status.json", lens),
        ("rl-recovery-failure.json", failed),
    ]:
        (tmp_path / name).write_text(json.dumps(value))
    selected = tmp_path / "selected-rl.json"
    selected.write_text(json.dumps({"status": "registered", "run_id": "selected"}))
    report = build_loop_report(tmp_path, rl_state=selected)
    phases = report["phases"]
    assert phases["Observe"]["agent_lens_status"] == lens
    assert phases["Observe"]["status"] == "not_started"
    assert phases["Improve"]["rl_recovery_failure"] == failed
    assert phases["Improve"]["rl"] == {"status": "registered", "run_id": "selected"}
    assert phases["Evaluate"]["heldout_baseline_visualization"] == baseline
    assert phases["Evaluate"]["heldout_visualization"] == additional
    assert (
        phases["Evaluate"]["status"] == "not_started"
    )  # Receipts cannot fabricate benchmark completion.
    page = write_loop_report(tmp_path, tmp_path / "report", rl_state=selected).read_text()
    assert f'href="{baseline["run_url"]}"' in page
    assert f'href="{additional["run_url"]}"' in page


def test_report_missing_optional_live_receipts_stay_null(tmp_path):
    from bufogen.workflows.loop_report import build_loop_report

    phases = build_loop_report(tmp_path)["phases"]
    assert phases["Observe"]["agent_lens_status"] is None
    assert phases["Improve"]["rl_recovery_failure"] is None
    assert phases["Improve"]["rl"] == {"status": "not_started"}
    assert phases["Evaluate"]["heldout_baseline_visualization"] is None
    assert phases["Evaluate"]["heldout_visualization"] is None
    assert phases["Evaluate"]["status"] == "not_started"


async def test_benchmark_failure_row_keeps_calls_recorded_before_the_error(tmp_path, monkeypatch):
    import bufogen.evaluation.benchmark as benchmark
    import bufogen.integrations.observability as observability
    import bufogen.integrations.telemetry as telemetry
    from bufogen.rendering.renderer import RenderResult

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(observability, "record_generation", lambda row: None)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) > 1:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "object": "chat.completion",
                "created": 0,
                "model": "resolved-gemma",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "cut off"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 6000, "total_tokens": 6001},
            },
        )

    original = httpx.AsyncClient

    class MockClient(original):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    class Renderer:
        async def render(self, svg):
            return RenderResult(b"png", True, None)

        async def close(self):
            pass

    monkeypatch.setattr(benchmark, "Renderer", Renderer)
    prompt = dict(
        id="p0", family="f0", split="train", prompt="Draw Bufo", requirements=["Hand touches prop"]
    )
    rows = await benchmark.run_benchmark(
        [prompt], [load_backends()["forge_proxy"]], tmp_path, concurrency=1
    )
    (row,) = rows
    assert row.benchmark_metadata["status"] == "generation_error"
    assert row.truncated_calls >= 1
    assert any(call["finish_reason"] == "length" for call in row.model_calls)


async def test_build_references_dispatch(tmp_path, monkeypatch, capsys):
    import bufogen.evaluation.references as references

    calls = []

    def build(run_dir, out_dir, *, backend):
        calls.append((run_dir, out_dir, backend))
        return {}

    monkeypatch.setattr(references, "build_reference_manifest", build)
    out = tmp_path / "refs"
    await dispatch(arguments("build-references", "--output", tmp_path, "--references-out", out))
    assert calls == [(tmp_path, out, "closed_quality_bar")]
    assert str(out / "references.json") in capsys.readouterr().out
    await dispatch(
        arguments(
            "build-references",
            "--output",
            tmp_path,
            "--references-out",
            out,
            "--reference-backend",
            "open_base",
        )
    )
    assert calls[-1][2] == "open_base"
    with pytest.raises(ValueError, match="references-out"):
        await dispatch(arguments("build-references", "--output", tmp_path))


async def test_benchmark_score_astra_forwards_flags(tmp_path, monkeypatch):
    import bufogen.evaluation.benchmark as benchmark

    calls = []

    async def score(output, **kwargs):
        calls.append((output, kwargs))

    monkeypatch.setattr(benchmark, "score_benchmark", score)
    ids = tmp_path / "ids.txt"
    ids.write_text(" a \n\nb\n   \n")
    await dispatch(
        arguments(
            "benchmark-score",
            "--output",
            tmp_path,
            "--judge",
            "astra",
            "--references",
            tmp_path / "refs",
            "--self-reference",
            "--baseline-arm",
            "open_base",
            "--prompt-ids-file",
            ids,
            "--confirmation",
        )
    )
    output, kwargs = calls[0]
    assert output == tmp_path
    assert kwargs["judge"] == "astra"
    assert kwargs["references"] == tmp_path / "refs"
    assert kwargs["self_reference"] is True
    assert kwargs["baseline_arm"] == "open_base"
    assert kwargs["limit_prompt_ids"] == ["a", "b"]
    assert kwargs["confirmation"] is True


async def test_benchmark_score_defaults_unchanged(tmp_path, monkeypatch):
    import bufogen.evaluation.benchmark as benchmark

    calls = []

    async def score(output, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(benchmark, "score_benchmark", score)
    await dispatch(arguments("benchmark-score", "--output", tmp_path))
    kwargs = calls[0]
    assert kwargs["judge"] == "scene"
    assert kwargs["references"] is None and kwargs["self_reference"] is False
    assert kwargs["baseline_arm"] is None and kwargs["limit_prompt_ids"] is None
    assert kwargs["confirmation"] is False


def test_checkpoint_of_another_base_model_uses_its_own_entry():
    from bufogen.cli import _checkpoint_template

    configs = load_backends()
    assert _checkpoint_template(configs, "sft_j1").name == "open_base"
    with pytest.raises(ValueError, match="not_in_yaml.*models.yaml"):
        _checkpoint_template(configs, "not_in_yaml")
    assert _checkpoint_template(configs, "sft_q3").base_model == "Qwen/Qwen3.8-27B"
    assert _checkpoint_template(configs, "sft_n3").base_model.startswith("nvidia/")
    assert _checkpoint_template(configs, "rl_q3").base_model == "Qwen/Qwen3.8-27B"


@pytest.mark.parametrize(
    ("extra", "base_model", "max_tokens", "disable_thinking"),
    [
        ([], "google/gemma-4-26B-A4B-it", 6000, False),
        (
            ["--base-model", "Qwen/Qwen3.8-27B", "--max-tokens", 20000, "--disable-thinking"],
            "Qwen/Qwen3.8-27B",
            20000,
            True,
        ),
    ],
)
async def test_rl_cli_forwards_base_model_and_agent_configuration(
    tmp_path, monkeypatch, extra, base_model, max_tokens, disable_thinking
):
    from bufogen.training import rl as train_rl

    seen = {}

    class FakeTrainer:
        def __init__(self, dataset_ref, **kwargs):
            seen.update(kwargs)

        async def train(self, *, steps):
            return "ckpt"

    monkeypatch.setattr(train_rl, "RLTrainer", FakeTrainer)
    await dispatch(
        arguments(
            "train-rl",
            "--dataset-ref",
            "e/p/d:v0",
            "--name",
            "run",
            "--calibration",
            tmp_path / "c.json",
            *extra,
        )
    )
    assert seen["base_model"] == base_model
    assert seen["max_tokens"] == max_tokens
    assert seen["disable_thinking"] is disable_thinking


def test_rl_arms_resolve_to_their_own_base_in_checkpoint_template():
    from bufogen.cli import _checkpoint_template

    configs = load_backends()
    assert configs["rl_q3"].tuning_method == "rl"
    assert _checkpoint_template(configs, "rl_q3").base_model == "Qwen/Qwen3.8-27B"
    assert _checkpoint_template(configs, "rl_n3").base_model.startswith("nvidia/")


async def test_rl_cli_block_flags_default_and_forward(tmp_path, monkeypatch):
    from bufogen.training import rl as train_rl

    seen = []

    class FakeTrainer:
        def __init__(self, dataset_ref, **kwargs):
            seen.append(kwargs)

        async def train(self, *, steps):
            return "ckpt"

    monkeypatch.setattr(train_rl, "RLTrainer", FakeTrainer)
    base = ["train-rl", "--dataset-ref", "e/p/d:v0", "--name", "run", "--calibration", tmp_path]
    await dispatch(arguments(*base))
    await dispatch(
        arguments(*base, "--tasks-per-step", "2", "--rollouts", "3", "--output-dir", "runs/block2")
    )
    assert (seen[0]["tasks_per_step"], seen[0]["rollouts"], seen[0]["output_dir"]) == (1, 4, None)
    assert (seen[1]["tasks_per_step"], seen[1]["rollouts"]) == (2, 3)
    assert seen[1]["output_dir"] == Path("runs/block2")


async def test_rl_cli_expect_start_step_has_no_default(tmp_path, monkeypatch):
    from bufogen.training import rl as train_rl

    seen = []

    class FakeTrainer:
        def __init__(self, dataset_ref, **kwargs):
            seen.append(kwargs)

        async def train(self, *, steps):
            return "ckpt"

    monkeypatch.setattr(train_rl, "RLTrainer", FakeTrainer)
    base = ["train-rl", "--dataset-ref", "e/p/d:v0", "--name", "run", "--calibration", tmp_path]
    await dispatch(arguments(*base))
    await dispatch(arguments(*base, "--expect-start-step", "6"))
    assert seen[0]["expect_start_step"] is None and seen[1]["expect_start_step"] == 6


@pytest.mark.parametrize("stage", ["run", "demo"])
async def test_cli_forwards_explicit_token_override(stage, tmp_path, monkeypatch):
    import bufogen.cli as cli

    calls = []

    async def pipeline(prompts, backends, output, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(cli, "run_pipeline", pipeline)
    await dispatch(
        arguments(stage, "--output", tmp_path, "--theme", "Bufo", "--max-tokens", "12345")
    )
    assert calls[0]["max_tokens"] == 12345
