import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import yaml
from openai import AsyncOpenAI
from test_rl_recovery import BASE, completion

from bufogen.generation import BufoGeneration
from bufogen.training import rl as train_rl
from bufogen.training.rl_contract import write_environment_tasks

CONFIG = {"rubric_version": "bufo-astra-v5", "samples": 3, "threshold": 0.65}
REQUIREMENTS = ["Hand grips pointer"]


def make_reference(tmp_path, content=b"png-bytes"):
    path = tmp_path / "ref.png"
    path.write_bytes(content)
    return {"png_path": str(path), "png_sha256": hashlib.sha256(content).hexdigest()}


def write_tasks(tmp_path, reference):
    row = {
        "id": "p",
        "prompt": "Bufo grips a pointer",
        "family": "presenter",
        "split": "train",
        "requirements": REQUIREMENTS,
    }
    if reference is not None:
        row["reference"] = reference
    write_environment_tasks([row], tmp_path)


def astra_row(reference):
    row = BufoGeneration(
        prompt_id="p",
        prompt="Bufo grips a pointer",
        backend="open_base",
        model="student",
        group_id="g",
        svg="<svg/>",
        valid=True,
        latency_s=1,
        benchmark_metadata={
            "status": "completed",
            "requirements": REQUIREMENTS,
            "reference": reference,
        },
    )
    row.scores["astra_judge"] = {
        "status": "ok",
        "generation_id": row.id,
        "bufo_present": True,
        "similarity": 0.8,
        "requirement_coverage": 1.0,
        "anatomy": 0.9,
        "composition": 0.9,
        "readability": 0.9,
        "scorer_configuration": CONFIG,
    }
    return row


def build(tmp_path, monkeypatch, reference, *, runtime=None):
    import art.serverless

    import wandb
    from bufogen.evaluation.scorers import astra_judge
    from bufogen.integrations import telemetry

    write_tasks(tmp_path, reference)
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    model = SimpleNamespace(
        register=AsyncMock(),
        get_step=AsyncMock(return_value=0),
        log=AsyncMock(),
        run_id="run",
        id="model",
    )
    result = SimpleNamespace(step=1, artifact_name="ckpt:v1", metrics={})
    backend = SimpleNamespace(
        _client=SimpleNamespace(training_jobs=SimpleNamespace(create=AsyncMock())),
        close=AsyncMock(),
        train=AsyncMock(return_value=result),
    )
    runtime_mock = runtime or Mock(return_value={})
    monkeypatch.setattr(train_rl, "require_rl_runtime", runtime_mock)
    monkeypatch.setattr(train_rl, "require_rl_logging_dependencies", lambda: None)
    monkeypatch.setattr(train_rl, "require_astra_calibration", lambda report, **kw: {"ok": True})
    monkeypatch.setattr(train_rl, "resolve_rl_checkpoint", lambda *a: {"checkpoint_artifact": "c"})
    monkeypatch.setattr(art, "TrainableModel", lambda **kwargs: model)
    monkeypatch.setattr(art.serverless, "ServerlessBackend", lambda **kwargs: backend)
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(train_rl, "ROOT", tmp_path)
    monkeypatch.setattr(train_rl, "required", lambda key: "test")
    monkeypatch.setattr(train_rl, "Renderer", lambda: SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(wandb, "Api", Mock())
    state = {
        "in_capture": False,
        "judged": [],
        "factory_kwargs": [],
        "valid": True,
        "error": None,
        "similarity": None,  # callable(index) -> similarity; default varies per rollout
    }
    real_rollout = train_rl.student_rollout

    async def tracked_rollout(**kwargs):
        state["in_capture"] = True
        try:
            return await real_rollout(**kwargs)
        finally:
            state["in_capture"] = False

    monkeypatch.setattr(train_rl, "student_rollout", tracked_rollout)

    async def score(row, *, rescore):
        assert not state["in_capture"], "judge must run outside the ART capture"
        state["judged"].append(row["id"])
        index = len(state["judged"]) - 1
        similarity = state["similarity"](index) if state["similarity"] else 0.8 - 0.1 * (index % 2)
        return astra_row(reference).scores["astra_judge"] | {
            "generation_id": row["id"],
            "similarity": similarity,
        }

    monkeypatch.setattr(
        astra_judge,
        "AstraReferenceJudge",
        lambda **kw: SimpleNamespace(configuration=lambda: CONFIG, score=score, kw=kw),
    )
    client = AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=completion(1)))
        ),
    )
    model.openai_client = Mock(return_value=SimpleNamespace(with_options=Mock(return_value=client)))

    def factory(*args, **kwargs):
        state["factory_kwargs"].append(kwargs)

        async def generate(prompt, **options):
            await client.chat.completions.create(
                model=args[0].model, messages=[{"role": "user", "content": prompt}]
            )
            row = astra_row(reference)
            # As the real agent returns it: the metadata it was given, with no status.
            row.benchmark_metadata = kwargs["benchmark_metadata"]
            row.valid, row.error = state["valid"], state["error"]
            row.scores = {}
            return row

        return SimpleNamespace(generate=generate)

    monkeypatch.setattr(train_rl, "BufoAgent", factory)
    trainer = train_rl.RLTrainer(
        "dataset:v0",
        name="attempt",
        base_model=BASE,
        calibration_path=calibration,
        reward_kind="astra",
    )
    trainer.registry = Mock()
    trainer.registry.download.return_value = tmp_path
    return trainer, model, backend, runtime_mock, state, client


async def test_astra_rollouts_score_outside_capture_and_record_serving_path(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    async with client:
        assert await trainer.train(steps=1) == "c"
    assert len(state["judged"]) == 4 and backend.train.await_count == 1
    groups = backend.train.await_args.args[1]
    # Rollouts finish concurrently, so the mock judge's call order is not the group order.
    assert sorted(t.reward for t in groups[0].trajectories) == pytest.approx(
        sorted([0.8 * 0.9, 0.7 * 0.9] * 2)
    )
    for kwargs in state["factory_kwargs"]:
        assert kwargs["benchmark_metadata"]["reference"] == reference
        assert kwargs["benchmark_metadata"]["serving_path"] == "art"
        assert kwargs["max_turns"] == 100 and kwargs["max_tokens"] == 6000
    out = tmp_path / "runs/attempt"
    assert json.loads((out / "rl-training.json").read_text())["serving_path"] == "art"
    captures = [json.loads(p.read_text()) for p in sorted((out / "captures").glob("*.json"))]
    assert len(captures) == 4
    assert all(c["status"] == "scored" and c["serving_path"] == "art" for c in captures)
    assert all(c["training_tokens"] for c in captures)


async def test_astra_judge_is_built_with_the_v5_rubric_and_three_samples(tmp_path, monkeypatch):
    from bufogen.evaluation.scorers import astra_judge

    reference = make_reference(tmp_path)
    trainer, *_, client = build(tmp_path, monkeypatch, reference)
    built = []
    factory = astra_judge.AstraReferenceJudge
    monkeypatch.setattr(
        astra_judge, "AstraReferenceJudge", lambda **kw: built.append(kw) or factory(**kw)
    )
    async with client:
        await trainer.train(steps=1)
    assert built == [{"rubric_version": "bufo-astra-v5", "samples": 3}]


@pytest.mark.parametrize("fault", ["missing_file", "sha_mismatch", "no_reference", "bad_shape"])
async def test_reference_guard_fails_before_any_art_call(tmp_path, monkeypatch, fault):
    reference = make_reference(tmp_path)
    if fault == "missing_file":
        Path(reference["png_path"]).unlink()
    elif fault == "sha_mismatch":
        Path(reference["png_path"]).write_bytes(b"changed")
    runtime = Mock(side_effect=AssertionError("ART runtime must not be reached"))
    supplied = {
        "no_reference": None,
        "bad_shape": {"png_path": reference["png_path"]},
    }.get(fault, reference)
    trainer, model, backend, _, state, client = build(
        tmp_path, monkeypatch, supplied, runtime=runtime
    )
    async with client:
        with pytest.raises(ValueError, match="[Rr]eference"):
            await trainer.train(steps=1)
    runtime.assert_not_called()
    model.register.assert_not_awaited()
    assert not (tmp_path / "runs/attempt").exists()


async def test_capture_validation_is_unchanged_for_astra(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    monkeypatch.setattr(
        train_rl,
        "validate_training_tokens",
        Mock(side_effect=ValueError("Actual training tokens are missing")),
    )
    async with client:
        with pytest.raises(ValueError, match="training tokens"):
            await trainer.train(steps=1)
    assert state["judged"] == []
    backend.train.assert_not_awaited()


def load_build_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_rl_tasks.py"
    spec = importlib.util.spec_from_file_location("build_rl_tasks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_rl_tasks_selects_train_split_with_references(tmp_path):
    script = load_build_script()
    refs = tmp_path / "refs"
    (refs / "images").mkdir(parents=True)
    rows = {}
    prompts = []
    for pid, split in [("t1", "train"), ("t2", "train"), ("v1", "validation")]:
        (refs / "images" / f"{pid}.png").write_bytes(pid.encode())
        rows[pid] = {
            "png_path": f"images/{pid}.png",
            "png_sha256": hashlib.sha256(pid.encode()).hexdigest(),
            "generation_id": f"g{pid}",
        }
        prompts.append(
            {
                "id": pid,
                "family": f"fam-{split}-{pid}",
                "split": split,
                "prompt": f"prompt {pid}",
                "requirements": ["a thing"],
            }
        )
    (refs / "references.json").write_text(
        json.dumps({"version": "bufo-references-v1", "rows": rows})
    )
    prompts_path = tmp_path / "prompts.yaml"
    prompts_path.write_text(yaml.safe_dump(prompts))
    output = tmp_path / "out" / "tasks-train.jsonl"
    metadata = script.build(prompts_path, refs / "references.json", output)
    tasks = [json.loads(line) for line in output.read_text().splitlines()]
    assert [t["prompt_id"] for t in tasks] == ["t1", "t2"] and metadata["count"] == 2
    for task in tasks:
        reference = task["reference"]
        assert Path(reference["png_path"]).is_file()
        assert reference["png_sha256"] == hashlib.sha256(task["prompt_id"].encode()).hexdigest()
    assert json.loads(output.with_suffix(".metadata.json").read_text())["count"] == 2
    train_rl.validate_task_references(tasks)


def test_build_rl_tasks_rejects_train_prompt_without_reference(tmp_path):
    script = load_build_script()
    prompts = [{"id": "t1", "family": "f", "split": "train", "prompt": "p", "requirements": ["x"]}]
    manifest = {"rows": {}, "base_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="No reference"):
        script.select_rows(prompts, manifest)


async def test_failed_generation_rewards_zero_and_is_not_completed(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    state["valid"], state["error"] = False, "render failed"
    async with client:
        with pytest.raises(RuntimeError, match="learning signal"):
            await trainer.train(steps=1)
    record = json.loads((tmp_path / "runs/attempt/rl-training.json").read_text())
    assert record["status"] == "no_learning_signal" and record["step_counts"]
    manifest = [
        json.loads(line)
        for line in (tmp_path / "runs/attempt/manifest.jsonl").read_text().splitlines()
    ]
    assert {r["benchmark_metadata"]["status"] for r in manifest} == {"generation_error"}
    counts = json.loads((tmp_path / "runs/attempt/rl-training.json").read_text())["step_counts"]
    assert counts[0]["completed"] == 0 and counts[0]["trained"] == 0
    backend.train.assert_not_awaited()


async def test_flat_batch_skips_update_and_records_counts(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    state["similarity"] = lambda index: 0.5 if index < 4 else 0.8 - 0.1 * (index % 2)
    async with client:
        assert await trainer.train(steps=2) == "c"
    assert backend.train.await_count == 1
    record = json.loads((tmp_path / "runs/attempt/rl-training.json").read_text())
    assert record["actual_completed_updates"] == 1
    names = sorted(p.name for p in (tmp_path / "runs/attempt/captures").glob("*.json"))
    assert "step-0-rollout-0-attempt-1.json" in names
    assert "step-0-skip1-rollout-0-attempt-1.json" in names
    counts = record["step_counts"]
    assert [c["learning_signal"] for c in counts] == [False, True]
    assert [c["trained"] for c in counts] == [0, 4]
    for c in counts:
        assert (c["attempted"], c["completed"], c["token_valid"], c["scored"]) == (4, 4, 4, 4)


async def test_trained_count_is_zero_if_the_update_raises(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    backend.train.side_effect = ValueError("boom")
    async with client:
        with pytest.raises(ValueError, match="boom"):
            await trainer.train(steps=1)
    record = json.loads((tmp_path / "runs/attempt/rl-training.json").read_text())
    assert record["status"] == "failed" and record["step_counts"][0]["trained"] == 0
    assert record["step_counts"][0]["learning_signal"] is True


@pytest.mark.parametrize(
    ("base_model", "base_arm"),
    [
        ("Qwen/Qwen3.8-27B", "qwen_base"),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B", "nemotron_base"),
        (BASE, "open_base"),
    ],
)
async def test_every_allowlisted_base_trains_with_its_own_arm_and_tokenizer(
    tmp_path, monkeypatch, base_model, base_arm
):
    import art

    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    trainer.base_model = base_model
    tokenized = []
    real = train_rl.validate_training_tokens
    monkeypatch.setattr(
        train_rl,
        "validate_training_tokens",
        lambda trajectory, base_model: tokenized.append(base_model) or real(trajectory, base_model),
    )
    real_tokenize = art.Trajectory.tokenize
    seen = []
    monkeypatch.setattr(
        art.Trajectory,
        "tokenize",
        lambda self, **kw: seen.append(kw["base_model"]) or real_tokenize(self, **kw),
    )
    async with client:
        await trainer.train(steps=1)
    assert set(tokenized) == {base_model} and set(seen) == {base_model}
    record = json.loads((tmp_path / "runs/attempt/rl-training.json").read_text())
    assert record["base_model"] == base_model and record["base_arm"] == base_arm
    assert record["arm"] == f"{base_arm}_rl"


async def test_unlisted_base_raises_before_any_network_call(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, runtime_mock, _, client = build(tmp_path, monkeypatch, reference)
    trainer.base_model = "meta/unlisted-70B"
    async with client:
        with pytest.raises(ValueError, match="allowlisted"):
            await trainer.train(steps=1)
    model.register.assert_not_awaited()
    runtime_mock.assert_not_called()
    assert not (tmp_path / "runs/attempt").exists()


async def test_rollout_agent_uses_the_evaluation_configuration(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    trainer.max_tokens, trainer.disable_thinking = 20000, True
    async with client:
        await trainer.train(steps=1)
    for kwargs in state["factory_kwargs"]:
        assert kwargs["max_tokens"] == 20000 and kwargs["disable_thinking"] is True
        assert kwargs["max_turns"] == 100
    record = json.loads((tmp_path / "runs/attempt/rl-training.json").read_text())
    assert record["max_tokens"] == 20000 and record["disable_thinking"] is True


async def test_rollout_agent_defaults_match_the_original_configuration(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    async with client:
        await trainer.train(steps=1)
    assert all(
        k["max_tokens"] == 6000 and k["disable_thinking"] is False for k in state["factory_kwargs"]
    )


async def test_rollout_client_timeout_follows_the_environment(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    monkeypatch.setenv("BUFO_INFERENCE_TIMEOUT_S", "900")
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    async with client:
        await trainer.train(steps=1)
    options = model.openai_client.return_value.with_options.call_args.kwargs
    assert options["timeout"] == 900


async def test_rollout_client_timeout_defaults_to_180(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    monkeypatch.delenv("BUFO_INFERENCE_TIMEOUT_S", raising=False)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    async with client:
        await trainer.train(steps=1)
    assert model.openai_client.return_value.with_options.call_args.kwargs["timeout"] == 180


def test_tasks_are_ordered_round_robin_across_families():
    tasks = [{"prompt_id": f"{f}{i}", "family": f} for f in ("a", "b", "c") for i in range(3)]
    ordered = train_rl.family_round_robin(tasks)
    assert [t["prompt_id"] for t in ordered] == [
        "a0",
        "b0",
        "c0",
        "a1",
        "b1",
        "c1",
        "a2",
        "b2",
        "c2",
    ]
    assert len({t["family"] for t in ordered[:3]}) == 3
    assert train_rl.family_round_robin(tasks) == ordered
    assert train_rl.family_round_robin([]) == []


async def test_run_writes_a_per_step_summary(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    async with client:
        await trainer.train(steps=1)
    summary = json.loads((tmp_path / "runs/attempt/rl-steps.json").read_text())
    (step,) = summary["steps"]
    assert step["task_id"] == "p" and step["family"] == "presenter"
    assert (step["attempted"], step["completed"], step["token_valid"], step["scored"]) == (
        4,
        4,
        4,
        4,
    )
    assert step["trained"] == 4 and step["reward_mean"] == pytest.approx(0.675, abs=0.001)
    assert {"rollout_latency_median_s", "training_job_id"} <= set(step)


def test_sft_arm_label_requires_the_same_base_model():
    from dataclasses import replace

    from bufogen.models import load_backends

    arms = load_backends()
    q3 = arms["sft_q3"].model
    assert train_rl.sft_arm_label(q3, "qwen_base", arms) == "sft_q3"
    with pytest.raises(ValueError, match="Qwen/Qwen3.8-27B.*nvidia|nvidia.*Qwen/Qwen3.8-27B"):
        train_rl.sft_arm_label(q3, "nemotron_base", arms)
    other = dict(arms) | {"sft_q3": replace(arms["sft_q3"], base_model="meta/other")}
    with pytest.raises(ValueError, match="meta/other"):
        train_rl.sft_arm_label(q3, "qwen_base", other)


def test_base_arm_resolution_requires_exactly_one_coreweave_arm(monkeypatch):
    from bufogen.models import load_backends

    arms = load_backends()
    assert train_rl.resolve_base_arm("Qwen/Qwen3.8-27B", arms) == "qwen_base"
    with pytest.raises(ValueError, match="No coreweave base arm"):
        train_rl.resolve_base_arm("meta/unlisted-70B", arms)
    from dataclasses import replace

    dup = dict(arms) | {"qwen_dup": replace(arms["qwen_base"], name="qwen_dup")}
    with pytest.raises(ValueError, match="Several"):
        train_rl.resolve_base_arm("Qwen/Qwen3.8-27B", dup)


def test_sft_arm_label_comes_from_models_yaml_with_fallback():
    from bufogen.models import load_backends

    arms = load_backends()
    q3 = arms["sft_q3"].model
    assert train_rl.sft_arm_label(q3, "qwen_base", arms) == "sft_q3"
    assert train_rl.sft_arm_label("wandb-artifact:///x/y/z:v1", "qwen_base", arms) == (
        "qwen_base_sft_rl"
    )


async def test_one_failed_rollout_does_not_cancel_its_siblings(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    calls = {"n": 0}
    real = train_rl.validate_training_tokens

    def flaky(trajectory, base_model):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("Could not load tokenizer")
        return real(trajectory, base_model)

    monkeypatch.setattr(train_rl, "validate_training_tokens", flaky)
    async with client:
        await trainer.train(steps=1)
    groups = backend.train.await_args.args[1]
    assert [len(g.trajectories) for g in groups] == [3]
    (step,) = json.loads((tmp_path / "runs/attempt/rl-steps.json").read_text())["steps"]
    assert (step["attempted"], step["completed"], step["token_valid"], step["trained"]) == (
        4,
        4,
        3,
        3,
    )
    assert step["failed"] == 1 and step["failure_types"] == ["ValueError"]


async def test_step_aborts_when_every_group_has_fewer_than_two_valid(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    calls = {"n": 0}
    real = train_rl.validate_training_tokens

    def flaky(trajectory, base_model):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("boom")
        return real(trajectory, base_model)

    monkeypatch.setattr(train_rl, "validate_training_tokens", flaky)
    async with client:
        with pytest.raises(ValueError, match="boom"):
            await trainer.train(steps=1)
    backend.train.assert_not_awaited()


def write_two_tasks(tmp_path, reference):
    rows = [
        {
            "id": f"t{i}",
            "prompt": f"Bufo {i}",
            "family": f"fam{i}",
            "split": "train",
            "requirements": REQUIREMENTS,
            "reference": reference,
        }
        for i in range(2)
    ]
    write_environment_tasks(rows, tmp_path)


def wrap_generate(monkeypatch, wrapper):
    """Wrap each fake student agent's generate with wrapper(generate, prompt, options)."""
    factory = train_rl.BufoAgent

    def wrapped_factory(*args, **kwargs):
        agent = factory(*args, **kwargs)
        generate = agent.generate

        async def tracked(prompt, **options):
            return await wrapper(generate, prompt, options)

        return SimpleNamespace(generate=tracked)

    monkeypatch.setattr(train_rl, "BufoAgent", wrapped_factory)


async def test_two_tasks_by_four_rollouts_make_two_groups_run_concurrently(tmp_path, monkeypatch):
    import asyncio

    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    write_two_tasks(tmp_path, reference)
    trainer.tasks_per_step, trainer.rollouts = 2, 4
    trainer.output_dir = Path("runs/block")
    inflight = {"now": 0, "max": 0}
    release = asyncio.Event()

    async def tracked(generate, prompt, options):
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        if inflight["now"] == 8:
            release.set()
        try:
            await asyncio.wait_for(release.wait(), 5)  # times out unless 8 are in flight together
            return await generate(prompt, **options)
        finally:
            inflight["now"] -= 1

    wrap_generate(monkeypatch, tracked)
    async with client:
        await trainer.train(steps=1)
    assert inflight["max"] == 8
    groups = backend.train.await_args.args[1]
    assert [len(g.trajectories) for g in groups] == [4, 4]
    out = tmp_path / "runs/block"
    assert not (tmp_path / "runs/attempt").exists()
    summary = json.loads((out / "rl-steps.json").read_text())
    assert summary["step_convention"].startswith("step = pre-update step")
    (step,) = summary["steps"]
    assert step["task_ids"] == ["t0", "t1"] and step["families"] == ["fam0", "fam1"]
    assert "task_id" not in step and step["groups_trained"] == 2 and step["groups_skipped"] == 0
    assert (step["attempted"], step["scored"], step["trained"]) == (8, 8, 8)
    assert len({p.name for p in (out / "captures").glob("*.json")}) == 8


async def test_group_with_one_valid_rollout_is_skipped_and_counted(tmp_path, monkeypatch):
    reference = make_reference(tmp_path)
    trainer, model, backend, _, state, client = build(tmp_path, monkeypatch, reference)
    write_two_tasks(tmp_path, reference)
    trainer.tasks_per_step, trainer.rollouts = 2, 4
    seen = {"Bufo 1": 0}

    async def flaky(generate, prompt, options):
        if prompt in seen:
            seen[prompt] += 1
            if seen[prompt] > 1:
                raise ValueError("boom")
        return await generate(prompt, **options)

    wrap_generate(monkeypatch, flaky)
    async with client:
        await trainer.train(steps=1)
    assert [len(g.trajectories) for g in backend.train.await_args.args[1]] == [4]
    (step,) = json.loads((tmp_path / "runs/attempt/rl-steps.json").read_text())["steps"]
    assert (step["groups_trained"], step["groups_skipped"], step["failed"]) == (1, 1, 3)
    assert (step["token_valid"], step["trained"]) == (5, 4)


@pytest.mark.parametrize(
    "kwargs", [{"tasks_per_step": 0}, {"rollouts": 1}, {"rollouts": 0}, {"expect_start_step": 1}]
)
def test_rl_trainer_constructor_guards(kwargs):
    with pytest.raises(ValueError):
        train_rl.RLTrainer("d:v0", name="n", base_model=BASE, **kwargs)


def _tool(body):
    return {"role": "tool", "content": json.dumps(body)}


def test_render_diagnosis_separates_infrastructure_from_model_zeros():
    sandbox = {"valid": False, "error_type": "SandboxError", "error": "sandbox is not running"}
    ok = {"valid": True}
    row = lambda *results, **kw: SimpleNamespace(  # noqa: E731
        trajectory={"messages": [_tool(r) for r in results]},
        valid=kw.get("valid", True),
        error=None,
        truncated_calls=kw.get("truncated", 0),
    )
    assert train_rl.render_diagnosis(row(sandbox, ok)) == {
        "infra_tool_errors": 1,
        "infra_unrecovered": False,
    }
    stuck = row(sandbox, sandbox, valid=False)
    diagnosis = train_rl.render_diagnosis(stuck)
    assert diagnosis == {"infra_tool_errors": 2, "infra_unrecovered": True}
    assert train_rl.zero_cause(stuck, diagnosis) == "infra"
    capped = row(valid=False, truncated=1)
    assert train_rl.zero_cause(capped, train_rl.render_diagnosis(capped)) == "cap_truncated"
    assert train_rl.zero_cause(row(ok), train_rl.render_diagnosis(row(ok))) is None
