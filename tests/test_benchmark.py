import asyncio
import json
from dataclasses import replace

import pytest
import yaml

from bufogen.evaluation.benchmark import load_benchmark, run_benchmark, score_benchmark
from bufogen.generation import BufoGeneration
from bufogen.models import BackendConfig


def prompts():
    return [
        dict(
            id=f"p{i}",
            family=f"family{i}",
            split="train",
            prompt=f"Draw scene {i}",
            requirements=["Hand touches prop"],
        )
        for i in range(3)
    ]


def backend(name="student"):
    return BackendConfig(
        name,
        name,
        "test-model",
        "TEST_MODEL",
        "coreweave",
        "https://example.invalid/v1",
        "TEST_API_KEY",
    )


def test_family_cannot_cross_splits(tmp_path):
    rows = prompts()
    rows[1].update(family=rows[0]["family"], split="heldout")
    path = tmp_path / "leaking.yaml"
    path.write_text(yaml.safe_dump(rows))
    with pytest.raises(ValueError, match="family"):
        load_benchmark(path)


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump([prompts()[0]] * 2))
    with pytest.raises(ValueError, match="id"):
        load_benchmark(path)


def test_partition_manifest_has_full_heldout_and_development():
    from bufogen.settings import ROOT

    rows = load_benchmark(ROOT / "config/forge_prompts.yaml")
    assert len([r for r in rows if r["split"] == "heldout"]) == 100
    assert len([r for r in rows if r["split"] == "train"]) >= 12
    assert "concert" in rows[0]["family"]
    assert "f1" in rows[1]["family"]


class FakeRenderer:
    async def close(self):
        pass


class FakeAgent:
    active = 0
    peak = 0
    calls = []

    def __init__(self, config, renderer, output_dir, **kwargs):
        self.backend = config
        self.output_dir = output_dir
        self.metadata = kwargs["benchmark_metadata"]
        assert kwargs["web_search_enabled"] is False
        assert kwargs["max_tokens"] == 6000

    async def generate(self, prompt, **kwargs):
        type(self).active += 1
        type(self).peak = max(self.peak, self.active)
        self.calls.append((kwargs["prompt_id"], self.backend.name))
        try:
            await asyncio.sleep(0.01)
            if kwargs["prompt_id"] == "p1":
                raise RuntimeError("generation failed")
            path = self.output_dir / (kwargs["prompt_id"] + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"pixels preserved")
            return BufoGeneration(
                prompt_id=kwargs["prompt_id"],
                prompt=prompt,
                backend=self.backend.name,
                model=self.backend.model,
                group_id=kwargs["group_id"],
                svg="<svg/>",
                valid=False,
                latency_s=0.1,
                png_path=str(path),
                benchmark_metadata=self.metadata,
            )
        finally:
            type(self).active -= 1


@pytest.fixture
def fake_generation(monkeypatch):
    import bufogen.evaluation.benchmark as benchmark

    FakeAgent.calls = []
    FakeAgent.active = FakeAgent.peak = 0
    monkeypatch.setattr(benchmark, "BufoAgent", FakeAgent)
    monkeypatch.setattr(benchmark, "Renderer", FakeRenderer)


async def test_paired_bounded_durable_failure_and_matching_resume(tmp_path, fake_generation):
    rows = await run_benchmark(prompts(), [backend(), backend("teacher")], tmp_path, concurrency=2)
    assert len(rows) == 6
    assert FakeAgent.peak == 2
    assert sum(r.benchmark_metadata["status"] == "generation_error" for r in rows) == 2
    assert all(r.png_path for r in rows if r.prompt_id != "p1")
    assert len((tmp_path / "manifest.jsonl").read_text().splitlines()) == 6
    await run_benchmark(prompts(), [backend(), backend("teacher")], tmp_path, concurrency=2)
    assert len(FakeAgent.calls) == 6
    with pytest.raises(ValueError, match="configuration"):
        await run_benchmark(prompts(), [replace(backend(), model="other")], tmp_path)


async def test_concurrent_run_rejected(tmp_path, fake_generation):
    first = asyncio.create_task(run_benchmark(prompts(), [backend()], tmp_path))
    await asyncio.sleep(0.001)
    with pytest.raises(RuntimeError, match="locked"):
        await run_benchmark(prompts(), [backend()], tmp_path)
    await first


async def test_scoring_retains_errors_and_denominators(tmp_path, fake_generation, monkeypatch):
    from bufogen.evaluation.scorers.scene_quality import SceneQuality

    await run_benchmark(prompts(), [backend()], tmp_path)

    async def fake_score(self, output, rescore=False):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(SceneQuality, "score", fake_score)
    rows = await score_benchmark(tmp_path, publish=False)
    assert len(rows) == 3
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["arms"]["student"]["total"] == 3
    assert summary["arms"]["student"]["judge_errors"] == 3
    assert summary["arms"]["student"]["scored"] == 0
    assert (tmp_path / "comparison.html").exists()


async def test_interrupted_started_item_is_not_reissued(tmp_path, fake_generation):
    import bufogen.evaluation.benchmark as benchmark

    rows = await run_benchmark(prompts(), [backend()], tmp_path)
    # Model a crash after the start checkpoint but before one result was appended.
    kept = [r.to_dict() for r in rows if r.prompt_id != "p0"]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    resumed = await benchmark.run_benchmark(prompts(), [backend()], tmp_path)
    assert len(FakeAgent.calls) == 3
    assert len(resumed) == 3
    interrupted = next(r for r in resumed if r.prompt_id == "p0")
    assert interrupted.benchmark_metadata["status"] == "generation_error"
    assert "Interrupted" in interrupted.error


async def test_modified_prompt_requirements_reject_resume(tmp_path, fake_generation):
    await run_benchmark(prompts(), [backend()], tmp_path)
    changed = prompts()
    changed[0]["requirements"] = ["Different contact"]
    with pytest.raises(ValueError, match="configuration"):
        await run_benchmark(changed, [backend()], tmp_path)


async def test_generation_keeps_pixels_when_later_model_request_fails(tmp_path):
    from bufogen.agents.agent import BufoAgent
    from bufogen.rendering.renderer import RenderResult

    class Renderer:
        async def render(self, svg):
            return RenderResult(b"PNG", True, None)

    async def fail_after_render(prompt, group_id, render):
        await render('<svg><circle r="10"/></svg>')
        raise RuntimeError("provider failed after drawing")

    agent = BufoAgent(
        backend(),
        Renderer(),
        tmp_path,
        run=fail_after_render,
        benchmark_metadata={"requirements": ["circle"]},
        web_search_enabled=False,
    )
    with pytest.raises(RuntimeError):
        await agent.generate("Draw circle", prompt_id="p0")
    assert agent.partial_generation.png_path
    assert (tmp_path / "partial.json").exists()
    assert agent.partial_generation.valid


async def test_changed_saved_score_requirements_rejected(tmp_path, fake_generation, monkeypatch):
    from bufogen.evaluation.scorers.scene_quality import SceneQuality, failure_result

    await run_benchmark(prompts(), [backend()], tmp_path)

    async def fake_score(self, output, rescore=False):
        return failure_result("judge_error") | {
            "scorer_configuration": self.configuration(),
            "requirements": output["benchmark_metadata"]["requirements"],
            "generation_id": output["id"],
        }

    monkeypatch.setattr(SceneQuality, "score", fake_score)
    await score_benchmark(tmp_path, publish=False)
    path = tmp_path / "scored.jsonl"
    rows = [json.loads(r) for r in path.read_text().splitlines()]
    rows[0]["benchmark_metadata"]["requirements"] = ["tampered"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="scored"):
        await score_benchmark(tmp_path, publish=False)


async def test_corrupt_partial_checkpoint_recovers_as_failure(tmp_path, fake_generation):
    from bufogen.evaluation.benchmark import digest

    rows = await run_benchmark(prompts(), [backend()], tmp_path)
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(r.to_dict()) + "\n" for r in rows if r.prompt_id != "p0")
    )
    item = tmp_path / "images" / digest(("p0", "student"))[:24]
    (item / "partial.json").write_text("{broken")
    resumed = await run_benchmark(prompts(), [backend()], tmp_path)
    assert len(resumed) == 3
    assert (item / "partial.damaged.json").exists()
    assert len(FakeAgent.calls) == 3


def test_native_eval_missing_score_rejected():
    from bufogen.evaluation.benchmark import _validate_eval_coverage

    row = BufoGeneration(
        prompt_id="p",
        prompt="p",
        backend="b",
        model="m",
        group_id="g",
        svg="",
        valid=False,
        latency_s=0,
    )
    row.scores["scene_quality"] = {"status": "judge_error", "generation_id": row.id}
    with pytest.raises(RuntimeError, match="omitted"):
        _validate_eval_coverage([{"output": None, "scores": {"SceneQuality": {}}}], [row])
    _validate_eval_coverage([{"scores": {"SceneQuality": row.scores["scene_quality"]}}], [row])


def test_comparison_fingerprint_excludes_arms_and_parallelism(tmp_path, fake_generation):
    from bufogen.evaluation.benchmark import _configuration, comparison_fingerprint

    a = _configuration(
        prompts(),
        [backend()],
        temperature=0.7,
        max_tokens=6000,
        max_turns=8,
        concurrency=2,
        disable_thinking=False,
    )
    b = _configuration(
        prompts(),
        [replace(backend(), model="tuned")],
        temperature=0.7,
        max_tokens=6000,
        max_turns=8,
        concurrency=4,
        disable_thinking=False,
    )
    assert comparison_fingerprint(a) == comparison_fingerprint(b)
    b["policy"]["max_turns_per_attempt"] = 9
    assert comparison_fingerprint(a) != comparison_fingerprint(b)


def test_configuration_records_disable_thinking_and_changes_fingerprint():
    from bufogen.evaluation.benchmark import _configuration, comparison_fingerprint

    kw = dict(temperature=0.7, max_tokens=6000, max_turns=100, concurrency=2)
    a = _configuration(prompts(), [backend()], disable_thinking=False, **kw)
    b = _configuration(prompts(), [backend()], disable_thinking=True, **kw)
    assert a["policy"]["disable_thinking"] is False and b["policy"]["disable_thinking"] is True
    assert comparison_fingerprint(a) != comparison_fingerprint(b)


def _aug_rows():
    base = dict(
        id="gap-train-a-base",
        family="a",
        split="train",
        prompt="Draw a",
        requirements=["Hand touches prop"],
    )
    aug = dict(
        id="aug-train-a-01",
        family="a",
        split="train",
        prompt="Draw a, red",
        requirements=["Hand touches red prop"],
        provenance="stage-b-augment-2026-10",
        augmented_from="gap-train-a-base",
    )
    val = dict(
        id="gap-validation-v-base",
        family="v",
        split="validation",
        prompt="Draw v",
        requirements=["Hand touches prop"],
    )
    return base, aug, val


def test_augmented_prompt_must_be_in_train_family(tmp_path):
    base, aug, val = _aug_rows()
    aug.update(family="v", augmented_from="gap-train-a-base")
    path = tmp_path / "aug.yaml"
    path.write_text(yaml.safe_dump([base, aug, val]))
    with pytest.raises(ValueError, match="family"):
        load_benchmark(path)
    # A train prompt placed in a heldout/validation-only family is rejected too.
    base, aug, val = _aug_rows()
    val.update(id="gap-train-v-x", split="train")
    path.write_text(
        yaml.safe_dump([base, val, dict(base, id="b2", family="v", split="validation")])
    )
    with pytest.raises(ValueError, match="family"):
        load_benchmark(path)
    base, aug, val = _aug_rows()
    path.write_text(yaml.safe_dump([base, aug, val]))
    assert len(load_benchmark(path)) == 3


def test_augmented_prompts_carry_provenance(tmp_path):
    path = tmp_path / "aug.yaml"
    for mutate, match in (
        (lambda a: a.pop("provenance"), "provenance"),
        (lambda a: a.pop("augmented_from"), "augmented_from"),
        (lambda a: a.update(augmented_from="missing"), "augmented_from"),
        (lambda a: a.update(augmented_from="gap-validation-v-base"), "augmented_from"),
    ):
        base, aug, val = _aug_rows()
        mutate(aug)
        path.write_text(yaml.safe_dump([base, aug, val]))
        with pytest.raises(ValueError, match=match):
            load_benchmark(path)


def test_augmented_provenance_cannot_be_bypassed_by_id(tmp_path):
    path = tmp_path / "aug.yaml"
    cases = (
        # provenance/augmented_from on a row whose id does not look augmented
        (lambda a: a.update(id="gap-train-a-sneaky"), "aug-train"),
        # augmented-looking id whose family segment differs from the row's family
        (lambda a: a.update(id="aug-train-b-01"), "aug-train"),
        (lambda a: a.update(id="aug-train-a-1"), "aug-train"),
        (lambda a: a.update(provenance="something-else"), "provenance"),
        (lambda a: a.update(provenance="stage-b-augment-2026-10 "), "provenance"),
    )
    for mutate, match in cases:
        base, aug, val = _aug_rows()
        mutate(aug)
        path.write_text(yaml.safe_dump([base, aug, val]))
        with pytest.raises(ValueError, match=match):
            load_benchmark(path)
    # only one of the two markers is still an augmented row
    for key in ("provenance", "augmented_from"):
        base, aug, val = _aug_rows()
        aug.pop(key)
        aug.update(id="gap-train-a-sneaky")
        path.write_text(yaml.safe_dump([base, aug, val]))
        with pytest.raises(ValueError, match="augmented"):
            load_benchmark(path)
    # an augmented base cannot be a non-original row either
    base, aug, val = _aug_rows()
    base.update(provenance="stage-b-augment-2026-10")
    path.write_text(yaml.safe_dump([base, aug, val]))
    with pytest.raises(ValueError):
        load_benchmark(path)


async def test_run_benchmark_prompt_ids_filter(tmp_path, fake_generation):
    rows = await run_benchmark(prompts(), [backend()], tmp_path, prompt_ids=["p2", "p0"])
    assert sorted(r.prompt_id for r in rows) == ["p0", "p2"]
    for bad in (["p0", "p0"], ["nope"], []):
        with pytest.raises(ValueError, match="prompt_ids"):
            await run_benchmark(prompts(), [backend()], tmp_path / "bad", prompt_ids=bad)
