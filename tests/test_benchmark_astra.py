import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from test_astra_judge import judgment
from test_benchmark import FakeAgent, backend, fake_generation, prompts  # noqa: F401

from bufogen.evaluation.benchmark import _validate_eval_coverage, run_benchmark, score_benchmark
from bufogen.evaluation.references import build_reference_manifest
from bufogen.evaluation.scorers.astra_judge import AstraReferenceJudge
from bufogen.generation import BufoGeneration


class PngAgent(FakeAgent):
    async def generate(self, prompt, **kwargs):
        row = await super().generate(prompt, **kwargs)
        color = "green" if self.backend.name == "student" else "olive"
        Image.new("RGB", (8, 8), color).save(row.png_path)
        row.valid = True
        return row


@pytest.fixture
def astra_generation(monkeypatch, fake_generation):  # noqa: F811
    import bufogen.evaluation.benchmark as benchmark

    monkeypatch.setattr(benchmark, "BufoAgent", PngAgent)

    async def fake_judge(self, output, requirements, reference_png):
        # The student's p2 is weak; everything else passes.
        if output["backend"] == "student" and output["prompt_id"] == "p2":
            return judgment(requirements, readability=0.1, vs_reference="worse")
        return judgment(requirements)

    monkeypatch.setattr(AstraReferenceJudge, "_judge", fake_judge)


async def paired_run(tmp_path):
    out = tmp_path / "run"
    await run_benchmark(prompts(), [backend(), backend("teacher")], out)
    refs = tmp_path / "refs"
    build_reference_manifest(out, refs, backend="teacher")
    return out, refs


async def test_astra_scoring_writes_under_astra_dir_and_leaves_scene_files_alone(
    tmp_path, astra_generation
):
    out, refs = await paired_run(tmp_path)
    rows = await score_benchmark(
        out, publish=False, judge="astra", references=refs, baseline_arm="teacher"
    )
    assert len(rows) == 6
    assert all(set(r.scores) == {"astra_judge", "efficiency"} for r in rows)
    for name in ("scorer.json", "scored.jsonl", "summary.json", "comparison.html"):
        assert (out / "astra" / name).exists()
        assert not (out / name).exists()
    s = json.loads((out / "astra/summary.json").read_text())
    assert s["judge"] == "bufo-astra-v1" and s["calibration_status"] == "uncalibrated"
    student, teacher = s["arms"]["student"], s["arms"]["teacher"]
    # p1 failed generation (no teacher reference), p2 is weak for the student.
    assert student["total"] == 3 and student["scored"] == 2
    assert student["missing_references"] == 1 and student["judge_errors"] == 0
    assert student["strict_passes"] == 1 and student["strict_pass_rate"] == pytest.approx(1 / 3)
    assert teacher["strict_pass_rate"] == pytest.approx(2 / 3)
    assert student["vs_reference"] == {"better": 0, "equal": 1, "worse": 1}
    assert student["craft_means"]["readability"] == pytest.approx(0.5)
    assert student["serving_path"] == "wandb_inference"
    assert student["latency_p50_s"] == pytest.approx(0.1)
    paired = s["paired"]["student"]
    assert set(s["paired"]) == {"student"}
    assert paired["difference"] == pytest.approx(-1 / 3) and len(paired["ci95"]) == 2
    scorer = json.loads((out / "astra/scorer.json").read_text())
    assert scorer["judge"] == AstraReferenceJudge().configuration()
    assert scorer["efficiency"] == {"scorer": "efficiency-v1"}
    assert scorer["self_reference"] is False
    assert scorer["references"]["version"] == "bufo-references-v1"
    assert len(scorer["references"]["sha256"]) == 64
    # Replays from checkpoint without new judge calls; a references dir or file both work.
    again = await score_benchmark(
        out, publish=False, judge="astra", references=refs / "references.json"
    )
    assert [r.scores for r in again] == [r.scores for r in rows]


async def test_astra_requires_references_unless_self(tmp_path, astra_generation):
    out, _ = await paired_run(tmp_path)
    with pytest.raises(ValueError, match="references"):
        await score_benchmark(out, publish=False, judge="astra")
    with pytest.raises(ValueError, match="judge"):
        await score_benchmark(out, publish=False, judge="clip")


async def test_astra_rescore_rejects_changed_scorer_config(tmp_path, astra_generation, monkeypatch):
    import bufogen.evaluation.benchmark as benchmark

    out, refs = await paired_run(tmp_path)
    await score_benchmark(out, publish=False, judge="astra", references=refs)

    class Stricter(AstraReferenceJudge):
        threshold: float = 0.9

    monkeypatch.setattr(benchmark, "AstraReferenceJudge", Stricter)
    with pytest.raises(ValueError, match="Astra scorer configuration changed"):
        await score_benchmark(out, publish=False, judge="astra", references=refs)


async def test_missing_reference_rows_counted_in_denominator(tmp_path, astra_generation):
    out, refs = await paired_run(tmp_path)
    manifest = json.loads((refs / "references.json").read_text())
    del manifest["rows"]["p0"]
    (refs / "references.json").write_text(json.dumps(manifest))
    await score_benchmark(out, publish=False, judge="astra", references=refs)
    s = json.loads((out / "astra/summary.json").read_text())
    assert s["arms"]["student"]["total"] == 3
    assert s["arms"]["student"]["missing_references"] == 2  # p0 removed, p1 never had one
    assert s["arms"]["teacher"]["strict_pass_rate"] == pytest.approx(1 / 3)
    assert s["paired"] == {}


async def test_self_reference_limit_prompt_ids(tmp_path, astra_generation):
    out = tmp_path / "run"
    await run_benchmark(prompts(), [backend()], out)
    rows = await score_benchmark(
        out, publish=False, judge="astra", self_reference=True, limit_prompt_ids=["p0", "p2"]
    )
    assert sorted(r.prompt_id for r in rows) == ["p0", "p2"]
    assert all(r.benchmark_metadata["reference"]["is_self"] for r in rows)
    assert not (out / "astra").exists()
    s = json.loads((out / "astra-self/summary.json").read_text())
    assert s["arms"]["student"]["total"] == 2 and s["arms"]["student"]["scored"] == 2
    scorer = json.loads((out / "astra-self/scorer.json").read_text())
    assert scorer["self_reference"] is True and scorer["references"] is None
    with pytest.raises(ValueError, match="prompt"):
        await score_benchmark(
            out, publish=False, judge="astra", self_reference=True, limit_prompt_ids=["nope"]
        )


async def test_scene_scoring_ignores_unlisted_new_scorer_files(
    tmp_path,
    fake_generation,  # noqa: F811
    monkeypatch,
):
    import bufogen.evaluation.benchmark as benchmark
    from bufogen.evaluation.scorers.scene_quality import SceneQuality, failure_result
    from bufogen.settings import ROOT

    tree = tmp_path / "tree"
    for relative in ("src/bufogen/evaluation/scorers/scene_quality.py", "config/brand.yaml"):
        (tree / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, tree / relative)
    monkeypatch.setattr(benchmark, "ROOT", tree)
    out = tmp_path / "run"
    await run_benchmark(prompts(), [backend()], out)
    listed = json.loads((out / "benchmark.json").read_text())["configuration"]["source_hashes"]
    assert set(listed) == {"src/bufogen/evaluation/scorers/scene_quality.py", "config/brand.yaml"}

    async def fake_score(self, output, rescore=False):
        return failure_result("judge_error") | {
            "scorer_configuration": self.configuration(),
            "requirements": output["benchmark_metadata"]["requirements"],
            "generation_id": output["id"],
        }

    monkeypatch.setattr(SceneQuality, "score", fake_score)
    (tree / "src/bufogen/evaluation/scorers/stray_new_scorer.py").write_text("X = 1\n")
    (tree / "assets/bufo_style").mkdir(parents=True)
    (tree / "assets/bufo_style/new.png").write_bytes(b"new")
    assert len(await score_benchmark(out, publish=False)) == 3
    (tree / "src/bufogen/evaluation/scorers/scene_quality.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="scorer source changed"):
        await score_benchmark(out, publish=False)


def generation(row_id="g1"):
    row = BufoGeneration(
        prompt_id="p",
        prompt="p",
        backend="b",
        model="m",
        group_id="g",
        svg="",
        valid=False,
        latency_s=0,
        id=row_id,
    )
    row.scores["astra_judge"] = {"status": "ok", "generation_id": row.id}
    row.scores["efficiency"] = {"status": "completed", "generation_id": row.id}
    return row


def test_eval_coverage_requires_one_score_per_scorer():
    row = generation()
    keys = {"AstraReferenceJudge": "astra_judge", "EfficiencyScorer": "efficiency"}
    both = {
        "AstraReferenceJudge": row.scores["astra_judge"],
        "EfficiencyScorer": row.scores["efficiency"],
    }
    _validate_eval_coverage([{"scores": both}], [row], score_keys=keys)
    with pytest.raises(RuntimeError, match="omitted"):
        _validate_eval_coverage(
            [{"scores": {"AstraReferenceJudge": row.scores["astra_judge"]}}], [row], score_keys=keys
        )
    wrong = dict(both, EfficiencyScorer={"status": "completed", "generation_id": "other"})
    with pytest.raises(RuntimeError, match="omitted"):
        _validate_eval_coverage([{"scores": wrong}], [row], score_keys=keys)
    with pytest.raises(RuntimeError, match="omitted"):
        _validate_eval_coverage([], [row], score_keys=keys)
    # Same generation_id, changed content under its own scorer name.
    changed = dict(both, AstraReferenceJudge={"status": "judge_error", "generation_id": row.id})
    with pytest.raises(RuntimeError, match="changed"):
        _validate_eval_coverage([{"scores": changed}], [row], score_keys=keys)
    # Correct values filed under the wrong scorers are rejected, not matched anywhere.
    swapped = {
        "AstraReferenceJudge": row.scores["efficiency"],
        "EfficiencyScorer": row.scores["astra_judge"],
    }
    with pytest.raises(RuntimeError, match="changed"):
        _validate_eval_coverage([{"scores": swapped}], [row], score_keys=keys)
    with pytest.raises(ValueError, match="mapping"):
        _validate_eval_coverage([{"scores": both}], [row], score_keys=tuple(keys.values()))


def test_astra_summary_counts_other_failures_separately():
    from bufogen.evaluation.benchmark import _astra_summary
    from bufogen.evaluation.scorers.astra_judge import judge_failure

    settings = {
        "run_id": "r",
        "configuration_hash": "h",
        "configuration": {"arms": [{"name": "b", "provider": "coreweave"}]},
    }
    rows = []
    for index, status in enumerate(["judge_error", "missing_image", "missing_reference"]):
        row = generation(f"g{index}")
        row.benchmark_metadata = {"split": "train", "family": f"f{index}"}
        row.scores["astra_judge"] = judge_failure(status)
        row.scores["efficiency"] = {
            "latency_s": 1.0,
            "output_tokens": 10,
            "reasoning_tokens": 0,
            "truncated_calls": 0,
        }
        rows.append(row)
    arm = _astra_summary(rows, settings, baseline_arm=None)["arms"]["b"]
    assert arm["total"] == 3 and arm["scored"] == 0 and arm["strict_pass_rate"] == 0
    assert arm["judge_errors"] == 1 and arm["other_failures"] == 1
    assert arm["missing_references"] == 1 and arm["reference_mismatches"] == 0


async def test_uneven_selected_families_are_paired_and_flagged(
    tmp_path, astra_generation, monkeypatch
):
    uneven = prompts()
    uneven[2]["family"] = uneven[0]["family"]  # family0 has p0 and p2, family1 has p1
    out = tmp_path / "run"
    await run_benchmark(uneven, [backend(), backend("teacher")], out)
    refs = tmp_path / "refs"
    build_reference_manifest(out, refs, backend="teacher")
    calls = []

    async def counting_judge(self, output, requirements, reference_png):
        calls.append(output["id"])
        return judgment(requirements)

    monkeypatch.setattr(AstraReferenceJudge, "_judge", counting_judge)
    await score_benchmark(
        out,
        publish=False,
        judge="astra",
        references=refs,
        baseline_arm="teacher",
    )
    summary = json.loads((out / "astra/summary.json").read_text())
    paired = summary["paired"]["student"]
    assert paired["families"] == 2 and paired["prompts"] == 3
    assert paired["balanced"] is False
    assert paired["prompts_per_family"] == {uneven[0]["family"]: 2, uneven[1]["family"]: 1}
    assert calls  # every row was judged; none was held back by a balance check


async def test_all_judge_errors_fail_loudly_and_never_publish(
    tmp_path, astra_generation, monkeypatch
):
    out = tmp_path / "run"
    await run_benchmark(prompts(), [backend()], out)
    refs = tmp_path / "refs"
    build_reference_manifest(out, refs, backend="student")

    class Rejected(Exception):
        status_code = 400

    async def failing_judge(self, output, requirements, reference_png):
        raise Rejected("Unsupported parameter: the provider message must not leak")

    monkeypatch.setattr(AstraReferenceJudge, "_judge", failing_judge)
    published = []
    monkeypatch.setattr(
        "bufogen.evaluation.benchmark._publish_astra", lambda *a, **k: published.append(a) or _noop()
    )
    with pytest.raises(RuntimeError, match="no verdicts"):
        await score_benchmark(out, publish=True, judge="astra", references=refs)
    assert published == []
    scored = [json.loads(line) for line in (out / "astra/scored.jsonl").read_text().splitlines()]
    errors_rows = [r for r in scored if r["scores"]["astra_judge"]["status"] == "judge_error"]
    assert errors_rows and {r["scores"]["astra_judge"]["error"] for r in errors_rows} == {
        "Rejected: image judgment failed (HTTP 400)"
    }
    summary = json.loads((out / "astra/summary.json").read_text())
    statuses = [r["scores"]["astra_judge"]["status"] for r in scored]
    assert "ok" not in statuses and summary["arms"]["student"]["judge_errors"] == len(errors_rows)


async def test_rescoring_retries_cached_judge_errors_only(tmp_path, astra_generation, monkeypatch):
    out = tmp_path / "run"
    await run_benchmark(prompts(), [backend()], out)
    refs = tmp_path / "refs"
    build_reference_manifest(out, refs, backend="student")
    attempts = []

    async def flaky_judge(self, output, requirements, reference_png):
        attempts.append(output["prompt_id"])
        if len(attempts) == 1:  # the first judged row fails once
            raise RuntimeError("transient")
        return judgment(requirements)

    monkeypatch.setattr(AstraReferenceJudge, "_judge", flaky_judge)
    first = await score_benchmark(out, publish=False, judge="astra", references=refs)
    errored = [r for r in first if r.scores["astra_judge"]["status"] == "judge_error"]
    assert [r.prompt_id for r in errored] == [attempts[0]]
    judged_first = len(attempts)
    second = await score_benchmark(out, publish=False, judge="astra", references=refs)
    assert attempts[judged_first:] == [attempts[0]]
    assert all(r.scores["astra_judge"]["status"] != "judge_error" for r in second)
    scored = [json.loads(line) for line in (out / "astra/scored.jsonl").read_text().splitlines()]
    assert len(scored) == len({r["id"] for r in scored}) == len(second)


async def _noop():
    return None


async def test_publish_astra_rejects_stale_arm_publication(tmp_path, astra_generation, monkeypatch):
    import weave

    import bufogen.integrations.telemetry as telemetry

    out, refs = await paired_run(tmp_path)
    await score_benchmark(out, publish=False, judge="astra", references=refs)
    monkeypatch.setattr(telemetry, "init_weave", lambda: SimpleNamespace(flush=lambda: None))

    def no_publish(obj):
        raise AssertionError("must not republish")

    monkeypatch.setattr(weave, "publish", no_publish)
    state_path = out / "astra/weave-publication.json"
    full = ["p0", "p1", "p2"]
    state = {
        "evaluations": {
            "student": {"ref": "x", "summary": {}, "scored_rows": 3, "prompt_ids": full},
            "teacher": {"ref": "y", "summary": {}, "scored_rows": 3, "prompt_ids": full},
        }
    }
    state_path.write_text(json.dumps(state))
    # Identical rows already published: replay skips both arms.
    await score_benchmark(out, judge="astra", references=refs)
    # An arm published from a subset is an error, never silently stale or republished.
    state["evaluations"]["teacher"] |= {"scored_rows": 2, "prompt_ids": ["p0", "p2"]}
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="teacher.*new output directory"):
        await score_benchmark(out, judge="astra", references=refs)


REQUIRED_METADATA = {
    "rubric_version",
    "judge_samples",
    "judge_model",
    "judge_prompt_sha256",
    "schema_hash",
    "prompt_version",
    "configuration_hash",
    "prompt_set_sha256",
    "backend",
    "model",
    "base_model",
    "serving_path",
    "checkpoint_artifact",
    "checkpoint_artifact_url",
    "max_turns",
    "max_tokens",
    "disable_thinking",
    "retry_loop_enabled",
    "web_search_enabled",
    "reference_run_id",
    "self_reference",
    "split",
    "code_sha256",
    "confirmation",
    "calibration_status",
    "calibration_record",
    "forge_job_id",
    "forge_dataset_id",
    "sft_config",
    "checkpoint_version",
}


async def test_publish_astra_metadata_has_no_env_and_has_required_keys(
    tmp_path, astra_generation, monkeypatch
):
    from dataclasses import replace

    import weave

    import bufogen.integrations.telemetry as telemetry

    out = tmp_path / "run"
    artifact = "wandb-artifact:///team/proj/tuned:v3"
    sidecar = tmp_path / "checkpoints.json"
    sidecar.write_text(
        json.dumps(
            [
                {
                    "arm_name": "teacher",
                    "artifact": artifact,
                    "forge_job_id": "job-1",
                    "forge_dataset_id": "ds-1",
                    "sft_config": SFT,
                }
            ]
        )
    )
    monkeypatch.setattr("bufogen.evaluation.benchmark.CHECKPOINTS_SIDECAR", sidecar)
    await run_benchmark(prompts(), [backend(), replace(backend("teacher"), model=artifact)], out)
    refs = tmp_path / "refs"
    build_reference_manifest(out, refs, backend="teacher")
    for name in ("OPEN_MODEL", "SOL_MODEL", "TEST_MODEL", "WANDB_PROJECT"):
        monkeypatch.setenv(name, "ENV-LEAK-VALUE")

    published, datasets, evaluations = [], [], []

    class FakeDataset:
        def __init__(self, name, rows):
            self.name, self.rows = name, rows
            datasets.append(self)

    class FakeEvaluation:
        def __init__(self, *, name, dataset, scorers, metadata, evaluation_name):
            self.name, self.dataset, self.scorers, self.metadata = name, dataset, scorers, metadata
            evaluations.append(self)

        async def get_eval_results(self, model):
            table = []
            for item in self.dataset.rows:
                if hasattr(model, "predict"):
                    # Shared prompt-only dataset: the arm replays its saved generation by id.
                    assert "generation" not in item
                    output = model.predict(prompt=item["prompt"], prompt_id=item["prompt_id"])
                else:
                    output = {"generation": item["generation"]}
                scores = {}
                for scorer in self.scorers:
                    value = scorer.score(output)
                    scores[type(scorer).__name__] = (
                        await value if hasattr(value, "__await__") else value
                    )
                table.append({"output": output, "scores": scores})
            return SimpleNamespace(rows=table)

        async def evaluate(self, model):
            self.model = model
            return {"rows": len((await self.get_eval_results(model)).rows)}

    monkeypatch.setattr(weave, "Dataset", FakeDataset)
    monkeypatch.setattr(weave, "Evaluation", FakeEvaluation)
    monkeypatch.setattr(
        weave,
        "publish",
        lambda obj: published.append(obj) or SimpleNamespace(uri=lambda: "weave:///x"),
    )
    monkeypatch.setattr(telemetry, "init_weave", lambda: SimpleNamespace(flush=lambda: None))

    monkeypatch.setattr("bufogen.evaluation.references.ROOT", tmp_path)
    await score_benchmark(out, judge="astra", references=refs, baseline_arm="teacher")
    assert len(evaluations) == 2 and len(published) == 2
    by_backend = {e.metadata["backend"]: e for e in evaluations}
    for evaluation in evaluations:
        meta = evaluation.metadata
        assert set(meta) == REQUIRED_METADATA
        assert "ENV-LEAK-VALUE" not in json.dumps(meta)
        assert meta["retry_loop_enabled"] is True and meta["web_search_enabled"] is False
        assert meta["self_reference"] is False and meta["split"] == "train"
        assert meta["confirmation"] is False
        assert meta["max_tokens"] == 6000 and meta["max_turns"] == 100
        assert meta["judge_model"] == "gpt-6-astra" and meta["rubric_version"] == "bufo-astra-v1"
        assert set(meta["code_sha256"]) == {
            "agent.py",
            "scorers/astra_judge.py",
            "svg.py",
            "render.py",
        }
        assert evaluation.name.startswith("bufo-astra-")
    settings = json.loads((out / "benchmark.json").read_text())
    student = by_backend["student"].metadata
    assert student["checkpoint_artifact"] is None and student["model"] == "test-model"
    assert student["configuration_hash"] == settings["configuration_hash"]
    assert student["prompt_set_sha256"] == settings["configuration"]["prompt_hash"]
    assert student["reference_run_id"] == settings["run_id"]
    assert student["serving_path"] == "wandb_inference"
    assert by_backend["teacher"].metadata["checkpoint_artifact"] == artifact
    # One prompt-only dataset shared by every arm; the outputs live in each arm's replay model.
    assert len({d.name for d in datasets}) == 1 and datasets[0].name.startswith("bufo-")
    assert all("generation" not in item for d in datasets for item in d.rows)
    generations = [g for e in evaluations for g in e.model._generations.values()]
    assert generations
    published_text = json.dumps([d.rows for d in datasets] + generations)
    assert "/Users/" not in published_text and str(tmp_path) not in published_text
    for generation in generations:
        for key in ("png_path", "svg_path", "input_image_path"):
            value = generation.get(key)
            assert not value or not Path(value).is_absolute()
    for evaluation in evaluations:
        assert evaluation.metadata["calibration_status"] == "uncalibrated"
        assert evaluation.metadata["calibration_record"] is None
    first = datasets[0].rows[0]
    assert set(first) == {"prompt", "prompt_id", "family", "split", "requirements"}
    # Per-arm facts (reference, scores, paths) travel with the replayed generation instead.
    replayed = next(iter(evaluations[0].model._generations.values()))
    assert {"benchmark_metadata", "scores"} <= set(replayed)
    state = json.loads((out / "astra/weave-publication.json").read_text())
    assert set(state["evaluations"]) == {"student", "teacher"}
    assert state["evaluations"]["student"]["prompt_ids"] == ["p0", "p1", "p2"]
    assert state["evaluations"]["student"]["scored_rows"] == 3


async def test_scene_publish_keeps_names_metadata_and_coverage(
    tmp_path,
    fake_generation,  # noqa: F811
    monkeypatch,
):
    import weave

    import bufogen.integrations.telemetry as telemetry
    from bufogen.evaluation.benchmark import _publish
    from bufogen.evaluation.scorers.scene_quality import SceneQuality, failure_result
    from bufogen.models import generation_model_presentation

    await run_benchmark(prompts(), [backend()], tmp_path)

    async def fake_score(self, output, rescore=False, saved_score=None):
        output = output.get("generation", output)
        if "scene_quality" in output.get("scores", {}):
            return output["scores"]["scene_quality"]
        return failure_result("judge_error") | {
            "scorer_configuration": self.configuration(),
            "requirements": output["benchmark_metadata"]["requirements"],
            "generation_id": output["id"],
        }

    monkeypatch.setattr(SceneQuality, "score", fake_score)
    rows = await score_benchmark(tmp_path, publish=False)
    settings = json.loads((tmp_path / "benchmark.json").read_text())
    (tmp_path / "weave-publication.json").write_text(
        json.dumps({"feedback": {r.id: {"call_id": "c"} for r in rows}, "evaluations": {}})
    )
    created = []

    class FakeEvaluation:
        def __init__(self, *, name, dataset, scorers, metadata, evaluation_name):
            self.name, self.dataset, self.scorers, self.metadata = name, dataset, scorers, metadata
            created.append(self)

        async def get_eval_results(self, model):
            table = []
            for item in self.dataset.rows:
                if hasattr(model, "predict"):
                    # Shared prompt-only dataset: the arm replays its saved generation by id.
                    assert "generation" not in item
                    output = model.predict(prompt=item["prompt"], prompt_id=item["prompt_id"])
                else:
                    output = {"generation": item["generation"]}
                table.append({"scores": {"SceneQuality": await self.scorers[0].score(output)}})
            return SimpleNamespace(rows=table)

        async def evaluate(self, model):
            self.model = model
            return {"rows": len((await self.get_eval_results(model)).rows)}

    monkeypatch.setattr(weave, "Dataset", lambda name, rows: SimpleNamespace(name=name, rows=rows))
    monkeypatch.setattr(weave, "Evaluation", FakeEvaluation)
    monkeypatch.setattr(weave, "publish", lambda obj: SimpleNamespace(uri=lambda: "weave:///s"))
    monkeypatch.setattr(telemetry, "init_weave", lambda: SimpleNamespace(flush=lambda: None))
    await _publish(rows, tmp_path, settings, SceneQuality())
    (evaluation,) = created
    presentation = generation_model_presentation(
        {"backend": "student", "model": "test-model"},
        config=settings["configuration"]["arms"][0],
    )
    label = presentation["model_label"]
    assert type(evaluation).__name__ == "CompleteSceneEvaluation"
    assert evaluation.name == f"bufo-scene-{label}"
    assert evaluation.dataset.name == f"bufo-scene-{settings['run_id']}-{label}"
    assert set(evaluation.dataset.rows[0]) == {"prompt", "generation"}
    assert evaluation.metadata == {"configuration_hash": settings["configuration_hash"]} | (
        presentation
    )
    state = json.loads((tmp_path / "weave-publication.json").read_text())
    assert state["evaluations"]["student"] == {
        "ref": "weave:///s",
        "summary": {"rows": 3},
        "scored_rows": 3,
    }


@pytest.mark.parametrize("split", ["heldout", "test"])
async def test_astra_refuses_heldout_rows_without_confirmation(
    tmp_path, astra_generation, monkeypatch, split
):
    rows = prompts()
    rows[2]["split"] = split
    out = tmp_path / "run"
    await run_benchmark(rows, [backend()], out)
    calls = []

    async def counting_judge(self, output, requirements, reference_png):
        calls.append(output["prompt_id"])
        return judgment(requirements)

    monkeypatch.setattr(AstraReferenceJudge, "_judge", counting_judge)
    with pytest.raises(ValueError, match="confirmation"):
        await score_benchmark(out, publish=False, judge="astra", self_reference=True)
    assert calls == [] and not (out / "astra-self").exists()
    # Rows outside the confirmation splits may still be judged from the same run.
    judged = await score_benchmark(
        out, publish=False, judge="astra", self_reference=True, limit_prompt_ids=["p0", "p1"]
    )
    assert sorted(r.prompt_id for r in judged) == ["p0", "p1"]
    scorer = json.loads((out / "astra-self/scorer.json").read_text())
    assert scorer["confirmation"] is False


async def test_astra_confirmation_flag_is_recorded(tmp_path, astra_generation):
    rows = prompts()
    for row in rows:
        row["split"] = "heldout"
    out = tmp_path / "run"
    await run_benchmark(rows, [backend()], out)
    judged = await score_benchmark(
        out, publish=False, judge="astra", self_reference=True, confirmation=True
    )
    assert len(judged) == 3
    scorer = json.loads((out / "astra-self/scorer.json").read_text())
    assert scorer["confirmation"] is True
    # The record is frozen: a later non-confirmation pass over the same rows is refused.
    with pytest.raises(ValueError, match="confirmation"):
        await score_benchmark(out, publish=False, judge="astra", self_reference=True)


async def test_scorer_record_without_confirmation_key_still_replays(tmp_path, astra_generation):
    out, refs = await paired_run(tmp_path)
    first = await score_benchmark(out, publish=False, judge="astra", references=refs)
    path = out / "astra/scorer.json"
    record = json.loads(path.read_text())
    del record["confirmation"]  # written before the confirmation guard existed
    path.write_text(json.dumps(record))
    again = await score_benchmark(out, publish=False, judge="astra", references=refs)
    assert [r.scores for r in again] == [r.scores for r in first]


async def test_scene_judge_rejects_confirmation(tmp_path, astra_generation):
    out, _ = await paired_run(tmp_path)
    with pytest.raises(ValueError, match="judge='astra'"):
        await score_benchmark(out, publish=False, confirmation=True)


def fake_weave(monkeypatch):
    """Offline stand-ins for weave.Dataset/Evaluation/publish; returns the published list."""
    import weave

    import bufogen.integrations.telemetry as telemetry

    published, evaluations = [], []

    class FakeDataset:
        def __init__(self, name, rows):
            self.name, self.rows = name, rows

    class FakeEvaluation:
        def __init__(self, *, name, dataset, scorers, metadata, evaluation_name):
            self.name, self.dataset, self.scorers, self.metadata = name, dataset, scorers, metadata
            evaluations.append(self)

        async def get_eval_results(self, model):
            table = []
            for item in self.dataset.rows:
                if hasattr(model, "predict"):
                    # Shared prompt-only dataset: the arm replays its saved generation by id.
                    assert "generation" not in item
                    output = model.predict(prompt=item["prompt"], prompt_id=item["prompt_id"])
                else:
                    output = {"generation": item["generation"]}
                scores = {}
                for scorer in self.scorers:
                    value = scorer.score(output)
                    scores[type(scorer).__name__] = (
                        await value if hasattr(value, "__await__") else value
                    )
                table.append({"output": output, "scores": scores})
            return SimpleNamespace(rows=table)

        async def evaluate(self, model):
            self.model = model
            return {"rows": len((await self.get_eval_results(model)).rows)}

    def publish(obj):
        published.append(obj)
        ref = f"weave:///eval/{len(published)}"
        return SimpleNamespace(uri=lambda: ref)

    monkeypatch.setattr(weave, "Dataset", FakeDataset)
    monkeypatch.setattr(weave, "Evaluation", FakeEvaluation)
    monkeypatch.setattr(weave, "publish", publish)
    monkeypatch.setattr(telemetry, "init_weave", lambda: SimpleNamespace(flush=lambda: None))
    return published, evaluations


async def test_republish_after_retried_judge_errors_supersedes(
    tmp_path, astra_generation, monkeypatch
):
    out = tmp_path / "run"
    await run_benchmark(prompts(), [backend()], out)
    refs = tmp_path / "refs"
    build_reference_manifest(out, refs, backend="student")
    attempts = []

    async def flaky_judge(self, output, requirements, reference_png):
        attempts.append(output["prompt_id"])
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return judgment(requirements)

    monkeypatch.setattr(AstraReferenceJudge, "_judge", flaky_judge)
    published, evaluations = fake_weave(monkeypatch)
    state_path = out / "astra/weave-publication.json"

    first = await score_benchmark(out, judge="astra", references=refs)
    assert sum(r.scores["astra_judge"]["status"] == "judge_error" for r in first) == 1
    assert len(published) == 1
    state = json.loads(state_path.read_text())
    entry = state["evaluations"]["student"]
    assert entry["ref"] == "weave:///eval/1" and len(entry["scores_sha256"]) == 64
    assert "superseded" not in entry

    # The retry turns the judge error into a verdict: the publication must not be skipped.
    second = await score_benchmark(out, judge="astra", references=refs)
    assert all(r.scores["astra_judge"]["status"] != "judge_error" for r in second)
    assert len(published) == 2
    entry = json.loads(state_path.read_text())["evaluations"]["student"]
    assert entry["ref"] == "weave:///eval/2"
    assert entry["superseded"] == ["weave:///eval/1"]
    assert entry["scores_sha256"] != state["evaluations"]["student"]["scores_sha256"]
    shown = [g["scores"]["astra_judge"] for g in evaluations[-1].model._generations.values()]
    assert all(score["status"] != "judge_error" for score in shown)

    # Unchanged scores replay without publishing again.
    await score_benchmark(out, judge="astra", references=refs)
    assert len(published) == 2
    assert json.loads(state_path.read_text())["evaluations"]["student"] == entry


async def test_no_ok_verdicts_never_publish_even_without_judge_errors(
    tmp_path, astra_generation, monkeypatch
):
    out, refs = await paired_run(tmp_path)
    manifest = json.loads((refs / "references.json").read_text())
    manifest["rows"] = {}  # e.g. a wrong --references: every row is missing_reference
    (refs / "references.json").write_text(json.dumps(manifest))
    published = []
    monkeypatch.setattr(
        "bufogen.evaluation.benchmark._publish_astra", lambda *a, **k: published.append(a) or _noop()
    )
    with pytest.raises(RuntimeError, match="no verdicts.*missing_reference"):
        await score_benchmark(out, publish=True, judge="astra", references=refs)
    assert published == []


@pytest.mark.parametrize("legacy_layout", [False, True])
async def test_code_sha256_uses_frozen_generation_hashes(
    tmp_path, astra_generation, monkeypatch, legacy_layout
):
    import hashlib

    from bufogen.evaluation import benchmark
    from bufogen.settings import ROOT

    out, refs = await paired_run(tmp_path)
    path = out / "benchmark.json"
    settings = json.loads(path.read_text())
    frozen = settings["configuration"]["source_hashes"]
    for name in ("agent.py", "svg.py", "render.py"):
        current_path = f"src/bufogen/{benchmark.ASTRA_CODE_PATHS[name]}"
        frozen.pop(current_path, None)
        path_key = f"src/bufogen/{name}" if legacy_layout else current_path
        frozen[path_key] = name[0] * 64  # stands in for generation-time bytes
    path.write_text(json.dumps(settings))
    _, evaluations = fake_weave(monkeypatch)
    await score_benchmark(out, judge="astra", references=refs)
    for evaluation in evaluations:
        code = evaluation.metadata["code_sha256"]
        assert code["agent.py"] == "a" * 64
        assert code["svg.py"] == "s" * 64
        assert code["render.py"] == "r" * 64
        # The judge is not part of generation; its hash is the scoring-time file.
        current = hashlib.sha256((ROOT / "src/bufogen/evaluation/scorers/astra_judge.py").read_bytes())
        assert code["scorers/astra_judge.py"] == current.hexdigest()


async def test_rubric_v2_scores_into_its_own_dir_and_honours_generation_ids(
    tmp_path, astra_generation, monkeypatch
):
    out, refs = await paired_run(tmp_path)
    # v1 first, so the v2 run can be shown to leave the v1 directory untouched.
    v1 = await score_benchmark(out, publish=False, judge="astra", references=refs)
    v1_bytes = (out / "astra/scored.jsonl").read_bytes()

    async def v2_judge(self, output, requirements, reference_png):
        data = judgment(requirements)
        for check in data["prompt_checks"]:
            check["directly_visible"] = True
        # The teacher's p2 contact is inferred, not seen: v2 must not pass it.
        if output["backend"] == "teacher" and output["prompt_id"] == "p2":
            data["prompt_checks"][0]["directly_visible"] = False
        return data

    monkeypatch.setattr(AstraReferenceJudge, "_judge", v2_judge)
    wanted = sorted(r.id for r in v1 if r.backend == "teacher")
    rows = await score_benchmark(
        out,
        publish=False,
        judge="astra",
        references=refs,
        rubric_version="bufo-astra-v2",
        limit_generation_ids=wanted,
    )
    assert sorted(r.id for r in rows) == wanted
    assert (out / "astra-v2/scored.jsonl").exists()
    assert (out / "astra/scored.jsonl").read_bytes() == v1_bytes
    scorer = json.loads((out / "astra-v2/scorer.json").read_text())
    assert scorer["judge"]["rubric_version"] == "bufo-astra-v2"
    by_prompt = {r.prompt_id: r.scores["astra_judge"] for r in rows}
    assert by_prompt["p2"]["strict_pass"] is False
    assert by_prompt["p2"]["visibility_overrides"] == [by_prompt["p2"]["requirements"][0]]
    assert all(s["rubric_version"] == "bufo-astra-v2" for s in by_prompt.values())
    with pytest.raises(ValueError, match="limit_generation_ids"):
        await score_benchmark(
            out,
            publish=False,
            judge="astra",
            references=refs,
            rubric_version="bufo-astra-v2",
            limit_generation_ids=["not-a-row"],
        )
    with pytest.raises(ValueError, match="rubric"):
        await score_benchmark(
            out, publish=False, judge="astra", references=refs, rubric_version="v9"
        )
    with pytest.raises(ValueError, match="require judge='astra'"):
        await score_benchmark(out, publish=False, rubric_version="bufo-astra-v2")


async def test_judge_samples_get_their_own_dir_and_suffix(tmp_path, astra_generation, monkeypatch):
    from bufogen.evaluation.benchmark import _astra_dir_name, _rubric_suffix

    assert _astra_dir_name(False, "bufo-astra-v3", 3) == "astra-v3-k3"
    assert _astra_dir_name(True, "bufo-astra-v1", 1) == "astra-self"
    assert _astra_dir_name(False, "bufo-astra-v1", 2) == "astra-k2"
    assert _rubric_suffix("bufo-astra-v3", 3) == "-v3-k3"
    out, refs = await paired_run(tmp_path)

    async def v3_judge(self, output, requirements, reference_png):
        data = judgment(requirements)
        for check in data["prompt_checks"]:
            check["directly_visible"] = True
        return data

    monkeypatch.setattr(AstraReferenceJudge, "_judge", v3_judge)
    rows = await score_benchmark(
        out,
        publish=False,
        judge="astra",
        references=refs,
        rubric_version="bufo-astra-v3",
        judge_samples=2,
    )
    assert (out / "astra-v3-k2/scorer.json").exists()
    scorer = json.loads((out / "astra-v3-k2/scorer.json").read_text())
    assert scorer["judge"]["samples"] == 2
    ok = [r for r in rows if r.scores["astra_judge"]["status"] == "ok"]
    assert ok and all(r.scores["astra_judge"]["samples"] == 2 for r in ok)
    with pytest.raises(ValueError, match="require judge='astra'"):
        await score_benchmark(out, publish=False, judge_samples=2)


async def test_published_evaluation_name_and_metadata_carry_rubric_and_samples(
    tmp_path, astra_generation, monkeypatch
):
    import bufogen.evaluation.benchmark as benchmark
    from bufogen.integrations import telemetry

    out, refs = await paired_run(tmp_path)

    async def v3_judge(self, output, requirements, reference_png):
        data = judgment(requirements)
        for check in data["prompt_checks"]:
            check["directly_visible"] = True
        return data

    monkeypatch.setattr(AstraReferenceJudge, "_judge", v3_judge)
    monkeypatch.setattr(telemetry, "init_weave", lambda: SimpleNamespace(flush=lambda: None))
    seen = []

    async def fake_publish(state, path, name, **kw):
        seen.append((name, kw["evaluation_name"], kw["dataset_name"], kw["metadata"]))
        state["evaluations"][name] = {"ref": "weave:///x", "scored_rows": len(kw["subset"])} | (
            kw.get("record") or {}
        )

    monkeypatch.setattr(benchmark, "_publish_arm_evaluation", fake_publish)
    await score_benchmark(
        out, judge="astra", references=refs, rubric_version="bufo-astra-v3", judge_samples=3
    )
    assert seen, "nothing published"
    for name, evaluation_name, dataset_name, metadata in seen:
        assert evaluation_name.endswith("-v3-k3")
        assert metadata["rubric_version"] == "bufo-astra-v3" and metadata["judge_samples"] == 3
    # Every arm is published against the same prompt-only dataset so the UI can compare them.
    assert len({dataset_name for _, _, dataset_name, _ in seen}) == 1
    assert next(iter(seen))[2].startswith("bufo-") and "-v3" not in next(iter(seen))[2]
    seen.clear()
    out2, refs2 = await paired_run(tmp_path / "second")
    await score_benchmark(out2, judge="astra", references=refs2)
    for name, evaluation_name, dataset_name, metadata in seen:
        assert not evaluation_name.endswith("-k1") and "-v1" not in evaluation_name
        assert metadata["judge_samples"] == 1


def test_astra_metadata_reports_registry_status_at_publish_time(tmp_path, monkeypatch):
    from bufogen.evaluation import calibration
    from bufogen.evaluation.benchmark import _astra_metadata

    settings = {
        "run_id": "r",
        "configuration_hash": "h",
        "configuration": {
            "policy": {"max_turns_per_attempt": 1, "max_tokens": 1},
            "prompt_version": "v",
            "prompt_hash": "p",
        },
    }
    arm = {"name": "a", "model": "m", "provider": "x"}
    # The saved scorer.json still says uncalibrated; the registry decides at publish time.
    judge = dict(calibration.CALIBRATED_CONFIGS[("bufo-astra-v5", 3)])
    judge["calibration_status"] = "uncalibrated"
    record = {"judge": judge, "references": None, "self_reference": False}
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"gate": {"passed": True}}))
    monkeypatch.setitem(calibration.CALIBRATED, ("bufo-astra-v5", 3), str(path))
    meta = _astra_metadata(arm, [], settings, record, {}, {})
    assert meta["calibration_status"] == "calibrated"
    assert meta["calibration_record"] == str(path)
    path.write_text(json.dumps({"gate": {"passed": False}}))
    meta = _astra_metadata(arm, [], settings, record, {}, {})
    assert meta["calibration_status"] == "uncalibrated" and meta["calibration_record"] is None


CHECKPOINT_KEYS = {"forge_job_id", "forge_dataset_id", "sft_config", "checkpoint_version"}
SFT = {
    "epochs": 3,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "lora_rank": 1,
    "assistant_turns": "all",
}


def _checkpoint_inputs():
    settings = {
        "run_id": "r",
        "configuration_hash": "h",
        "configuration": {
            "policy": {"max_turns_per_attempt": 1, "max_tokens": 1},
            "prompt_version": "v",
            "prompt_hash": "p",
        },
    }
    judge = {
        "rubric_version": "bufo-astra-v5",
        "samples": 3,
        "judge_model": "j",
        "rubric_hash": "r",
        "schema_hash": "s",
    }
    record = {"judge": judge, "references": None, "self_reference": False}
    return settings, record


def _entry(**changes):
    return {
        "arm_name": "sft_j1",
        "artifact": "wandb-artifact:///e/p/sft-j1:v2",
        "forge_job_id": "job-9",
        "forge_dataset_id": "ds-9",
        "sft_config": SFT,
        "adapter_rank_observed": 1,
        **changes,
    }


def test_checkpoint_metadata_comes_from_sidecar_for_artifact_arms(tmp_path, monkeypatch):
    import bufogen.evaluation.benchmark as benchmark

    sidecar = tmp_path / "checkpoints.json"
    sidecar.write_text(json.dumps([_entry()]))
    monkeypatch.setattr(benchmark, "CHECKPOINTS_SIDECAR", sidecar)
    settings, record = _checkpoint_inputs()
    arm = {"name": "sft_j1", "model": "wandb-artifact:///e/p/sft-j1:v2", "provider": "coreweave"}
    meta = benchmark._astra_metadata(arm, [], settings, record, {}, {})
    assert meta["forge_job_id"] == "job-9" and meta["forge_dataset_id"] == "ds-9"
    assert meta["sft_config"] == SFT and meta["checkpoint_version"] == "v2"
    plain = {"name": "base", "model": "google/gemma", "provider": "coreweave"}
    meta = benchmark._astra_metadata(plain, [], settings, record, {}, {})
    assert {meta[key] for key in CHECKPOINT_KEYS if key != "sft_config"} == {None}
    assert meta["sft_config"] is None


def test_checkpoint_metadata_may_ride_on_the_arm_config(tmp_path, monkeypatch):
    import bufogen.evaluation.benchmark as benchmark

    monkeypatch.setattr(benchmark, "CHECKPOINTS_SIDECAR", tmp_path / "absent.json")
    settings, record = _checkpoint_inputs()
    arm = {
        "name": "x",
        "model": "wandb-artifact:///e/p/x:v7",
        "provider": "coreweave",
        "forge_job_id": "j",
        "forge_dataset_id": "d",
        "sft_config": SFT,
    }
    meta = benchmark._astra_metadata(arm, [], settings, record, {}, {})
    assert meta["forge_job_id"] == "j" and meta["checkpoint_version"] == "v7"


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [_entry(arm_name="other")],
        [_entry(artifact="wandb-artifact:///e/p/sft-j1:v3")],
        [_entry(forge_job_id="")],
        [_entry(forge_dataset_id=None)],
        [_entry(sft_config={})],
        [_entry(sft_config={"epochs": 3, "lora_rank": 1})],
    ],
)
def test_publish_refuses_artifact_arm_without_checkpoint_metadata(tmp_path, monkeypatch, entries):
    import bufogen.evaluation.benchmark as benchmark

    sidecar = tmp_path / "checkpoints.json"
    sidecar.write_text(json.dumps(entries))
    monkeypatch.setattr(benchmark, "CHECKPOINTS_SIDECAR", sidecar)
    settings, record = _checkpoint_inputs()
    arm = {"name": "sft_j1", "model": "wandb-artifact:///e/p/sft-j1:v2", "provider": "coreweave"}
    with pytest.raises(ValueError, match="checkpoint metadata"):
        benchmark._astra_metadata(arm, [], settings, record, {}, {})
    with pytest.raises(ValueError, match="checkpoint metadata"):
        benchmark._require_checkpoint_metadata([arm])


def test_shipped_sidecar_is_a_json_list():
    import bufogen.evaluation.benchmark as benchmark

    entries = json.loads(benchmark.CHECKPOINTS_SIDECAR.read_text())
    assert isinstance(entries, list)
    for entry in entries:
        arm = {"name": entry["arm_name"], "model": entry["artifact"]}
        assert benchmark._checkpoint_metadata(arm)["forge_job_id"] == entry["forge_job_id"]
        assert entry["adapter_rank_observed"] == entry["sft_config"]["lora_rank"]


def test_checkpoint_version_must_be_immutable_vn_tag_and_supplied_value_is_kept(
    tmp_path, monkeypatch
):
    import bufogen.evaluation.benchmark as benchmark

    monkeypatch.setattr(benchmark, "CHECKPOINTS_SIDECAR", tmp_path / "absent.json")
    base = {"name": "x", "forge_job_id": "j", "forge_dataset_id": "d", "sft_config": SFT}
    latest = {**base, "model": "wandb-artifact:///e/p/x:latest"}
    with pytest.raises(ValueError, match="vN tag"):
        benchmark._checkpoint_metadata(latest)
    untagged = {**base, "model": "wandb-artifact:///e/p/x", "checkpoint_version": "v4"}
    assert benchmark._checkpoint_metadata(untagged)["checkpoint_version"] == "v4"
    alias = {**untagged, "checkpoint_version": "latest"}
    with pytest.raises(ValueError, match="vN tag"):
        benchmark._checkpoint_metadata(alias)


@pytest.mark.parametrize("prefix", ["src/bufogen/scorers", "src/bufogen/evaluation/scorers"])
def test_frozen_scorer_guard_checks_both_layouts(tmp_path, monkeypatch, prefix):
    import hashlib

    from bufogen.evaluation import benchmark

    path = f"{prefix}/scene_quality.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text("original scorer")
    config = {"source_hashes": {path: hashlib.sha256(file.read_bytes()).hexdigest()}}
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    benchmark._check_frozen_scoring_sources(config)
    file.write_text("changed scorer")
    with pytest.raises(ValueError, match="scorer source changed"):
        benchmark._check_frozen_scoring_sources(config)
    file.unlink()
    with pytest.raises(ValueError, match="scorer source changed"):
        benchmark._check_frozen_scoring_sources(config)
