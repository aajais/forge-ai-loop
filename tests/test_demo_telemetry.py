import json
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from bufogen.generation import BufoGeneration, Manifest


def generation():
    return BufoGeneration("p", "Draw", "student", "model", "conversation", "", False, 1)


def test_native_root_identity_roundtrips_with_execution_metadata(tmp_path, monkeypatch):
    from bufogen.integrations.telemetry import current_agent_root, generation_turn

    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    try:
        with generation_turn(
            "Bufo", "model", "conversation", "Draw", attributes={"bufo.origin": "benchmark"}
        ):
            row = generation()
            for key, value in current_agent_root().items():
                setattr(row, key, value)
            row.execution_metadata = {"max_tokens": 20000}
        (root,) = exporter.get_finished_spans()
        manifest = Manifest(tmp_path / "rows.jsonl")
        manifest.append(row)
        (restored,) = manifest.read()
        assert restored.native_trace_id == format(root.context.trace_id, "032x")
        assert restored.native_span_id == format(root.context.span_id, "016x")
        assert restored.execution_metadata == {"max_tokens": 20000}
        assert root.attributes["bufo.origin"] == "benchmark"
        assert current_agent_root() == {"native_trace_id": None, "native_span_id": None}
    finally:
        provider.shutdown()


class FeedbackStore:
    def __init__(self):
        self.entries = []

    def __iter__(self):
        return iter(self.entries)

    def add(self, feedback_type, payload, creator):
        entry = SimpleNamespace(
            id=f"f{len(self.entries)}", feedback_type=feedback_type, payload=payload
        )
        self.entries.append(entry)
        return entry.id


async def test_saved_source_feedback_uses_original_root_and_reconciles_lost_receipt(tmp_path):
    from bufogen.evaluation.benchmark import _publish_source_feedback

    row = generation()
    row.native_span_id, row.native_trace_id = "1234567890abcdef", "1" * 32
    row.scores = {"astra_judge": {"generation_id": row.id, "status": "ok", "strict_pass": False}}
    store = FeedbackStore()
    client = SimpleNamespace(
        get_agent_span_feedback=lambda span: store if span == row.native_span_id else None,
        flush=lambda: None,
    )
    state = {}
    path = tmp_path / "publication.json"
    await _publish_source_feedback(client, [row], {"astra_judge": object()}, state, path)
    assert len(store.entries) == 1
    assert store.entries[0].payload["score"] == row.scores["astra_judge"]
    assert store.entries[0].payload["native_trace_id"] == "1" * 32
    receipt = json.loads(path.read_text())["source_feedback"][row.id]["astra_judge"]
    assert receipt["native"]["status"] == "published"
    assert receipt["native"]["score_columns"] == "unsupported_custom_feedback"
    assert receipt["classic"]["status"] == "missing_source_call"
    await _publish_source_feedback(client, [row], {"astra_judge": object()}, state, path)
    # Remote reconciliation protects the success-before-local-checkpoint crash window.
    await _publish_source_feedback(client, [row], {"astra_judge": object()}, {}, path)
    assert len(store.entries) == 1


async def test_feedback_failure_is_recorded_and_retryable_without_claiming_success(tmp_path):
    from bufogen.evaluation.benchmark import _publish_source_feedback

    row = generation()
    row.native_span_id, row.native_trace_id = "1234567890abcdef", "1" * 32
    row.scores = {
        "astra_judge": {"generation_id": row.id, "status": "judge_error", "strict_pass": False}
    }
    store = FeedbackStore()
    failures = [True, False]

    def feedback(span):
        if failures.pop(0):
            raise ConnectionError("offline")
        return store

    client = SimpleNamespace(get_agent_span_feedback=feedback, flush=lambda: None)
    state, path = {}, tmp_path / "publication.json"
    await _publish_source_feedback(client, [row], {"astra_judge": object()}, state, path)
    assert state["source_feedback"][row.id]["astra_judge"]["native"]["status"] == "error"
    await _publish_source_feedback(client, [row], {"astra_judge": object()}, state, path)
    assert state["source_feedback"][row.id]["astra_judge"]["native"]["status"] == "published"
    assert store.entries[0].payload["score"]["status"] == "judge_error"


async def test_historical_missing_root_and_unsupported_sdk_are_explicit(tmp_path):
    from bufogen.evaluation.benchmark import _publish_source_feedback

    row = generation()
    row.scores = {"scene_quality": {"generation_id": row.id, "status": "ok"}}
    state = {}
    path = tmp_path / "publication.json"
    await _publish_source_feedback(
        SimpleNamespace(), [row], {"scene_quality": object()}, state, path
    )
    assert (
        state["source_feedback"][row.id]["scene_quality"]["native"]["status"]
        == "missing_source_root"
    )
    row.native_span_id, row.native_trace_id = "1234567890abcdef", "1" * 32
    await _publish_source_feedback(
        SimpleNamespace(), [row], {"scene_quality": object()}, state, path
    )
    assert (
        state["source_feedback"][row.id]["scene_quality"]["native"]["status"] == "unsupported_sdk"
    )


async def test_source_feedback_rejects_wrong_generation_before_remote_mutation(tmp_path):
    from bufogen.evaluation.benchmark import _publish_source_feedback

    row = generation()
    row.scores = {"astra_judge": {"generation_id": "different", "strict_pass": True}}
    with pytest.raises(ValueError, match="generation"):
        await _publish_source_feedback(
            SimpleNamespace(), [row], {"astra_judge": object()}, {}, tmp_path / "p.json"
        )


async def test_application_renderer_span_identifies_internal_media_publishing(monkeypatch):
    import weave

    from bufogen.integrations.telemetry import generation_turn, publish_render_png
    from bufogen.rendering.renderer import RenderResult

    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        weave,
        "publish",
        lambda value: SimpleNamespace(uri=lambda: "weave:///test/project/object/Content:digest"),
    )

    class Renderer:
        async def render(self, svg):
            return RenderResult(b"image", True, None)

    try:
        with generation_turn("Bufo", "model", "conversation", "Draw") as turn:
            await publish_render_png(turn, Renderer(), "<svg/>")
        tool = next(s for s in exporter.get_finished_spans() if s.name == "execute_tool render_png")
        assert tool.attributes["bufo.tool_origin"] == "application"
        assert tool.attributes["bufo.model_requested"] is False
        assert "render_svg" in tool.attributes["gen_ai.tool.description"]
    finally:
        provider.shutdown()


async def test_classic_feedback_reuses_saved_score_and_preserves_success_on_retry(tmp_path):
    from bufogen.evaluation.benchmark import _publish_source_feedback

    row = generation()
    row.weave_call_id = "classic-source"
    row.scores = {
        "scene_quality": {"generation_id": row.id, "status": "ok", "quality_failure": True}
    }
    store = FeedbackStore()

    class Call:
        feedback = store

        async def apply_scorer(self, scorer, additional_scorer_kwargs):
            store.add(
                "wandb.runnable.SceneQuality",
                {"output": additional_scorer_kwargs["saved_score"]},
                "scorer",
            )

    client = SimpleNamespace(
        get_call=lambda identity: Call() if identity == "classic-source" else None,
        flush=lambda: None,
    )
    path, state = tmp_path / "p.json", {}
    await _publish_source_feedback(client, [row], {"scene_quality": object()}, state, path)
    await _publish_source_feedback(client, [row], {"scene_quality": object()}, {}, path)
    assert len(store.entries) == 1
    assert store.entries[0].payload["output"] == row.scores["scene_quality"]
    assert state["source_feedback"][row.id]["scene_quality"]["classic"]["status"] == "published"


async def test_real_sdk_publication_names_evaluation_and_aggregates_saved_false_scores(
    tmp_path, monkeypatch
):
    import weave

    from bufogen.evaluation.benchmark import _publish_arm_evaluation

    class SavedVerdict(weave.Scorer):
        @weave.op()
        def score(self, output: dict) -> dict:
            return output["generation"]["scores"]["astra_judge"]

    rows = [generation(), generation()]
    for row, passed in zip(rows, [True, False], strict=True):
        row.scores = {
            "astra_judge": {"generation_id": row.id, "status": "ok", "strict_pass": passed}
        }
    published = []
    monkeypatch.setattr(
        weave,
        "publish",
        lambda obj: (
            published.append(obj)
            or SimpleNamespace(uri=lambda: "weave:///test/project/object/eval:digest")
        ),
    )
    state = {"evaluations": {}}
    await _publish_arm_evaluation(
        state,
        tmp_path / "p.json",
        "student",
        class_name="CompleteAstraEvaluation",
        dataset_name="run-student",
        evaluation_name="Bufo ft-Qwen validation",
        dataset_rows=[{"prompt": r.prompt, "generation": r.to_dict()} for r in rows],
        subset=rows,
        scorers=[SavedVerdict()],
        metadata={"model": "exact-model"},
        score_keys={"SavedVerdict": "astra_judge"},
    )
    assert state["evaluations"]["student"]["summary"]["SavedVerdict"]["strict_pass"] == {
        "true_count": 1,
        "true_fraction": 0.5,
    }
    assert published[0].evaluation_name == "Bufo ft-Qwen validation"


def test_forge_checkpoint_link_uses_lora_artifact_type_and_immutable_version():
    from bufogen.evaluation.benchmark import checkpoint_artifact_url

    assert (
        checkpoint_artifact_url("wandb-artifact:///team/project/qwen-sft:v3")
        == "https://forge.coreweave.com/wandb/team/project/artifacts/lora/qwen-sft/v3"
    )
    assert checkpoint_artifact_url("Qwen/Qwen3.8-27B") is None
    assert checkpoint_artifact_url("wandb-artifact:///team/project/qwen-sft:latest") is None


def test_saved_scorer_replays_post_generation_evidence_without_inference():
    from bufogen.integrations.observability import SavedGenerationScore

    scorer = SavedGenerationScore(name="SavedAstraReferenceJudge")
    # Original visual generation predates reference selection and score calculation.
    output = {"generation": {"id": "source-row", "scores": {}, "benchmark_metadata": {}}}
    saved = {
        "generation_id": "source-row",
        "strict_pass": False,
        "reference": {"png_sha256": "selected-later"},
        "status": "ok",
    }
    assert scorer.score(output, saved_score=saved) == saved
    with pytest.raises(ValueError, match="generation"):
        scorer.score(output, saved_score={**saved, "generation_id": "other-row"})
    with pytest.raises(ValueError, match="saved"):
        scorer.score(output)
