import copy
import json
from types import SimpleNamespace

import pytest

from bufogen.evaluation.scorers.scene_quality import SceneQuality, scene_result
from bufogen.generation import BufoGeneration
from bufogen.training.common import reward
from bufogen.training.rl import (
    RLTrainer,
    resolve_sft_identity,
    validate_student_capture,
    verify_registered_continuation,
)
from bufogen.training.rl_contract import (
    build_calibration_report,
    load_environment_tasks,
    require_calibration,
    write_environment_tasks,
)


def scored_row(positive=True):
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
            "requirements": ["Hand grips pointer"],
            "png_sha256": "pixels",
            "split": "train",
        },
    )
    config = SceneQuality().configuration()
    score = scene_result(
        {
            "bufo_present": True,
            "similarity": 0.9,
            "anatomy": 0.9,
            "composition": 0.9,
            "readability": 0.9,
            "reason": "Visible hand",
            "prompt_checks": [
                {
                    "requirement": "Hand grips pointer",
                    "satisfied": positive,
                    "evidence": "Visible grip",
                }
            ],
        },
        row.benchmark_metadata["requirements"],
    )
    row.scores["scene_quality"] = score | {
        "generation_id": row.id,
        "scorer_configuration": config,
        "requirements": row.benchmark_metadata["requirements"],
    }
    review = {
        "source_generation_id": row.id,
        "prompt_id": "p",
        "reviewer_kind": "assistant",
        "reviewer_id": "codex/test",
        "reviewed_at": "2026-10-15T00:00:00+00:00",
        "evidence": "Visible grip",
        "source_ref": "https://example.test/trace",
        "png_sha256": "pixels",
        "visual_quality_pass": positive,
        "partition": "train",
        "requirements": copy.deepcopy(score["prompt_checks"]),
        "identity": 4,
        "anatomy": 4,
        "composition": 4,
        "readability": 4,
    }
    return row, review, config


def test_joint_reward_requires_visible_action_not_caption_and_ignores_svg_length():
    row, _, config = scored_row()
    assert reward(row, scorer_configuration=config) == pytest.approx(0.81)
    row.svg *= 100000
    assert reward(row, scorer_configuration=config) == pytest.approx(0.81)
    row.scores["scene_quality"]["prompt_checks"][0]["satisfied"] = False
    row.scores["scene_quality"]["constraint_coverage"] = 1  # unchecked derived value is ignored
    assert reward(row, scorer_configuration=config) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "invalid",
        "nan",
        "range",
        "bool",
        "absent",
        "foreign",
        "config",
        "requirements",
        "error",
    ],
)
def test_unusable_or_unbound_judgment_cannot_earn_reward(mutation):
    row, _, config = scored_row()
    score = row.scores["scene_quality"]
    if mutation == "missing":
        row.scores = {}
    if mutation == "invalid":
        row.valid = False
    if mutation == "nan":
        score["anatomy"] = float("nan")
    if mutation == "range":
        score["readability"] = 1.1
    if mutation == "bool":
        score["similarity"] = True
    if mutation == "absent":
        score["bufo_present"] = False
    if mutation == "foreign":
        score["generation_id"] = "other"
    if mutation == "config":
        score["scorer_configuration"] = {}
    if mutation == "requirements":
        score["requirements"] = ["other"]
    if mutation == "error":
        row.error = "generation ended early"
    assert reward(row, scorer_configuration=config) == 0


def test_calibration_separates_visual_review_from_sft_eligibility_and_exposes_false_positives():
    positive, yes, config = scored_row()
    negative, no, _ = scored_row(False)
    yes["decision"] = "reject"  # prior-turn SFT exclusion, visually good
    report = build_calibration_report(
        [positive.to_dict(), negative.to_dict()], [yes, no], scorer_configuration=config
    )
    assert report["rl_ready"] and report["agreement"] == 1
    assert not report["population_calibrated"]
    require_calibration(report, scorer_configuration=config)
    negative.scores["scene_quality"]["prompt_checks"][0]["satisfied"] = True
    failed = build_calibration_report(
        [positive.to_dict(), negative.to_dict()], [yes, no], scorer_configuration=config
    )
    assert failed["false_positives"] == 1 and not failed["rl_ready"]
    failed["rl_ready"] = True
    with pytest.raises(ValueError, match="blocked"):
        require_calibration(failed, scorer_configuration=config)


def test_calibration_requires_negative_and_rejects_foreign_scores_or_images():
    row, review, config = scored_row()
    report = build_calibration_report([row.to_dict()], [review], scorer_configuration=config)
    assert not report["rl_ready"]
    review["png_sha256"] = "other-image"
    with pytest.raises(ValueError, match="image hash"):
        build_calibration_report([row.to_dict()], [review], scorer_configuration=config)
    review["png_sha256"] = "pixels"
    row.scores["scene_quality"]["generation_id"] = "other"
    with pytest.raises(ValueError, match="bound"):
        build_calibration_report([row.to_dict()], [review], scorer_configuration=config)


def test_environment_artifact_excludes_eval_and_edits_but_training_loader_rejects_them(tmp_path):
    rows = [
        {
            "id": "p",
            "prompt": "Bufo grips pointer",
            "family": "presenter",
            "split": "train",
            "requirements": ["Hand grips pointer"],
        },
        {
            "id": "edit",
            "prompt": "Move pointer",
            "family": "presenter",
            "split": "train",
            "requirements": ["Pointer left"],
            "turns": ["Bufo grips pointer"],
        },
        {
            "id": "eval",
            "prompt": "Bufo drums",
            "family": "drums",
            "split": "heldout",
            "requirements": ["Hand grips drumstick"],
        },
    ]
    metadata = write_environment_tasks(rows, tmp_path)
    assert metadata["count"] == 1 and metadata["excluded_conversational_train"] == 1
    assert len(load_environment_tasks(tmp_path / "train.jsonl")) == 1
    (tmp_path / "train.jsonl").write_text(json.dumps(rows[-1]))
    with pytest.raises(ValueError, match="only single-turn train"):
        load_environment_tasks(tmp_path / "train.jsonl")
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"prompt_id": "legacy", "prompt": "Bufo", "split": "train"})
    )
    with pytest.raises(ValueError, match="family"):
        load_environment_tasks(tmp_path / "train.jsonl")


def captured(model="student", history=None):
    from openai.types.chat import ChatCompletion

    completion = ChatCompletion(
        id="x",
        created=0,
        model=model,
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Here is Bufo"},
            }
        ],
    )
    exchange = SimpleNamespace(
        model=model,
        request={"model": model, "messages": history or [{"role": "user", "content": "Draw Bufo"}]},
        response=completion,
    )
    return SimpleNamespace(
        exchanges=SimpleNamespace(
            chat_completions=[exchange], responses=[], completions=[], messages=[]
        )
    )


def test_capture_rejects_teacher_calls_and_orphan_tool_results():
    args = {
        "model": "student",
        "provider": "coreweave",
        "base_url": "https://api.inference.wandb.ai/v1",
    }
    assert validate_student_capture(captured(), **args) == 1
    with pytest.raises(ValueError, match="Nonstudent model"):
        validate_student_capture(captured("teacher"), **args)
    with pytest.raises(ValueError, match="originating"):
        validate_student_capture(
            captured(history=[{"role": "tool", "tool_call_id": "invented", "content": "PNG"}]),
            **args,
        )
    with pytest.raises(ValueError, match="provider"):
        validate_student_capture(captured(), **(args | {"base_url": "https://other.test/v1"}))
    trajectory = captured()
    trajectory.exchanges.responses.append(object())
    with pytest.raises(ValueError, match="protocol"):
        validate_student_capture(trajectory, **args)


def sft_fixture():
    run = SimpleNamespace(id="actual-run", entity="e", project="p", config={})
    artifact = SimpleNamespace(
        id="immutable-id",
        digest="digest",
        name="storage:v3",
        entity="e",
        project="p",
        metadata={"wandb.base_model": "gemma"},
        logged_by=lambda: run,
    )
    api = SimpleNamespace(artifact=lambda *a, **k: artifact)
    job = {
        "id": "job",
        "status": "deployed",
        "base_model": "gemma",
        "model_artifact_ref": "e/p/storage:latest",
        "wandb_run_id": "actual-run",
    }
    return api, job, artifact


async def test_sft_continuation_binds_real_provenance_and_service_checkpoint():
    api, job, artifact = sft_fixture()
    identity = resolve_sft_identity(
        initial_checkpoint="e/p/storage:latest",
        forge_job=job,
        api=api,
        entity="e",
        project="p",
        base_model="gemma",
    )
    assert identity["initial_checkpoint"].endswith("storage:v3")

    async def step():
        return 2

    model = SimpleNamespace(
        get_step=step,
        run_id="actual-run",
        entity="e",
        project="p",
        run_name="storage",
        base_model="gemma",
    )
    assert await verify_registered_continuation(model, identity, api) == 2
    artifact.digest = "another-checkpoint"
    with pytest.raises(ValueError, match="differs"):
        await verify_registered_continuation(model, identity, api)

    async def zero():
        return 0

    model.get_step = zero
    with pytest.raises(ValueError, match="step zero"):
        await verify_registered_continuation(model, identity, api)


def test_sft_initialization_cannot_be_a_fork_or_incomplete_job():
    api, job, artifact = sft_fixture()
    job["status"] = "training"
    with pytest.raises(ValueError, match="deployed"):
        resolve_sft_identity(
            initial_checkpoint="e/p/storage:v3",
            forge_job=job,
            api=api,
            entity="e",
            project="p",
            base_model="gemma",
        )
    job["status"] = "deployed"
    artifact.logged_by = lambda: SimpleNamespace(id="fork-run", entity="e", project="p")
    with pytest.raises(ValueError, match="provenance"):
        resolve_sft_identity(
            initial_checkpoint="e/p/storage:v3",
            forge_job=job,
            api=api,
            entity="e",
            project="p",
            base_model="gemma",
        )


async def test_unsupported_initial_checkpoint_fails_before_dataset_or_remote_registration(
    monkeypatch,
):
    trainer = RLTrainer(
        "never-download",
        name="test",
        base_model="google/gemma-4-26B-A4B-it",
        initial_checkpoint="unknown",
    )
    monkeypatch.setattr(
        trainer, "dataset", lambda: pytest.fail("must fail before remote dataset read")
    )
    with pytest.raises(ValueError, match="actual deployed Forge"):
        await trainer.train(steps=1)


async def test_installed_art_captures_only_student_scope_with_real_http_transport():
    art = pytest.importorskip("art")
    import httpx
    from openai import AsyncOpenAI

    def respond(request):
        model = json.loads(request.content)["model"]
        return httpx.Response(
            200,
            json={
                "id": "test",
                "created": 0,
                "model": model,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hello"},
                    }
                ],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    async with AsyncOpenAI(
        api_key="test", base_url="https://api.inference.wandb.ai/v1", http_client=http
    ) as client:
        with art.Trajectory() as trajectory:
            await client.chat.completions.create(
                model="student", messages=[{"role": "user", "content": "Draw Bufo"}]
            )
        await client.chat.completions.create(
            model="judge", messages=[{"role": "user", "content": "Judge image"}]
        )
    assert (
        validate_student_capture(
            trajectory,
            model="student",
            provider="coreweave",
            base_url="https://api.inference.wandb.ai/v1",
        )
        == 1
    )
    # Current ART requires a durable run_name; constructing locally never registers it.
    model = art.TrainableModel(
        name="display", run_name="storage", project="p", entity="e", base_model="gemma"
    )
    assert model.run_name == "storage"


async def test_current_plan_tool_schema_and_real_result_survive_student_capture():
    """Adding the shared layout tool must not reject otherwise authentic SDK exchanges."""
    import art
    import httpx
    from agents import function_tool
    from agents.tool_context import ToolContext
    from openai import AsyncOpenAI

    from bufogen.agents.planning import plan_scene
    from bufogen.curation.trajectories import tool_schemas

    tool = function_tool(plan_scene)
    tools = tool_schemas([tool])
    arguments = json.dumps(
        {
            "main_bufo": {"x": 100, "y": 80, "scale": 2},
            "coordinate_frame": "bufo_local",
            "contact_anchors": [],
            "paint_order": ["background", "bufo"],
            "framing": "fullbody",
        }
    )
    call = {
        "id": "actual-plan",
        "type": "function",
        "function": {"name": "plan_scene", "arguments": arguments},
    }
    history = [{"role": "user", "content": "Draw Bufo standing"}]

    def respond(request):
        body = json.loads(request.content)
        first = len(body["messages"]) == 1
        message = (
            {"role": "assistant", "content": None, "tool_calls": [call]}
            if first
            else {"role": "assistant", "content": "The layout is ready."}
        )
        return httpx.Response(
            200,
            json={
                "id": "capture-test",
                "created": 0,
                "model": "student",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if first else "stop",
                        "message": message,
                    }
                ],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    async with AsyncOpenAI(
        api_key="test", base_url="https://api.inference.wandb.ai/v1", http_client=http
    ) as client:
        with art.Trajectory() as trajectory:
            response = await client.chat.completions.create(
                model="student", messages=history, tools=tools
            )
            sampled = response.choices[0].message
            result = await tool.on_invoke_tool(
                ToolContext(
                    context=None,
                    tool_name="plan_scene",
                    tool_call_id="actual-plan",
                    tool_arguments=arguments,
                ),
                sampled.tool_calls[0].function.arguments,
            )
            history.extend(
                [
                    sampled.model_dump(exclude_none=True),
                    {"role": "tool", "tool_call_id": "actual-plan", "content": result},
                ]
            )
            await client.chat.completions.create(model="student", messages=history, tools=tools)
    assert json.loads(result)["bufo_canvas_bounds"] == [100, 80, 356, 432]
    assert (
        validate_student_capture(
            trajectory,
            model="student",
            provider="coreweave",
            base_url="https://api.inference.wandb.ai/v1",
        )
        == 2
    )
    # Accept only the explicitly supported extra tool, never an arbitrary schema.
    trajectory.exchanges.chat_completions[0].request["tools"].append(
        {"type": "function", "function": {"name": "arbitrary_shell"}}
    )
    with pytest.raises(ValueError, match="Unexpected tool schema"):
        validate_student_capture(
            trajectory,
            model="student",
            provider="coreweave",
            base_url="https://api.inference.wandb.ai/v1",
        )


def _resume_fixture(step=6, *, at_step="rl6", ancestor="sft", recorded="rl6"):
    artifacts = {
        "e/p/storage:step6": SimpleNamespace(id="id-rl6", digest="d-rl6"),
        "e/p/storage:step1": SimpleNamespace(id="id-sft", digest="d-sft"),
        "e/p/storage:v6": SimpleNamespace(id="id-rl6", digest="d-rl6"),
    }
    if at_step != "rl6":
        artifacts["e/p/storage:step6"] = SimpleNamespace(id="id-x", digest="d-x")
    if ancestor != "sft":
        artifacts["e/p/storage:step1"] = SimpleNamespace(id="id-y", digest="d-y")
    if recorded != "rl6":
        artifacts["e/p/storage:v6"] = SimpleNamespace(id="id-z", digest="d-z")
    api = SimpleNamespace(artifact=lambda ref, type=None: artifacts[ref])
    identity = {
        "run_id": "run", "entity": "e", "project": "p", "storage_name": "storage",
        "base_model": "gemma", "artifact_id": "id-sft", "artifact_digest": "d-sft",
    }  # fmt: skip

    async def get_step():
        return step

    model = SimpleNamespace(
        get_step=get_step, run_id="run", entity="e", project="p", run_name="storage",
        base_model="gemma",
    )  # fmt: skip
    return model, identity, api


async def test_resume_start_step_matches_recorded_rl_checkpoint():
    model, identity, api = _resume_fixture()
    got = await verify_registered_continuation(
        model,
        identity,
        api,
        expect_start_step=6,
        recorded_artifact="wandb-artifact:///e/p/storage:v6",
    )
    assert got == 6


@pytest.mark.parametrize(
    "kwargs,expect,match",
    [
        ({"step": 7}, 6, "expected 6"),
        ({"at_step": "other"}, 6, "recorded artifact"),
        ({"recorded": "other"}, 6, "recorded artifact"),
        ({"ancestor": "other"}, 6, "ancestor"),
    ],
)
async def test_resume_rejects_mismatches(kwargs, expect, match):
    model, identity, api = _resume_fixture(**kwargs)
    with pytest.raises(ValueError, match=match):
        await verify_registered_continuation(
            model, identity, api, expect_start_step=expect, recorded_artifact="e/p/storage:v6"
        )


async def test_resume_past_the_sft_step_still_refuses_without_the_flag():
    model, identity, api = _resume_fixture()
    with pytest.raises(ValueError, match="differs from frozen SFT"):
        await verify_registered_continuation(model, identity, api)


def test_recorded_rl_artifact_is_read_from_checkpoints_json(tmp_path, monkeypatch):
    from bufogen.training import rl as train_rl

    folder = tmp_path / "runs/distill/stage-b"
    folder.mkdir(parents=True)
    pre = "wandb-artifact:///e/p/storage:"
    (folder / "checkpoints.json").write_text(
        json.dumps(
            [
                {"arm_name": "sft", "artifact": pre + "v1"},
                {"arm_name": "rl", "artifact": pre + "v6", "rl_config": {}},
            ]
        )
    )
    monkeypatch.setattr(train_rl, "ROOT", tmp_path)
    assert train_rl.recorded_rl_artifact("storage", "e", "p") == pre + "v6"
