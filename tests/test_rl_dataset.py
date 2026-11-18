import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from bufogen.curation.rl_dataset import finalize_dataset, reference_path


async def test_builder_records_missing_source_and_reuses_visual_captions(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    import httpx

    from bufogen.curation import rl_dataset

    images = {}
    for name, reverse in (("frog-a", False), ("frog-b", True)):
        image = Image.new("RGB", (16, 16))
        image.putdata(
            [((15 - x if reverse else x) * 16, 100, 50) for y in range(16) for x in range(16)]
        )
        data = io.BytesIO()
        image.save(data, format="PNG")
        images[name] = data.getvalue()

    def respond(request):
        if request.url.path.endswith(".json"):
            return httpx.Response(
                200,
                json={"bufos": [dict(id=n, fileType="png", tags=[]) for n in ["missing", *images]]},
            )
        name = request.url.path.rsplit("/", 1)[-1].removesuffix(".png")
        return httpx.Response(200, content=images[name]) if name in images else httpx.Response(404)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        rl_dataset.httpx,
        "AsyncClient",
        lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw),
    )

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(
        rl_dataset, "replace", lambda *a, **kw: SimpleNamespace(model="fake", client=Client)
    )
    describe = AsyncMock(return_value={"usable": True, "description": "A seated green frog."})
    monkeypatch.setattr(rl_dataset, "vision_json", describe)
    result = await rl_dataset.build_rl_dataset(tmp_path, limit=2, concurrency=4)
    assert result["examples"] == 2
    assert "404" in json.loads((tmp_path / "excluded.json").read_text())[0]["excluded"]
    assert describe.await_count == 2
    assert (tmp_path / "review.html").is_file()
    await rl_dataset.build_rl_dataset(tmp_path, limit=2, concurrency=4)
    assert describe.await_count == 2


def test_dataset_groups_duplicates_and_keeps_images_portable(tmp_path):
    rows = []
    for i in range(10):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i * 20, 100, 50)).save(path)
        rows.append(
            dict(
                prompt_id=str(i),
                prompt=f"Frog {i}",
                reference_png=path.name,
                pixel_sha256=str(i),
                perceptual_hash=f"{(1 << (i * 6)) - 1:016x}",
            )
        )
    rows.append({**rows[0], "prompt_id": "duplicate"})
    summary = finalize_dataset(rows, tmp_path, heldout_fraction=0.2)
    train = [json.loads(s) for s in (tmp_path / "train.jsonl").read_text().splitlines()]
    heldout = [json.loads(s) for s in (tmp_path / "heldout.jsonl").read_text().splitlines()]
    assert summary["examples"] == 10
    assert len(train) == 8 and len(heldout) == 2
    assert not {x["split_group"] for x in train} & {x["split_group"] for x in heldout}
    assert all(reference_path(tmp_path, r).is_file() for r in train + heldout)
    before = (tmp_path / "train.jsonl").read_text()
    finalize_dataset(list(reversed(rows)), tmp_path, heldout_fraction=0.2)
    assert (tmp_path / "train.jsonl").read_text() == before


def test_dataset_near_duplicates_stay_in_same_split(tmp_path):
    rows = [
        dict(
            prompt_id=str(i),
            prompt="frog",
            pixel_sha256=str(i),
            reference_png=f"{i}.png",
            perceptual_hash=h,
        )
        for i, h in enumerate(["0000000000000000", "0000000000000001", "ffffffffffffffff"])
    ]
    for row in rows:
        Image.new("RGB", (8, 8)).save(tmp_path / row["reference_png"])
    finalize_dataset(rows, tmp_path, heldout_fraction=0.2)
    splits = [
        [json.loads(s) for s in (tmp_path / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "heldout")
    ]
    assert any({"0", "1"} <= {r["prompt_id"] for r in split} for split in splits)


def test_reference_paths_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="reference"):
        reference_path(tmp_path, {"reference_png": "../outside.png"})
    with pytest.raises(ValueError, match="reference"):
        reference_path(tmp_path, {"reference_png": "missing.png"})
    assert reference_path(tmp_path, {"prompt": "legacy prompt"}) is None


async def test_judge_receives_target_image_without_generic_references(tmp_path, monkeypatch):
    from bufogen.evaluation.scorers import vlm_judge

    target, candidate = tmp_path / "target.png", tmp_path / "candidate.png"
    Image.new("RGB", (8, 8), "green").save(target)
    Image.new("RGB", (8, 8), "blue").save(candidate)
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            dict(green_bufo=0.8, blue_accent=1, theme_match=0.7, visual_quality=0.8)
                        )
                    )
                )
            ]
        )

    class Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(
        vlm_judge, "replace", lambda *a, **kw: SimpleNamespace(model="judge", client=Client)
    )
    result = await vlm_judge.VLMJudge(reference_dir=str(tmp_path)).score(
        dict(valid=True, png_path=str(candidate), reference_png=str(target), prompt="A green frog")
    )
    images = [x for x in calls[0]["messages"][1]["content"] if x["type"] == "image_url"]
    assert images == [vlm_judge.image_content(target), vlm_judge.image_content(candidate)]
    assert result["reference_mode"] == "target"


@pytest.mark.parametrize("failure_stage", [None, "promotion", "metrics", "artifact"])
async def test_serverless_rollouts_use_calibrated_tasks_and_keep_judge_outside_capture(
    tmp_path, monkeypatch, failure_stage
):
    """Exercise the current training boundary without a paid training job."""
    import sys
    from unittest.mock import AsyncMock, Mock

    from test_rl_contract import captured, scored_row

    from bufogen.evaluation.scorers import scene_quality
    from bufogen.integrations import telemetry
    from bufogen.training import rl as train_rl
    from bufogen.training.rl_contract import build_calibration_report, write_environment_tasks

    positive, yes, config = scored_row()
    negative, no, _ = scored_row(False)
    calibration = build_calibration_report(
        [positive.to_dict(), negative.to_dict()], [yes, no], scorer_configuration=config
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    write_environment_tasks(
        [
            {
                "id": "p",
                "prompt": positive.prompt,
                "family": "presenter",
                "split": "train",
                "requirements": positive.benchmark_metadata["requirements"],
            }
        ],
        tmp_path,
    )
    registry = Mock()
    registry.download.return_value = tmp_path
    if failure_stage == "promotion":
        from wandb.errors import CommError

        registry.link_checkpoint.side_effect = CommError("Registry organization unavailable")
    entity, project = train_rl.project_path().split("/")
    returned_ref = f"{entity}/{project}/test:step1"
    immutable_ref = f"wandb-artifact:///{entity}/{project}/test:v1"
    model = SimpleNamespace(
        register=AsyncMock(),
        get_step=AsyncMock(return_value=0),
        log=AsyncMock(),
        run_id="test",
        run_name="test",
        entity=entity,
        project=project,
        base_model="google/gemma-4-26B-A4B-it",
        id="actual-model-id",
        openai_client=Mock(
            return_value=SimpleNamespace(
                with_options=Mock(
                    return_value=SimpleNamespace(
                        close=AsyncMock(), base_url="https://api.inference.wandb.ai/v1"
                    )
                )
            )
        ),
    )
    if failure_stage == "metrics":

        async def log(*args, **kwargs):
            if "metrics" in kwargs:
                raise RuntimeError("Metrics upload unavailable")

        model.log.side_effect = log
    backend = SimpleNamespace(
        _client=SimpleNamespace(
            training_jobs=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(id="server-job-1"))
            )
        ),
        close=AsyncMock(),
        train=AsyncMock(
            return_value=SimpleNamespace(artifact_name=returned_ref, metrics={}, step=1)
        ),
    )

    async def submit_training(actual_model, groups, **kwargs):
        await backend._client.training_jobs.create(
            model_id=actual_model.id, trajectory_groups=groups
        )
        return SimpleNamespace(artifact_name=returned_ref, metrics={}, step=1)

    backend.train.side_effect = submit_training
    capture_active = False
    base = "google/gemma-4-26B-A4B-it"

    class Trajectory:
        def __enter__(self):
            nonlocal capture_active
            capture_active = True
            self.exchanges = captured(base).exchanges
            self.metrics = {}
            return self

        def __exit__(self, *args):
            nonlocal capture_active
            capture_active = False

    monkeypatch.setitem(
        sys.modules,
        "art",
        SimpleNamespace(
            TrainableModel=lambda **kw: model,
            Trajectory=Trajectory,
            TrajectoryGroup=lambda rows: rows,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "art.serverless", SimpleNamespace(ServerlessBackend=lambda **kw: backend)
    )
    monkeypatch.setattr(telemetry, "init_weave", lambda: None)
    monkeypatch.setattr(train_rl, "required", lambda name: "test-key")
    monkeypatch.setattr(train_rl, "version", lambda name: "0.5.20")
    # This test isolates orchestration with synthetic Trajectory objects;
    # actual token evidence and upstream tokenization are exercised separately.
    token_validation = Mock(return_value={"histories": 1})
    monkeypatch.setattr(train_rl, "validate_training_tokens", token_validation)
    monkeypatch.setattr(train_rl, "ROOT", tmp_path)
    monkeypatch.setattr(train_rl, "Renderer", lambda: SimpleNamespace(close=AsyncMock()))
    import wandb

    artifact = SimpleNamespace(
        name="test:step1",
        version="v1",
        id="actual-checkpoint",
        digest="actual-bytes",
        entity=entity,
        project=project,
        metadata={"base_model": base},
        logged_by=lambda: SimpleNamespace(id="test", entity=entity, project=project),
    )
    api = Mock()
    api.artifact.return_value = artifact
    if failure_stage == "artifact":
        artifact.logged_by = lambda: SimpleNamespace(id="foreign", entity=entity, project=project)
    monkeypatch.setattr(wandb, "Api", lambda: api)
    generated, judged = [], []

    def agent_factory(*args, **kwargs):
        assert kwargs["web_search_enabled"] is False
        assert (
            kwargs["benchmark_metadata"]["requirements"]
            == positive.benchmark_metadata["requirements"]
        )

        async def generate(prompt, **generate_kwargs):
            assert capture_active
            generated.append(prompt)
            row, _, _ = scored_row()
            row.model = base
            return row

        return SimpleNamespace(generate=generate)

    async def score(row, *, rescore):
        assert not capture_active and rescore
        assert token_validation.call_count == len(judged) + 1
        judged.append(row["id"])
        return row["scores"]["scene_quality"]

    monkeypatch.setattr(train_rl, "BufoAgent", agent_factory)
    monkeypatch.setattr(
        scene_quality,
        "SceneQuality",
        lambda: SimpleNamespace(configuration=lambda: config, score=score),
    )
    trainer = train_rl.RLTrainer(
        "dataset:v0", name="test", base_model=base, calibration_path=calibration_path
    )
    trainer.registry = registry
    if failure_stage in {"metrics", "artifact"}:
        with pytest.raises(RuntimeError if failure_stage == "metrics" else ValueError):
            await trainer.train(steps=1)
        state = json.loads((tmp_path / "runs/test/rl-training.json").read_text())
        assert state["status"] == "post_training_failed"
        assert state["training_status"] == "completed"
        assert state["actual_completed_updates"] == 1
        assert state["completed_step"] == 1 and state["returned_checkpoint"] == returned_ref
        assert backend.train.await_count == 1
        registry.link_checkpoint.assert_not_called()
        return
    assert await trainer.train(steps=1) == immutable_ref
    inference_options = model.openai_client().with_options.call_args.kwargs
    assert inference_options["base_url"] == "https://api.inference.wandb.ai/v1"
    assert inference_options["max_retries"] == 0
    assert len(generated) == len(judged) == 4
    groups = backend.train.call_args.args[1]
    assert all(trajectory.reward > 0 for trajectory in groups[0])
    metric_logs = [call.kwargs for call in model.log.await_args_list if "metrics" in call.kwargs]
    assert len(metric_logs) == 1
    assert metric_logs[0]["split"] == "train"
    assert metric_logs[0]["step"] == 1
    state = json.loads((tmp_path / "runs/test/rl-training.json").read_text())
    assert state["arm"] == "open_base_rl" and state["initial_checkpoint"] is None
    assert state["status"] == "completed" and state["training_status"] == "completed"
    assert state["actual_completed_updates"] == 1
    assert state["checkpoint_artifact"] == immutable_ref
    assert state["checkpoint_artifact_id"] == "actual-checkpoint"
    assert state["checkpoint_artifact_digest"] == "actual-bytes"
    assert state["promotion_status"] == ("failed" if failure_stage else "completed")
    if failure_stage:
        assert state["promotion_failure"] == {"error_type": "CommError"}
    else:
        assert "promotion_failure" not in state
    registry.link_checkpoint.assert_called_once_with(
        immutable_ref, name="test", sources=["dataset:v0"]
    )
    assert state["training_job"] == {
        "model_id": "actual-model-id",
        "from_step": 0,
        "status": "created",
        "id": "server-job-1",
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(dict(prompt="leaked", split="heldout")))
    with pytest.raises(ValueError, match="only single-turn train"):
        await trainer.train(steps=1)
    assert backend.train.await_count == 1
