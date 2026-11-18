import pytest

from bufogen.evaluation.scorers.scene_quality import SceneQuality, scene_result


def judgment(requirements):
    return dict(
        bufo_present=True,
        similarity=0.9,
        anatomy=0.9,
        composition=0.9,
        readability=0.9,
        reason="Face and connected limbs visible.",
        prompt_checks=[
            dict(requirement=r, satisfied=True, evidence="Visible pixels show it.")
            for r in requirements
        ],
    )


@pytest.mark.parametrize("checks", [[], ["contact", "extra"], ["contact", "contact"]])
def test_exact_requirements_are_mandatory(checks):
    with pytest.raises(ValueError, match="requirements"):
        scene_result(judgment(checks), ["contact", "full body"])


@pytest.mark.parametrize("bad", [True, float("nan"), -1, 2, "1"])
def test_scene_metrics_are_strict(bad):
    data = judgment(["contact"])
    data["anatomy"] = bad
    with pytest.raises(ValueError, match="anatomy"):
        scene_result(data, ["contact"])


def test_failed_action_is_not_hidden_by_identity():
    data = judgment(["hand grips microphone"])
    data["prompt_checks"][0].update(satisfied=False, evidence="Microphone floats beside hand.")
    result = scene_result(data, ["hand grips microphone"])
    assert result["quality_failure"] is True
    assert result["constraint_coverage"] == 0
    assert "prompt_mismatch" in result["failure_modes"]
    assert result["calibration_status"] == "uncalibrated"


async def test_invalid_render_with_pixels_is_still_judged(tmp_path, monkeypatch):
    from PIL import Image

    path = tmp_path / "bad.png"
    Image.new("RGB", (16, 16)).save(path)

    async def fake_judge(self, output, requirements):
        return judgment(requirements)

    monkeypatch.setattr(SceneQuality, "_judge", fake_judge)
    result = await SceneQuality().score(
        dict(
            valid=False,
            png_path=str(path),
            prompt="contact",
            benchmark_metadata={"requirements": ["contact"]},
        )
    )
    assert result["status"] == "ok"
    assert result["anatomy"] == 0.9
    assert "invalid_render" in result["failure_modes"]
    assert path.exists()


async def test_judge_error_is_not_valid_zero(tmp_path, monkeypatch):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(path)

    async def fake_judge(self, output, requirements):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(SceneQuality, "_judge", fake_judge)
    result = await SceneQuality().score(
        dict(valid=True, png_path=str(path), benchmark_metadata={"requirements": ["contact"]})
    )
    assert result["status"] == "judge_error"
    assert result["anatomy"] is None
    assert result["quality_failure"] is None
    assert "judge_error" in result["failure_modes"]


async def test_prior_turn_pixels_are_not_judged_as_final_edit():
    result = await SceneQuality().score(
        {
            "id": "final",
            "valid": True,
            "png_path": "/no-image-needed.png",
            "benchmark_metadata": {"requirements": ["red prop"], "failed_turn_id": "prior"},
        }
    )
    assert result["status"] == "prior_turn_failure"
    assert result["anatomy"] is None


def test_visual_trace_retains_unreadable_image_failure(tmp_path):
    from bufogen.integrations.observability import visual_generation

    path = tmp_path / "bad.png"
    path.write_bytes(b"bad image")
    row = {"png_path": str(path), "id": "failed"}
    result = visual_generation("failed render", row)
    assert result["generation"] == row
    assert result["image"] is None
    assert result["image_error"]


@pytest.fixture
def scene_env(monkeypatch):
    for name in ("SCENE_JUDGE_BACKEND", "SCENE_JUDGE_MODEL", "JUDGE_MODEL", "SOL_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_scene_backend_defaults_and_model_overrides(scene_env, monkeypatch):
    default = SceneQuality()
    assert default.judge_model == "Qwen/Qwen3.8-27B"
    assert default.configuration()["judge_backend"] == "open_base"
    monkeypatch.setenv("SCENE_JUDGE_BACKEND", "closed_quality_bar")
    monkeypatch.setenv("SOL_MODEL", "sol-configured")
    assert SceneQuality().judge_model == "sol-configured"
    monkeypatch.setenv("SCENE_JUDGE_MODEL", "scene-specific")
    assert SceneQuality().judge_model == "scene-specific"
    assert SceneQuality(judge_model="explicit").judge_model == "explicit"


@pytest.mark.parametrize("options", [{"judge_backend": "unknown"}, {"judge_model": " "}])
def test_invalid_scene_configuration_fails_before_judgment(scene_env, options):
    with pytest.raises(ValueError):
        SceneQuality(**options)


async def test_scene_judge_request_shapes(scene_env, monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from PIL import Image

    from bufogen.evaluation.scorers import scene_quality
    from bufogen.models import BackendConfig

    image_path = tmp_path / "candidate.png"
    Image.new("RGB", (16, 16)).save(image_path)
    monkeypatch.setattr(scene_quality, "style_references", lambda: [image_path])
    expected = judgment(["contact"])
    responses = AsyncMock(return_value=SimpleNamespace(output_text=json.dumps(expected)))
    chat = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(expected)))],
        )
    )

    class Client:
        def __init__(self):
            self.responses = SimpleNamespace(create=responses)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=chat))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(BackendConfig, "client", lambda self: Client())
    row = dict(
        id="image",
        valid=True,
        png_path=str(image_path),
        prompt="contact",
        benchmark_metadata={"requirements": ["contact"], "turns": ["earlier request"]},
    )
    sol = SceneQuality(judge_backend="closed_quality_bar")
    result = await sol.score(row)
    assert result["status"] == "ok"
    assert result["raw_judgment"] == expected
    request = responses.call_args.kwargs
    assert request["model"] == sol.judge_model
    assert request["instructions"] == scene_quality.RUBRIC
    assert request["max_output_tokens"] == 4000
    assert request["text"] == {"format": {"type": "json_object"}}
    assert "temperature" not in request
    content = request["input"][0]["content"]
    assert [c["type"] for c in content] == [
        "input_text",
        "input_image",
        "input_text",
        "input_image",
    ]
    assert "JSON" in content[0]["text"]
    assert "prior_requests" not in content[2]["text"]
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    config = result["scorer_configuration"]
    assert config["api_mode"] == "responses"
    assert config["temperature"] is None
    assert config["max_output_tokens"] == 4000
    assert config["judge_provider"] == "openai"
    chat.assert_not_awaited()

    qwen = SceneQuality(judge_backend="open_base")
    result = await qwen.score(row)
    assert result["status"] == "ok"
    request = chat.call_args.kwargs
    assert request["model"] == "Qwen/Qwen3.8-27B"
    assert request["temperature"] == 0
    assert request["max_tokens"] == 3200
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert request["messages"][0] == {"role": "system", "content": scene_quality.RUBRIC}
    config = result["scorer_configuration"]
    assert config["api_mode"] == "chat_completions"
    assert config["max_tokens"] == [3200, 6400]
    assert config["temperature"] == 0
    assert config["judge_provider"] == "coreweave"


async def test_backend_change_invalidates_cached_score(scene_env, tmp_path, monkeypatch):
    from PIL import Image

    from bufogen.evaluation.benchmark import comparison_fingerprint

    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(path)
    calls = []

    async def judge(self, output, requirements):
        calls.append(self.judge_backend)
        return judgment(requirements)

    monkeypatch.setattr(SceneQuality, "_judge", judge)
    row = dict(
        id="generation",
        valid=True,
        png_path=str(path),
        prompt="contact",
        benchmark_metadata={"requirements": ["contact"]},
    )
    qwen = SceneQuality(judge_backend="open_base")
    sol = SceneQuality(judge_backend="closed_quality_bar")
    old = await qwen.score(row)
    row["scores"] = {"scene_quality": old}
    assert await qwen.score(row) == old
    assert (await sol.score(row))["judge_model"] == sol.judge_model
    assert calls == ["open_base", "closed_quality_bar"]
    with pytest.raises(ValueError, match="Saved score does not match"):
        await sol.score(row, saved_score=old)
    assert comparison_fingerprint(
        {"policy": {}, "source_hashes": {}, "scorer": qwen.configuration()}
    ) != comparison_fingerprint({"policy": {}, "source_hashes": {}, "scorer": sol.configuration()})


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "unsupported"},
        {"base_url": "not-a-url"},
        {"base_url": "https://secret:password@example.test/v1"},
    ],
)
def test_malformed_backend_configuration_is_rejected(scene_env, monkeypatch, changes):
    from dataclasses import replace

    from bufogen.evaluation.scorers import scene_quality

    backends = scene_quality.load_backends()
    backends["open_base"] = replace(backends["open_base"], **changes)
    monkeypatch.setattr(scene_quality, "load_backends", lambda: backends)
    with pytest.raises(ValueError, match="Scene judge backend"):
        SceneQuality()


def test_grounded_rubric_and_uncalibrated_status():
    from bufogen.evaluation.scorers.scene_quality import RUBRIC, RUBRIC_VERSION

    assert RUBRIC_VERSION == "bufo-scene-v2"
    assert "a shaft crossing an object does not mean its tip touches it" in RUBRIC
    assert SceneQuality().configuration()["calibration_status"] == "uncalibrated"


async def test_responses_bad_shape_and_transport_errors_are_abstentions(
    scene_env, monkeypatch, tmp_path
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from PIL import Image

    from bufogen.evaluation.scorers import scene_quality
    from bufogen.models import BackendConfig

    path = tmp_path / "candidate.png"
    Image.new("RGB", (16, 16)).save(path)
    monkeypatch.setattr(scene_quality, "style_references", lambda: [path])
    create = AsyncMock(
        side_effect=[
            SimpleNamespace(output_text="[]"),
            RuntimeError("credential=must-not-leak"),
        ]
    )

    class Client:
        responses = SimpleNamespace(create=create)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(BackendConfig, "client", lambda self: Client())
    scorer = SceneQuality(judge_backend="closed_quality_bar")
    row = dict(
        id="candidate",
        valid=True,
        png_path=str(path),
        prompt="contact",
        benchmark_metadata={"requirements": ["contact"]},
    )
    for _ in range(2):
        result = await scorer.score(row)
        assert result["status"] == "judge_error"
        assert result["anatomy"] is None
        assert result["quality_failure"] is None
        assert result["scorer_configuration"] == scorer.configuration()
        assert "credential" not in result["error"]
    assert create.await_count == 2
