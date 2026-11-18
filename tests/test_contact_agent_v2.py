"""Real tool validation and retries; only model/render/network boundaries are offline."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from openai.types.responses import ResponseFunctionToolCall

import test_contact_agent as original


def module():
    name = "contact_agent_v2_test"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "docs/forge-contact-pilot/contact_agent_v2.py"
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(original, "module", module)
    yield from original.offline.__wrapped__(monkeypatch)


async def test_invalid_reference_feedback_names_field_and_reason_then_null_retry_succeeds(
    offline, monkeypatch, tmp_path
):
    m, _, exporter, _, _, Renderer, backend = offline
    good = original.scene(m)
    assert good["objects"][0]["attach"] is None
    bad = deepcopy(good)
    bad["objects"][0]["attach"] = {"anchor": "grip", "to": "none", "dx": 0, "dy": 0}
    bad["objects"][0]["svg"] += "<!-- PRIVATE_INPUT_NOT_FOR_ERROR -->"
    model = original.OfflineModel(
        [
            original.call(bad, "invalid-reference"),
            original.call(good, "null-attachment"),
            original.reply(),
        ]
    )
    monkeypatch.setattr(m, "OpenAIChatCompletionsModel", lambda **kwargs: model)
    row = await m.ContactAgent(backend, Renderer(), tmp_path, client=object()).generate(
        "Hold a mug"
    )
    results = [json.loads(x["content"]) for x in row.trajectory["messages"] if x["role"] == "tool"]
    assert results[0]["valid"] is False
    assert "objects.0.attach.to" in results[0]["error"]
    assert "object.anchor" in results[0]["error"]
    assert "PRIVATE_INPUT_NOT_FOR_ERROR" not in results[0]["error"]
    assert "<!--" not in results[0]["error"]
    assert results[1]["valid"] is True and row.valid
    assert row.tool_calls == 2
    attempts = [s for s in exporter.get_finished_spans() if s.name == "execute_tool render_scene"]
    assert len(attempts) == 2
    assert "objects.0.attach.to" in attempts[0].attributes["gen_ai.tool.call.result"]


async def test_malformed_json_reports_location_without_echoing_input(
    offline, monkeypatch, tmp_path
):
    m, _, _, _, _, Renderer, backend = offline
    malformed = ResponseFunctionToolCall(
        type="function_call",
        name="render_scene",
        call_id="bad-json",
        arguments='{"PRIVATE_INPUT_NOT_FOR_ERROR":\n',
    )
    model = original.OfflineModel(
        [malformed, original.call(original.scene(m), "fixed"), original.reply()]
    )
    monkeypatch.setattr(m, "OpenAIChatCompletionsModel", lambda **kwargs: model)
    row = await m.ContactAgent(backend, Renderer(), tmp_path, client=object()).generate(
        "Hold a mug"
    )
    failure = next(
        json.loads(x["content"]) for x in row.trajectory["messages"] if x["role"] == "tool"
    )
    assert "line 2" in failure["error"] and "column 1" in failure["error"]
    assert "PRIVATE_INPUT_NOT_FOR_ERROR" not in failure["error"]
    assert row.valid


def test_feedback_redacts_secret_and_omits_validation_input(monkeypatch):
    m = module()
    secret = "private-test-value-7319"
    monkeypatch.setenv("CONTACT_TEST_API_KEY", secret)
    value = original.scene(m)
    value["objects"][0]["attach"] = {"anchor": "grip", "to": "none", secret: secret}
    with pytest.raises(Exception) as error:
        m.Scene.model_validate(value)
    outer = RuntimeError("Invalid JSON input for tool render_scene")
    outer.__context__ = error.value
    feedback = m.validation_feedback(outer)
    assert "objects.0.attach.to" in feedback
    assert secret not in feedback
    assert "input_value" not in feedback and "<rect" not in feedback


def test_unknown_failure_has_safe_bounded_fallback_even_with_cyclic_context():
    m = module()
    error = RuntimeError("PRIVATE_INPUT_NOT_FOR_ERROR" * 100)
    error.__context__ = error
    feedback = m.validation_feedback(error)
    assert "render_scene" in feedback and "arguments" in feedback
    assert "PRIVATE_INPUT_NOT_FOR_ERROR" not in feedback
    assert len(feedback) < 800
