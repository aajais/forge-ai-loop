import importlib.util
import json
from pathlib import Path

import pytest

import bufogen.agents.agent as agent
import bufogen.evaluation.benchmark as benchmark
from bufogen.curation.trajectories import export_training_example
from bufogen.generation import BufoGeneration


def module():
    path = Path(__file__).resolve().parents[1] / "docs/forge-iteration2/plan_continuation.py"
    spec = importlib.util.spec_from_file_location("plan_continuation_test", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def example():
    call = {
        "id": "actual-teacher-call",
        "type": "function",
        "function": {
            "name": "plan_scene",
            "arguments": json.dumps(
                {"contact_anchors": [{"name": "hand_mug_handle", "x": 132, "y": 118}]}
            ),
        },
    }
    return {
        "source_model": "gpt-6.1-sol",
        "source_generation_id": "teacher-generation",
        "prompt_id": "training-contact",
        "messages": [
            {"role": "system", "content": "Draw Bufo"},
            {"role": "user", "content": "Hold a mug"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(
                    {"version": "bufo-scene-plan-v1", "warnings": [], "anchors": []}
                ),
            },
        ],
    }


def test_extract_preserves_real_teacher_call_and_result():
    m, source = module(), example()
    plan = m.teacher_plan(source)
    assert plan["messages"] == source["messages"][2:]
    assert plan["items"][0]["call_id"] == "actual-teacher-call"
    assert plan["items"][1]["output"] == source["messages"][3]["content"]
    source["messages"][3]["content"] = '{"version":"bufo-scene-plan-v1","warnings":["cropped"]}'
    with pytest.raises(ValueError, match="successful"):
        m.teacher_plan(source)


def test_hook_is_scoped_and_marks_mixed_before_logging():
    m = module()
    original_input, original_capture, original_agent = (
        agent.conversation_input,
        agent.capture_trajectory,
        benchmark.BufoAgent,
    )
    plan = m.teacher_plan(example())
    with m.diagnostic_hooks({"p": plan}):
        assert agent.conversation_input("Hold a mug") == (
            [{"role": "user", "content": "Hold a mug"}],
            True,
        )
        token = m.CONTEXT.set({"arm": m.SUPPLIED, "plan": plan, "training_eligible": False})
        try:
            items, authentic = agent.conversation_input("Hold a mug")
            assert items[1:] == plan["items"]
            assert authentic is False
            captured = agent.capture_trajectory(
                items=items + [{"role": "assistant", "content": "Done"}],
                system_instructions="Draw Bufo",
                tools=[],
                provenance={"model": "student", "prompt_version": "v"},
                final_output="Done",
                history_authentic=authentic,
                expected_input=items,
            )
            assert captured["provenance"]["source_models"] == ["gpt-6.1-sol", "student"]
            assert captured["provenance"]["training_eligible"] is False
            assert captured["provenance"]["history_authentic"] is False
            row = BufoGeneration(
                prompt_id="p",
                prompt="Hold a mug",
                backend="x",
                model="student",
                group_id="g",
                svg="",
                valid=True,
                latency_s=0,
                trajectory=captured,
            )
            with pytest.raises(ValueError, match="diagnostic"):
                export_training_example(row)
        finally:
            m.CONTEXT.reset(token)
    assert (agent.conversation_input, agent.capture_trajectory, benchmark.BufoAgent) == (
        original_input,
        original_capture,
        original_agent,
    )


def test_ordinary_capture_also_training_ineligible():
    m = module()
    with m.diagnostic_hooks({}):
        token = m.CONTEXT.set({"arm": m.ORDINARY, "training_eligible": False})
        try:
            captured = agent.capture_trajectory(
                items=[
                    {"role": "user", "content": "Hold a mug"},
                    {"role": "assistant", "content": "Done"},
                ],
                system_instructions="Draw Bufo",
                tools=[],
                provenance={"model": "student"},
                final_output="Done",
            )
            assert captured["provenance"]["source_models"] == ["student"]
            assert captured["provenance"]["training_eligible"] is False
            assert any("not training eligible" in d for d in captured["diagnostics"])
        finally:
            m.CONTEXT.reset(token)


def test_fixed_fit_check_catches_wrong_requirements_or_coverage():
    m = module()
    prompts = [{"id": str(i), "requirements": ["Hand grips prop"]} for i in range(16)]
    rows = [
        {
            "prompt_id": p["id"],
            "benchmark_metadata": {"requirements": p["requirements"]},
            "scores": {
                "scene_quality": {
                    "status": "ok",
                    "requirements": p["requirements"],
                    "constraint_coverage": 0,
                    "prompt_checks": [{"requirement": "Hand grips prop", "satisfied": False}],
                }
            },
        }
        for p in prompts
    ]
    assert m.verify_fit_scores(rows, prompts)["zero_coverage_rows"] == 16
    rows[0]["scores"]["scene_quality"]["constraint_coverage"] = 1
    with pytest.raises(ValueError, match="inconsistency"):
        m.verify_fit_scores(rows, prompts)
