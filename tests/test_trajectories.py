import json
from types import SimpleNamespace

import pytest

from bufogen.agents.agent import BufoAgent
from bufogen.generation import BufoGeneration, Manifest
from bufogen.models import load_backends
from bufogen.rendering.renderer import RenderResult

SCENE = '<svg xmlns="http://www.w3.org/2000/svg"><use href="#bufo-fullbody"/></svg>'


def source_items(svg=SCENE, final="Here is Bufo!", call_id="actual-call-1"):
    return [
        {"role": "user", "content": "Draw Bufo"},
        {"type": "reasoning", "summary": [], "encrypted_content": "not-training-data"},
        {
            "type": "function_call",
            "name": "render_svg",
            "call_id": call_id,
            "arguments": json.dumps({"svg": svg, "emoji_name": ":bufo-cloud:"}),
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps({"valid": True, "composition_warnings": []}),
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final, "annotations": []}],
        },
    ]


def generation():
    return BufoGeneration(
        prompt_id="p",
        prompt="Draw Bufo",
        backend="teacher",
        model="m",
        group_id="g",
        svg=SCENE,
        valid=True,
        latency_s=1,
        prompt_version="v1",
    )


def captured_row(items=None):
    from bufogen.curation.trajectories import capture_trajectory

    row = generation()
    row.trajectory = capture_trajectory(
        items=source_items() if items is None else items,
        system_instructions="Frozen rules",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "render_svg",
                    "description": "Render",
                    "parameters": {
                        "type": "object",
                        "properties": {"svg": {"type": "string"}, "emoji_name": {"type": "string"}},
                        "required": ["svg", "emoji_name"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        provenance={
            "model": row.model,
            "prompt_id": row.prompt_id,
            "group_id": row.group_id,
            "prompt_version": row.prompt_version,
        },
        final_output="Here is Bufo!",
    )
    return row


def test_legacy_generation_is_not_fabricated(tmp_path):
    from bufogen.curation.trajectories import export_training_example

    path = tmp_path / "legacy.jsonl"
    legacy = generation().to_dict()
    legacy.pop("trajectory")
    legacy.pop("training_review")
    path.write_text(json.dumps(legacy) + "\n")
    row = Manifest(path).read()[0]
    with pytest.raises(ValueError, match="trajectory"):
        export_training_example(row)


def test_portable_export_keeps_actual_calls_compact_arguments_and_frozen_rules():
    from bufogen.curation.trajectories import export_training_example

    row = captured_row()
    example = export_training_example(row)
    assert example["messages"][0] == {"role": "system", "content": "Frozen rules"}
    call = example["messages"][2]["tool_calls"][0]
    assert call["id"] == "actual-call-1"
    assert json.loads(call["function"]["arguments"])["svg"] == SCENE
    assert example["messages"][3]["tool_call_id"] == call["id"]
    assert example["messages"][-1]["content"] == "Here is Bufo!"
    assert "not-training-data" not in json.dumps(row.trajectory)
    assert example["source_generation_id"] == row.id


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda items: items.pop(2), "orphan"),
        (lambda items: items.pop(3), "unresolved"),
        (lambda items: items[-1].update(content="<svg/>"), "SVG"),
        (lambda items: items[2].update(name="unknown"), "unknown tool"),
        (
            lambda items: items[0].update(content=[{"type": "input_image", "image_url": "data:x"}]),
            "multimodal",
        ),
    ],
)
def test_export_rejects_inconsistent_or_unsupported_trajectories(mutation, match):
    from bufogen.curation.trajectories import export_training_example

    items = source_items()
    mutation(items)
    with pytest.raises(ValueError, match=match):
        export_training_example(captured_row(items))


async def test_agent_preserves_repair_history_and_dynamic_tool_schema(tmp_path):
    class Renderer:
        async def render(self, svg):
            return RenderResult(
                b"png" if svg == "<svg/>" else None,
                svg == "<svg/>",
                None if svg == "<svg/>" else "invalid",
            )

    calls = []

    async def run(items, group_id, render):
        calls.append(items)
        svg = "<svg>" if len(calls) == 1 else "<svg/>"
        output = await render(svg, ":bufo-cloud:")
        additions = source_items(svg=svg, call_id=f"real-{len(calls)}")[2:]
        additions[1]["output"] = output
        return SimpleNamespace(
            final_output="Here is Bufo!",
            to_input_list=lambda: items + additions,
            context_wrapper=SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=1)),
        )

    row = await BufoAgent(load_backends()["open_base"], Renderer(), tmp_path, run=run).generate(
        "Draw Bufo", prompt_id="p", group_id="g"
    )
    assert row.trajectory is not None
    from bufogen.curation.trajectories import export_training_example

    example = export_training_example(row)
    calls = [c for m in example["messages"] for c in m.get("tool_calls", [])]
    assert [c["id"] for c in calls] == ["real-1", "real-2"]
    assert any("Rendering status" in str(m.get("content")) for m in example["messages"])
    assert {tool["function"]["name"] for tool in example["tools"]} == {
        "plan_scene",
        "render_svg",
        "search_web",
    }
    render_tool = next(
        tool["function"] for tool in example["tools"] if tool["function"]["name"] == "render_svg"
    )
    assert render_tool["parameters"]["properties"]["emoji_name"]


def test_followup_reuses_authentic_history_without_duplicate_or_invented_calls():
    from bufogen.curation.trajectories import conversation_input

    previous = captured_row()
    messages, authentic = conversation_input("Add a scarf", history=[previous.to_dict()])
    assert authentic
    assert [m["call_id"] for m in messages if m.get("type") == "function_call"] == ["actual-call-1"]
    assert messages[-1] == {"role": "user", "content": "Add a scarf"}
    legacy = generation()
    messages, authentic = conversation_input("Add a scarf", history=[legacy.to_dict()])
    assert not authentic
    assert not any(m.get("type") == "function_call" for m in messages)
    assert "previous_drawing_" not in json.dumps(messages)


def test_incomplete_sdk_snapshot_preserves_input_but_cannot_export():
    from bufogen.curation.trajectories import (
        capture_trajectory,
        conversation_input,
        export_training_example,
    )

    row = captured_row()
    original_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Draw Bufo"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ],
        }
    ]
    row.trajectory = capture_trajectory(
        items=[],
        system_instructions="Frozen rules",
        tools=row.trajectory["tools"],
        provenance=row.trajectory["provenance"],
        final_output="Here is Bufo!",
        expected_input=original_input,
    )
    with pytest.raises(ValueError, match="incomplete SDK"):
        export_training_example(row)
    messages, authentic = conversation_input("Add a scarf", history=[row.to_dict()])
    assert messages[0] == original_input[0]
    assert not authentic


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.trajectory["provenance"].update(model="different-model"), "provenance"),
        (lambda row: row.trajectory["messages"][0].update(content="new rules"), "system"),
        (
            lambda row: row.trajectory["messages"][2]["tool_calls"][0]["function"].update(
                arguments=json.dumps({"svg": '<svg><g id="bufo-character"/></svg>'})
            ),
            "canonical",
        ),
        (lambda row: row.trajectory["messages"][3].update(content='{"valid":false}'), "render"),
        (
            lambda row: row.trajectory["messages"][-1].update(content="invented reply"),
            "final reply",
        ),
    ],
)
def test_export_detects_changed_provenance_expanded_assets_or_failed_render(mutation, match):
    from bufogen.curation.trajectories import export_training_example

    row = captured_row()
    mutation(row)
    with pytest.raises(ValueError, match=match):
        export_training_example(row)


def test_parallel_tool_calls_keep_complete_associations():
    from bufogen.curation.trajectories import export_training_example

    items = source_items()
    items.insert(
        3,
        {
            "type": "function_call",
            "name": "render_svg",
            "call_id": "second",
            "arguments": json.dumps({"svg": "<svg/>"}),
        },
    )
    items.insert(
        5, {"type": "function_call_output", "call_id": "second", "output": '{"valid":true}'}
    )
    example = export_training_example(captured_row(items))
    assert len(example["messages"][2]["tool_calls"]) == 2
    assert example["messages"][4]["tool_call_id"] == "second"


def test_multimodal_capture_survives_manifest_for_continuation(tmp_path):
    from bufogen.curation.trajectories import conversation_input, export_training_example

    items = source_items()
    items[0]["content"] = [
        {"type": "input_text", "text": "Draw Bufo"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    row = captured_row(items)
    manifest = Manifest(tmp_path / "manifest.jsonl")
    manifest.append(row)
    restored = manifest.read()[0]
    with pytest.raises(ValueError, match="multimodal"):
        export_training_example(restored)
    continued, authentic = conversation_input("Add a scarf", history=[restored.to_dict()])
    assert authentic
    assert continued[0] == items[0]
    assert [item["call_id"] for item in continued if item.get("type") == "function_call"] == [
        "actual-call-1"
    ]


def test_changed_history_snapshot_is_retained_but_cannot_train():
    from bufogen.curation.trajectories import capture_trajectory, export_training_example

    row = captured_row()
    previous = row.trajectory
    row.trajectory = capture_trajectory(
        items=source_items(),
        system_instructions="Changed system rules",
        tools=previous["tools"],
        provenance=previous["provenance"],
        final_output="Here is Bufo!",
        history_snapshot=previous,
    )
    assert row.trajectory["messages"][2]["tool_calls"][0]["id"] == "actual-call-1"
    with pytest.raises(ValueError, match="history snapshot"):
        export_training_example(row)


def test_mixed_model_history_and_descendants_cannot_train():
    from bufogen.curation.trajectories import capture_trajectory, export_training_example

    row = captured_row()
    previous = row.trajectory
    row.model = "another-model"
    for _ in range(2):
        row.trajectory = capture_trajectory(
            items=source_items(),
            system_instructions=previous["system_instructions"],
            tools=previous["tools"],
            provenance={**previous["provenance"], "model": row.model},
            final_output="Here is Bufo!",
            history_snapshot=previous,
        )
        assert row.trajectory["provenance"]["source_models"] == ["m", "another-model"]
        with pytest.raises(ValueError, match="mixed-model"):
            export_training_example(row)
        previous = row.trajectory


def test_same_model_followup_cannot_launder_unreviewed_ancestor_targets():
    from bufogen.curation.trajectories import capture_trajectory, export_training_example

    row = captured_row()
    previous = row.trajectory
    row.trajectory = capture_trajectory(
        items=source_items(),
        system_instructions=previous["system_instructions"],
        tools=previous["tools"],
        provenance=previous["provenance"],
        final_output="Here is Bufo!",
        history_snapshot=previous,
    )
    assert row.trajectory["messages"][-1]["content"] == "Here is Bufo!"
    with pytest.raises(ValueError, match="prior conversation"):
        export_training_example(row)
