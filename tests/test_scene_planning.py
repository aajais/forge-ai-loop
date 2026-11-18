"""The scene-planning contract checks geometry, not the resulting picture."""

import importlib
import json

import pytest
from agents import function_tool
from agents.exceptions import ModelBehaviorError
from agents.tool_context import ToolContext


def planning():
    return importlib.import_module("bufogen.agents.planning")


def payload(**changes):
    data = {
        "main_bufo": {"x": 100, "y": 80, "scale": 2},
        "coordinate_frame": "bufo_local",
        "contact_anchors": [{"name": "hand_cup_handle", "x": 110, "y": 90}],
        "paint_order": ["background", "bufo", "cup", "hand"],
        "framing": "fullbody",
    }
    data.update(changes)
    return data


async def invoke(data):
    tool = function_tool(planning().plan_scene, failure_error_function=None)
    arguments = json.dumps(data)
    context = ToolContext(
        context=None, tool_name="plan_scene", tool_call_id="plan1", tool_arguments=arguments
    )
    return json.loads(await tool.on_invoke_tool(context, arguments))


async def test_plan_converts_local_anchors_and_character_bounds_to_canvas():
    result = await invoke(payload())
    assert result["bufo_canvas_bounds"] == [100, 80, 356, 432]
    assert result["anchors"][0] == {
        "name": "hand_cup_handle",
        "local": [110, 90],
        "canvas": [320, 260],
    }
    assert result["group_transform"] == "translate(100 80) scale(2)"
    assert result["paint_order"] == ["background", "bufo", "cup", "hand"]
    assert result["warnings"] == []
    assert result["verification"] == "planned_geometry_only"


async def test_crop_warning_is_measurement_not_a_render_rejection():
    result = await invoke(payload(main_bufo={"x": 300, "y": 300, "scale": 2}))
    assert result["bufo_canvas_bounds"] == [300, 300, 556, 652]
    assert any("bounds" in warning for warning in result["warnings"])
    assert any("anchor" in warning for warning in result["warnings"])
    assert "valid" not in result


async def test_close_up_permits_character_crop_but_reports_off_canvas_anchor():
    result = await invoke(payload(main_bufo={"x": 300, "y": 300, "scale": 2}, framing="close_up"))
    assert not any("bounds" in warning for warning in result["warnings"])
    assert any("anchor" in warning for warning in result["warnings"])


@pytest.mark.parametrize(
    "change",
    [
        {"main_bufo": {"x": float("nan"), "y": 0, "scale": 1}},
        {"main_bufo": {"x": 0, "y": float("inf"), "scale": 1}},
        {"main_bufo": {"x": 0, "y": 0, "scale": 0}},
        {"coordinate_frame": "canvas"},
        {"contact_anchors": [{"name": "a" * 65, "x": 1, "y": 1}]},
        {"contact_anchors": [{"name": "a", "x": 1, "y": 1}] * 13},
        {"paint_order": ["layer"] * 17},
        {"paint_order": []},
    ],
)
async def test_tool_rejects_nonfinite_or_unbounded_plan_inputs(change):
    with pytest.raises((ModelBehaviorError, ValueError)):
        await invoke(payload(**change))


async def test_empty_contacts_supported_for_simple_portraits():
    result = await invoke(payload(contact_anchors=[]))
    assert result["anchors"] == []
    assert len(json.dumps(result)) < 1600


async def test_duplicate_anchor_names_are_rejected_to_keep_coordinates_unambiguous():
    with pytest.raises(ValueError, match="unique"):
        await invoke(
            payload(
                contact_anchors=[
                    {"name": "hand", "x": 20, "y": 30},
                    {"name": "hand", "x": 80, "y": 30},
                ]
            )
        )


def test_tool_schema_is_strict_and_preserves_nested_geometry_constraints():
    schema = function_tool(planning().plan_scene).params_json_schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(payload())
    assert schema["properties"]["contact_anchors"]["maxItems"] == 12
    assert schema["$defs"]["BufoPlacement"]["properties"]["scale"]["exclusiveMinimum"] == 0


def test_serving_schema_omits_xgrammar_incompatible_pattern():
    schema = function_tool(planning().plan_scene).params_json_schema
    assert '"pattern"' not in json.dumps(schema)
    assert schema["properties"]["paint_order"]["items"]["minLength"] == 1


@pytest.mark.parametrize("change", [
    {"contact_anchors": [{"name": " \t\n", "x": 1, "y": 1}]},
    {"paint_order": ["background", "  \t"]},
])
async def test_runtime_rejects_blank_labels_without_regex_in_serving_schema(change):
    with pytest.raises((ModelBehaviorError, ValueError)):
        await invoke(payload(**change))
