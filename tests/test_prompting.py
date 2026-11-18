import json

from bufogen.agents.prompting import assemble_character, generation_input, prompt_version
from bufogen.rendering.svg import validate_svg


def test_character_guidance_is_system_only_without_reference_user_messages():
    import re

    from bufogen.agents.agent import instructions

    assert generation_input("A bufo tending GPUs") == [
        {"role": "user", "content": "A bufo tending GPUs"}
    ]
    system = instructions()
    assert "#bufo-fullbody" in system and "128x176" in system
    examples = re.findall(r"<svg.*?</svg>", system, flags=re.S)
    assert len(examples) == 2
    for example in examples:
        assert validate_svg(assemble_character(example))[0]
    assert "data:image" not in system
    assert prompt_version().startswith("bufo-v6-")


def test_character_fragment_becomes_portable_svg_and_preserves_scene():
    import xml.etree.ElementTree as ET

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
        '<use href="#bufo-character" transform="translate(70 110) scale(2.4)"/>'
        '<rect id="scene" x="380" y="320" width="85" height="85" fill="#2374FF"/></svg>'
    )
    resolved = assemble_character(svg)
    assert validate_svg(resolved)[0]
    root = ET.fromstring(resolved)
    definitions = [n for n in root.iter() if n.get("id") == "bufo-character"]
    assert len(definitions) == 1 and len(definitions[0]) > 10
    assert any(n.get("id") == "scene" and n.get("x") == "380" for n in root.iter())
    assert assemble_character(resolved) == resolved
    assert assemble_character("<svg>") == "<svg>"


def test_followup_keeps_all_requests_and_only_latest_drawing_in_tool_history():
    history = [
        {
            "id": "a",
            "prompt": "Bufo on a cloud",
            "svg": "<svg id='old'/>",
            "valid": True,
            "response_text": "Here is Bufo on a cloud.",
        },
        {
            "id": "b",
            "prompt": "Add a blue scarf",
            "svg": "<svg id='latest'/>",
            "valid": True,
            "response_text": "Here you go—with a blue scarf!",
        },
    ]
    messages = generation_input("Make the scarf red", history=history)
    assert [m["content"] for m in messages if m.get("role") == "user"] == [
        "Bufo on a cloud",
        "Add a blue scarf",
        "Make the scarf red",
    ]
    calls = [m for m in messages if m.get("type") == "function_call"]
    assert len(calls) == 1  # Only the actual previous drawing is conversation history.
    assert json.loads(calls[-1]["arguments"])["svg"] == history[-1]["svg"]
    assert all("<svg" not in m["content"] for m in messages if m.get("role") == "assistant")


def test_followup_omits_embedded_character_paths_but_preserves_editable_scene():
    import xml.etree.ElementTree as ET

    scene = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
        '<use href="#bufo-character" transform="translate(70 110) scale(2.4)"/>'
        '<rect id="blue-box" x="380" y="320" width="85" height="85" fill="#2374FF"/>'
        "</svg>"
    )
    saved = assemble_character(scene)
    messages = generation_input(
        "Add a hat",
        history=[
            {
                "id": "previous",
                "prompt": "Bufo next to a box",
                "svg": saved,
                "valid": True,
            }
        ],
    )
    call = next(m for m in messages if m.get("call_id") == "previous_drawing_previous")
    compact = json.loads(call["arguments"])["svg"]
    assert len(compact) < 500
    nodes = list(ET.fromstring(compact).iter())
    assert not any(n.get("id") == "bufo-character" for n in nodes)
    assert any(n.get("id") == "blue-box" for n in nodes)
    assert any(n.get("href") == "#bufo-character" for n in nodes)
    assert assemble_character(compact) == saved


def test_character_assembly_does_not_sanitize_unsafe_model_svg():
    svg = '<svg onload="bad()"><use href="#bufo-character"/></svg>'
    assert assemble_character(svg) == svg
    assert not validate_svg(assemble_character(svg))[0]


def test_screenshots_are_image_parts_in_current_and_historical_user_messages():
    current, previous = "data:image/png;base64,YQ==", "data:image/png;base64,Yg=="
    messages = generation_input(
        "Use this scene",
        image=current,
        history=[
            {
                "id": "previous",
                "prompt": "Use these colors",
                "image": previous,
                "valid": True,
                "response_text": "Here you go!",
            }
        ],
    )
    users = [m for m in messages if m.get("role") == "user"]
    assert users[-2]["content"][1]["image_url"] == previous
    assert users[-1]["content"] == [
        {"type": "input_text", "text": "Use this scene"},
        {"type": "input_image", "image_url": current, "detail": "high"},
    ]


def test_fullbody_fragment_renders_connected_and_roundtrips():
    import io

    import resvg_py
    from PIL import Image

    from bufogen.agents.prompting import editable_scene

    scene = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 176"><use href="#bufo-fullbody"/></svg>'
    resolved = assemble_character(scene)
    assert validate_svg(resolved)[0]
    assert assemble_character(resolved) == resolved
    assert assemble_character(editable_scene(resolved)) == resolved
    assert len(editable_scene(resolved)) < 300
    image = Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg_string=resolved))).convert("RGBA")
    # The formerly cropped neck edge now continues into torso, then attached feet.
    assert all(image.getpixel((64, y))[3] > 240 for y in range(90, 145))
    assert image.getpixel((20, 171))[3] > 240
