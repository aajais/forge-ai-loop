"""Shared, versioned character examples for every generation backend."""

import hashlib
import json
import xml.etree.ElementTree as ET

from bufogen.rendering.svg import validate_svg
from bufogen.settings import ROOT

REFERENCE_NAMES = ("bufo-concerned", "bufo-awkward-smile", "bufo-all-good")
CHARACTER_ID = "bufo-character"


def assemble_character(svg: str) -> str:
    """Resolve the reserved character fragment into a portable, self-contained SVG.

    The model controls composition. Only the canonical character definition is
    supplied by the application, identically across all model backends.
    """
    if not validate_svg(svg)[0]:
        return svg
    root = ET.fromstring(svg)
    references = {
        v[1:]
        for node in root.iter()
        if node.tag.split("}")[-1] == "use"
        for k, v in node.attrib.items()
        if k.split("}")[-1] == "href" and v.startswith("#")
    }
    needed = references & {CHARACTER_ID, "bufo-fullbody"}
    if not needed:
        return svg
    if "bufo-fullbody" in needed:
        needed.add(CHARACTER_ID)
    for parent in root.iter():
        for child in list(parent):
            if child.get("id") in needed:
                parent.remove(child)
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    defs = next((n for n in root if n.tag.split("}")[-1] == "defs"), None)
    if defs is None:
        defs = ET.Element("{" + namespace + "}defs")
        root.insert(0, defs)
    for name, filename in [
        (CHARACTER_ID, "neutral-example.svg"),
        ("bufo-fullbody", "fullbody.svg"),
    ]:
        if name in needed:
            group = ET.Element("{" + namespace + "}g", id=name)
            source = ET.fromstring((ROOT / "assets/bufo_style" / filename).read_text())
            group.extend(list(source))
            defs.insert(0, group)
    return ET.tostring(root, encoding="unicode")


def style_references():
    return [ROOT / "assets/bufo_style" / f"{name}.png" for name in REFERENCE_NAMES]


def editable_scene(svg: str) -> str:
    """Remove renderer-supplied geometry from history so the model edits the scene.

    The character's use/transform stays intact; assemble_character restores its
    canonical paths at render time. Other definitions and scene paths are kept.
    """
    if not validate_svg(svg)[0]:
        return svg
    root = ET.fromstring(svg)
    if not any(
        node.tag.split("}")[-1] == "use"
        and any(
            k.split("}")[-1] == "href" and v in {"#" + CHARACTER_ID, "#bufo-fullbody"}
            for k, v in node.attrib.items()
        )
        for node in root.iter()
    ):
        return svg
    for parent in root.iter():
        for child in list(parent):
            if child.get("id") in {CHARACTER_ID, "bufo-fullbody"}:
                parent.remove(child)
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    return ET.tostring(root, encoding="unicode")


def prompt_version():
    files = [
        ROOT / "src/bufogen/agents/prompting.py",
        ROOT / "src/bufogen/agents/agent.py",
        ROOT / "src/bufogen/rendering/composition.py",
        ROOT / "src/bufogen/agents/planning.py",
        ROOT / "src/bufogen/agents/web_context.py",
        ROOT / "config/brand.yaml",
        ROOT / "assets/bufo_style/neutral-example.svg",
        ROOT / "assets/bufo_style/fullbody.svg",
        ROOT / "assets/bufo_style/presenter-example.svg",
        *style_references(),
    ]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:12]
    return "bufo-v6-" + digest


def user_content(prompt: str, image: str | None = None):
    if not image:
        return prompt
    return [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": image, "detail": "high"},
    ]


def generation_input(
    prompt: str, *, history: list[dict] | None = None, image: str | None = None
) -> list[dict]:
    messages = []
    for index, previous in enumerate(history or []):
        messages.append(
            {"role": "user", "content": user_content(previous["prompt"], previous.get("image"))}
        )
        # Keep all conversational requests, but only the latest drawing's code.
        if index == len(history) - 1 and previous.get("svg"):
            call_id = "previous_drawing_" + previous["id"]
            messages.extend(
                [
                    {
                        "type": "function_call",
                        "name": "render_svg",
                        "call_id": call_id,
                        "arguments": json.dumps(
                            {
                                "svg": editable_scene(previous["svg"]),
                                "emoji_name": previous.get("emoji_name", ""),
                            }
                        ),
                    },
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(
                            {"valid": previous["valid"], "image_ready": previous["valid"]}
                        ),
                    },
                ]
            )
        messages.append(
            {
                "role": "assistant",
                "content": previous.get("response_text") or "Here is the drawing.",
            }
        )
    messages.append({"role": "user", "content": user_content(prompt, image)})
    return messages
