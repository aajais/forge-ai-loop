"""Observable scene layout; measures a proposed plan without inspecting its SVG."""

import json
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

PLANNING_VERSION = "bufo-scene-plan-v1"
Coordinate = Annotated[float, Field(ge=-1024, le=1024, allow_inf_nan=False)]


def nonblank_label(value: str) -> str:
    # The serving grammar rejects regex patterns; enforce nonblank labels locally.
    if not value.strip():
        raise ValueError("Scene labels must contain non-whitespace text.")
    return value


Label = Annotated[str, Field(min_length=1, max_length=64), AfterValidator(nonblank_label)]


class BufoPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: Coordinate
    y: Coordinate
    scale: Annotated[float, Field(gt=0, le=8, allow_inf_nan=False)]


class ContactAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Label
    x: Coordinate
    y: Coordinate


async def plan_scene(
    main_bufo: BufoPlacement,
    coordinate_frame: Literal["bufo_local"],
    contact_anchors: Annotated[list[ContactAnchor], Field(max_length=12)],
    paint_order: Annotated[list[Label], Field(min_length=1, max_length=16)],
    framing: Literal["fullbody", "close_up"],
) -> str:
    """Record a compact scene layout before drawing on a 512×512 canvas.

    Args:
        main_bufo: Whole-character translation and uniform scale, including clothes and arms.
        coordinate_frame: All contact anchors use bufo_local (untransformed 128×176 character).
        contact_anchors: Named shared contact points, e.g. hand_cup_handle; empty for no contacts.
        paint_order: Short layer names in back-to-front SVG order, such as background, bufo, cup, hand.
        framing: fullbody warns about cropping; close_up allows intentional character cropping.
    """
    names = [anchor.name.strip() for anchor in contact_anchors]
    if len(names) != len(set(names)):
        raise ValueError("Contact anchor names must be unique.")
    x, y, scale = main_bufo.x, main_bufo.y, main_bufo.scale
    bounds = [x, y, x + 128 * scale, y + 176 * scale]
    warnings = []
    if framing == "fullbody" and (
        bounds[0] < 0 or bounds[1] < 0 or bounds[2] > 512 or bounds[3] > 512
    ):
        warnings.append(
            "Planned Bufo bounds exceed the 512×512 canvas; move or scale the whole group."
        )
    anchors = []
    for name, anchor in zip(names, contact_anchors, strict=True):
        canvas = [x + anchor.x * scale, y + anchor.y * scale]
        anchors.append(
            {
                "name": name,
                "local": [anchor.x, anchor.y],
                "canvas": canvas,
            }
        )
        if any(value < 0 or value > 512 for value in canvas):
            warnings.append(f"Planned anchor {name} is outside the canvas.")
    return json.dumps(
        {
            "version": PLANNING_VERSION,
            "verification": "planned_geometry_only",
            "canvas": [0, 0, 512, 512],
            "coordinate_frame": coordinate_frame,
            "group_transform": f"translate({x:g} {y:g}) scale({scale:g})",
            "bufo_canvas_bounds": bounds,
            "anchors": anchors,
            "paint_order": paint_order,
            "warnings": warnings,
            "guidance": (
                "Draw Bufo, clothing and arms in one transformed group. Reuse local contact points "
                "inside that group; use converted canvas points for outside props. Paint back to front. "
                "These are proposed coordinates, not verified SVG contacts or image quality."
            ),
        },
        allow_nan=False,
        separators=(",", ":"),
    )
