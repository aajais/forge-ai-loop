"""Nonblocking geometry feedback for the supplied full-body character."""

import math
import re
import xml.etree.ElementTree as ET

IDENTITY = (1, 0, 0, 1, 0, 0)


def multiply(a, b):
    return (
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    )


def transform(value):
    result = IDENTITY
    for name, arguments in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", value):
        numbers = [float(n) for n in re.split(r"[\s,]+", arguments.strip()) if n]
        if not all(math.isfinite(n) for n in numbers):
            raise ValueError("Nonfinite transform")
        if name == "translate" and len(numbers) in (1, 2):
            matrix = (1, 0, 0, 1, numbers[0], numbers[1] if len(numbers) == 2 else 0)
        elif name == "scale" and len(numbers) in (1, 2):
            matrix = (numbers[0], 0, 0, numbers[-1], 0, 0)
        elif name == "matrix" and len(numbers) == 6:
            matrix = tuple(numbers)
        elif name == "rotate" and len(numbers) in (1, 3):
            angle = math.radians(numbers[0])
            c, s = math.cos(angle), math.sin(angle)
            matrix = (c, s, -s, c, 0, 0)
            if len(numbers) == 3:
                x, y = numbers[1:]
                matrix = multiply(multiply((1, 0, 0, 1, x, y), matrix), (1, 0, 0, 1, -x, -y))
        else:
            raise ValueError("Unsupported transform for bounds check")
        result = multiply(result, matrix)
    return result


def framing_warnings(svg: str, prompt: str) -> list[str]:
    # Portraits may deliberately crop the same asset. Rendering still handles
    # unsupported geometry; this check never invalidates an otherwise valid SVG.
    if re.search(r"close[ -]?up|portrait|cropp?ed|\bbust\b", prompt, flags=re.I):
        return []
    warnings = []
    try:
        root = ET.fromstring(svg)
        x, y, width, height = map(float, re.split(r"[\s,]+", root.get("viewBox", "0 0 512 512")))

        def walk(node, parent):
            if node.tag.split("}")[-1] in {"defs", "mask", "clipPath"}:
                return
            matrix = multiply(parent, transform(node.get("transform", "")))
            if node.get("href", node.get("{http://www.w3.org/1999/xlink}href")) == "#bufo-fullbody":
                matrix = multiply(
                    matrix, (1, 0, 0, 1, float(node.get("x", 0)), float(node.get("y", 0)))
                )
                points = [
                    (
                        matrix[0] * a + matrix[2] * b + matrix[4],
                        matrix[1] * a + matrix[3] * b + matrix[5],
                    )
                    for a, b in [(0, 0), (128, 0), (0, 176), (128, 176)]
                ]
                left, right = min(p[0] for p in points), max(p[0] for p in points)
                top, bottom = min(p[1] for p in points), max(p[1] for p in points)
                if left < x - 1 or top < y - 1 or right > x + width + 1 or bottom > y + height + 1:
                    warnings.append(
                        f"Bufo is cropped: character bounds ({left:.1f}, {top:.1f}) to "
                        f"({right:.1f}, {bottom:.1f}) exceed canvas ({x:g}, {y:g}) to "
                        f"({x + width:g}, {y + height:g}). Move or scale the WHOLE character group, "
                        "including its clothes and arms, to leave 16px margin. Render again."
                    )
            for child in node:
                walk(child, matrix)

        walk(root, IDENTITY)
    except (ET.ParseError, ValueError, OverflowError):
        pass
    return warnings
