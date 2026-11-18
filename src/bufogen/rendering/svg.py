"""Conservative SVG subset, shared verbatim with the isolated worker."""

import re
import xml.etree.ElementTree as ET

MAX_SVG_BYTES = 100_000
ALLOWED = set(
    "svg g defs path rect circle ellipse line polyline polygon text tspan title desc linearGradient radialGradient stop clipPath mask use".split()
)


class _SVGTreeBuilder(ET.TreeBuilder):
    """Allow XML headers/comments, but reject DTDs before entity expansion."""

    def doctype(self, name, pubid, system):
        raise ValueError("DOCTYPE declarations and entities are forbidden.")

    def pi(self, target, text):
        raise ValueError("Processing instructions are forbidden.")


def extract_svg(text: str) -> str:
    text = text.strip()
    # Some baseline responses leak a channel delimiter before the actual output.
    # Strip only that known wrapper; retain XML declarations for validation.
    text = re.sub(r"^<\|channel>(?:thought|final|analysis)\s*<channel\|>\s*", "", text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:svg|xml)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_svg(svg: str) -> tuple[bool, str | None]:
    if not isinstance(svg, str) or len(svg.encode()) > MAX_SVG_BYTES:
        return False, "SVG exceeds the 100 KB limit."
    try:
        root = ET.fromstring(svg, parser=ET.XMLParser(target=_SVGTreeBuilder()))
    except (ET.ParseError, ValueError) as exc:
        return False, f"Malformed SVG: {exc}"
    if root.tag.split("}")[-1] != "svg":
        return False, "Root must be svg."
    nodes = list(root.iter())
    if len(nodes) > 2500:
        return False, "Too many SVG elements."
    for node in nodes:
        tag = node.tag.split("}")[-1]
        if tag not in ALLOWED or (
            "}" in node.tag and not node.tag.startswith("{http://www.w3.org/2000/svg}")
        ):
            return False, f"Unsupported SVG element: {tag}"
        for attr, value in node.attrib.items():
            name = attr.split("}")[-1].lower()
            if name.startswith("on") or name in {"style", "base"}:
                return False, f"Forbidden SVG attribute: {name}"
            if name == "href" and not re.fullmatch(r"#[A-Za-z_][\w.-]*", value):
                return False, "Only internal fragment references are allowed."
            if re.search(r"url\s*\(", value, re.I) and not re.fullmatch(
                r"url\(#[A-Za-z_][\w.-]*\)", value
            ):
                return False, "External or obfuscated references are forbidden."
            if "\\" in value or "://" in value or "data:" in value.lower():
                return False, "External resources are forbidden."
            if name in {"width", "height"}:
                match = re.fullmatch(r"((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:px|%)?", value)
                if not match or not 0 < float(match[1]) <= 4096:
                    return False, "SVG dimensions must be positive numbers no greater than 4096."
    return True, None
