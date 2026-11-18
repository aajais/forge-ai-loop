import base64
import json

import pytest

from bufogen.generation import BufoGeneration, Manifest
from bufogen.models import load_backends
from bufogen.rendering.renderer import Renderer
from bufogen.rendering.svg import extract_svg, validate_svg

GOOD = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path fill="#62a744" d="M0 0h50v50z"/></svg>'


@pytest.mark.parametrize(
    "svg",
    [
        "<svg>",
        "<html/>",
        "<svg><script>alert(1)</script></svg>",
        '<svg onload="alert(1)"/>',
        '<svg><image href="https://example.com/a.png"/></svg>',
        '<svg><image href="file:///etc/passwd"/></svg>',
        '<svg><style>@import "https://example.com/x";</style></svg>',
        '<svg><path fill="url(https://example.com/x)"/></svg>',
        '<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/passwd">]><svg>&x;</svg>',
        '<!DOCTYPE svg SYSTEM "https://example.com/svg.dtd"><svg/>',
        '<?xml-stylesheet href="https://example.com/style.css"?><svg/>',
        "<svg><?unsafe instruction?></svg>",
        "<svg/><?unsafe instruction?>",
        '<!-- harmless --><!DOCTYPE svg [<!ENTITY x "expanded">]><svg>&x;</svg>',
        "<svg><!-- invalid -- comment --></svg>",
        "<svg><foreignObject><div>hi</div></foreignObject></svg>",
        '<svg><animate attributeName="href" values="x;https://bad"/></svg>',
        '<svg width="999999" height="999999"/>',
    ],
)
def test_unsafe_svg_is_rejected(svg):
    assert not validate_svg(svg)[0]


def test_safe_svg_and_internal_gradient_are_accepted():
    assert validate_svg(GOOD) == (True, None)
    assert validate_svg('<svg><defs><linearGradient id="a"/></defs><path fill="url(#a)"/></svg>')[0]
    assert extract_svg(f"```svg\n{GOOD}\n```") == GOOD


@pytest.mark.parametrize(
    "svg",
    [
        '<svg><!-- Background --><rect width="512" height="512"/></svg>',
        '<?xml version="1.0" encoding="UTF-8"?><svg><!-- Body --><circle r="10"/></svg>',
        "<!-- Drawing --><svg/><!-- End -->",
    ],
)
def test_comments_and_xml_header_are_accepted(svg):
    assert validate_svg(svg) == (True, None)


def test_model_channel_wrapper_is_removed_before_svg_validation():
    svg = '<svg><!-- The Bufo Frog --><circle r="10"/></svg>'
    output = f"<|channel>thought\n<channel|>{svg}"
    assert extract_svg(output) == svg
    assert validate_svg(extract_svg(output)) == (True, None)


def test_extraction_preserves_forbidden_declarations_for_validation():
    svg = '<!DOCTYPE svg [<!ENTITY x "expanded">]><svg>&x;</svg>'
    output = f"<|channel>thought\n<channel|>```svg\n{svg}\n```"
    assert extract_svg(output) == svg
    assert not validate_svg(extract_svg(output))[0]


def test_models_resolve_environment_at_load_time(monkeypatch):
    monkeypatch.setenv("OPEN_MODEL", "test/model")
    monkeypatch.setenv("OPEN_TUNED_MODEL", "")
    backends = load_backends()
    assert backends["open_base"].model == "test/model"
    with pytest.raises(ValueError, match="OPEN_TUNED_MODEL"):
        backends["open_tuned"].require_configured()


def test_manifest_roundtrip_and_unique_files(tmp_path):
    row = BufoGeneration(
        prompt_id="p1",
        prompt="test",
        backend="open_base",
        model="m",
        group_id="g",
        svg=GOOD,
        valid=True,
        latency_s=0.5,
    )
    manifest = Manifest(tmp_path / "manifest.jsonl")
    manifest.append(row)
    assert manifest.read() == [row]
    assert json.loads(manifest.path.read_text())["cost_usd"] is None


class Job:
    def __init__(self):
        self.calls = []

    async def execute(self, entrypoint, payload):
        self.calls.append(payload)
        return {"png_base64": base64.b64encode(b"png").decode(), "valid": True, "error": None}

    async def close(self):
        pass


async def test_render_rejects_unsafe_before_remote_execution_and_reuses_result():
    job = Job()
    renderer = Renderer(job)
    assert not (await renderer.render("<svg><script/></svg>")).valid
    assert not job.calls
    assert (await renderer.render(GOOD)).png_bytes == b"png"
    assert (await renderer.render(GOOD)).valid
    assert len(job.calls) == 1


@pytest.mark.parametrize("dimension", [".", "1..2", "NaN", "inf", "-10", "0", "1e999"])
def test_malformed_dimensions_return_invalid_instead_of_crashing(dimension):
    valid, error = validate_svg(f'<svg width="{dimension}"/>')
    assert not valid and error
