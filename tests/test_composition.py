import json
from types import SimpleNamespace

from bufogen.agents.agent import BufoAgent
from bufogen.agents.prompting import assemble_character
from bufogen.models import load_backends
from bufogen.rendering.composition import framing_warnings
from bufogen.rendering.renderer import RenderResult


def scene(y):
    return f'<svg viewBox="0 0 512 512"><g transform="translate(60 {y}) scale(1.8)"><use href="#bufo-fullbody"/></g></svg>'


def test_measured_bounds_catch_real_cropped_presenter_geometry():
    warning = framing_warnings(assemble_character(scene(260)), "Bufo presenting")
    assert len(warning) == 1 and "576.8" in warning[0]
    assert framing_warnings(assemble_character(scene(175)), "Bufo presenting") == []
    assert framing_warnings(scene(260), "A close-up portrait of Bufo") == []


async def test_agent_retries_cropped_but_valid_image_then_keeps_corrected_generation(tmp_path):
    feedback = []

    class Renderer:
        async def render(self, svg):
            return RenderResult(b"PNG", True, None)

    async def run(messages, group, render):
        feedback.append(json.loads(await render(scene(260 if not feedback else 175))))
        return SimpleNamespace(
            final_output="Here you go!",
            context_wrapper=SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=1)),
            to_input_list=lambda: messages,
        )

    row = await BufoAgent(load_backends()["open_base"], Renderer(), tmp_path, run=run).generate(
        "Bufo presenting"
    )
    assert row.valid and row.png_path
    assert len(feedback) == 2 and feedback[0]["composition_warnings"]
    assert feedback[1]["composition_warnings"] == row.composition_warnings == []
