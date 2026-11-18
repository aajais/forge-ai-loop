import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import student_tool_probe as probe  # noqa: E402


def message(name="render_svg", arguments=None):
    arguments = json.dumps({"svg": "<svg/>"}) if arguments is None else arguments
    call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(tool_calls=[call])


def test_well_formed_requires_render_svg_with_svg_argument():
    assert probe.well_formed(message())
    assert not probe.well_formed(message(name="other"))
    assert not probe.well_formed(message(arguments="{"))
    assert not probe.well_formed(message(arguments=json.dumps({"svg": "hello"})))
    assert not probe.well_formed(SimpleNamespace(tool_calls=None))
