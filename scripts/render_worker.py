"""Runs inside the CPU sandbox. JSON input on stdin, result in a file (or stdout)."""

import base64
import json
import resource
import sys
from pathlib import Path

from svg import validate_svg


def main(output_path=None):
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    request = json.loads(sys.stdin.read(150_000))
    valid, error = validate_svg(request["svg"])
    result = {"valid": valid, "error": error, "png_base64": None}
    if valid:
        try:
            import resvg_py

            font_path = Path(__file__).with_name("DejaVuSans.ttf")
            if not font_path.is_file():
                raise ValueError("Renderer font is missing from the sandbox.")
            png = resvg_py.svg_to_bytes(
                svg_string=request["svg"],
                width=512,
                height=512,
                skip_system_fonts=True,
                font_files=[str(font_path)],
                font_family="DejaVu Sans",
                sans_serif_family="DejaVu Sans",
                serif_family="DejaVu Sans",
                monospace_family="DejaVu Sans",
            )
            result["png_base64"] = base64.b64encode(png).decode()
        except Exception as exc:
            result.update(valid=False, error=f"Render failed: {exc}")
    if output_path:
        Path(output_path).write_text(json.dumps(result))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
