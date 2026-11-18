"""One render_svg tool-call probe per candidate student on W&B inference (Stage B, Step 6).

At most one retry per model, only after an error. Outcomes record class names and HTTP
status only, never response bodies or credentials.
"""

import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from bufogen.models import load_backends
from bufogen.settings import ROOT

MODELS = ["google/gemma-4-31B-it", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-27B"]
OUTPUT = ROOT / "runs/distill/stage-b/student-probe.json"
# Mirrors the agent's render_svg(svg: str, emoji_name: str = "") function tool.
RENDER_TOOL = {
    "type": "function",
    "function": {
        "name": "render_svg",
        "description": "Render the SVG and name it with a Slack shortcode, e.g. :bufo-gpu-gardener:.",
        "parameters": {
            "type": "object",
            "properties": {"svg": {"type": "string"}, "emoji_name": {"type": "string"}},
            "required": ["svg"],
        },
    },
}
MESSAGES = [
    {
        "role": "system",
        "content": "You draw Slack emoji. Create the drawing by calling render_svg with one "
        "complete SVG under 2000 characters.",
    },
    {"role": "user", "content": "Draw a small green frog holding a red balloon."},
]


def well_formed(message):
    calls = message.tool_calls or []
    if len(calls) < 1 or calls[0].function.name != "render_svg":
        return False
    try:
        arguments = json.loads(calls[0].function.arguments)
    except ValueError:
        return False
    return isinstance(arguments.get("svg"), str) and "<svg" in arguments["svg"]


async def probe_once(config):
    client = config.client().with_options(max_retries=0)
    start = time.monotonic()
    try:
        response = await client.chat.completions.create(
            model=config.model,
            messages=MESSAGES,
            tools=[RENDER_TOOL],
            tool_choice="auto",
            max_tokens=2500,
            extra_body=config.thinking_extra_body(True),
        )
    except Exception as error:
        return {
            "tool_call_emitted": False,
            "error_class": type(error).__name__,
            "http_status": getattr(error, "status_code", None),
            "latency_s": round(time.monotonic() - start, 2),
        }
    message = response.choices[0].message
    return {
        "tool_call_emitted": well_formed(message),
        "error_class": None,
        "http_status": None,
        "finish_reason": response.choices[0].finish_reason,
        "latency_s": round(time.monotonic() - start, 2),
    }


async def main(models=MODELS, output=OUTPUT, append=False):
    load_dotenv(ROOT / ".env")
    base = load_backends()["open_base"]
    results = []
    for model in models:
        config = replace(base, name="probe", model=model, model_env="PROBE_MODEL")
        result = await probe_once(config)
        attempts = 1
        if result["error_class"] is not None:  # a single retry, errors only
            result = await probe_once(config)
            attempts = 2
        results.append({"model": model, "attempts": attempts, **result})
        print(json.dumps(results[-1]))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if append and Path(output).exists():  # keep the existing rows, add the new ones
        results = [*json.loads(Path(output).read_text()), *results]
    Path(output).write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    # No arguments: the original three models. With model IDs: probe only those, appending rows.
    asyncio.run(main(sys.argv[1:], append=True) if sys.argv[1:] else main())
