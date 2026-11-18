"""One bounded serving-schema request, not an image or model-quality evaluation."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys

from agents import function_tool
from bufogen.models import load_backends
from bufogen.settings import ROOT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scene_renderer import Scene


async def render_scene(scene: Scene, emoji_name: str = "") -> str:
    """Compile the complete contact scene, render its PNG, and name the Slack emoji."""
    raise RuntimeError("Schema probe never executes tools")


async def main():
    path = ROOT / "runs/forge-contact-pilot/schema-probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tool = function_tool(render_scene)
    payload = {"type": "function", "function": {
        "name": tool.name, "description": tool.description,
        "parameters": tool.params_json_schema, "strict": True,
    }}
    receipt = {
        "purpose": "Tool-schema serving acceptance only; no image or quality score",
        "schema_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "compiler_sha256": hashlib.sha256((HERE / "scene_renderer.py").read_bytes()).hexdigest(),
        "status": "started", "max_tokens": 32,
    }
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    backend = load_backends()["open_base"]
    client = backend.client()
    try:
        response = await client.chat.completions.create(
            model=backend.model,
            messages=[{"role": "user", "content": "Draw Bufo holding a blue mug."}],
            tools=[payload],
            tool_choice={"type": "function", "function": {"name": tool.name}},
            temperature=0,
            max_tokens=32,
            extra_body=backend.request_extra_body("contact-pilot-schema-probe"),
        )
        receipt.update(status="accepted", model=response.model,
                       finish_reason=response.choices[0].finish_reason,
                       usage=response.usage.model_dump() if response.usage else None)
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        await client.close()
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
