"""Exercise the verified existing RL checkpoint; never submits training."""

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from bufogen.agents.agent import BufoAgent
from bufogen.generation import Manifest
from bufogen.models import load_backends
from bufogen.rendering.renderer import Renderer
from bufogen.integrations.telemetry import init_weave


async def main(out):
    result = json.loads(Path("runs/forge-gap/rl-direct-result.json").read_text())
    assert result["training_completed"] and result["completed_step"] == 2
    out.mkdir(exist_ok=False)
    config = replace(load_backends()["open_rl"], model=result["checkpoint_artifact"])
    renderer = Renderer()
    client = init_weave()
    try:
        row = await BufoAgent(config, renderer, out, web_search_enabled=False).generate(
            "Draw Bufo at a coffee cart, holding a blue mug in one hand. Keep Bufo's whole body visible.",
            prompt_id="rl-serving-coffee-smoke",
        )
        Manifest(out / "manifest.jsonl").append(row)
        receipt = {
            "model_id": config.display_name,
            "checkpoint": config.model,
            "valid": row.valid,
            "png_path": row.png_path,
            "trace": row.conversation_url,
            "reply": row.response_text,
            "scope": "Live serving and rendering verification; not a quality improvement evaluation.",
        }
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        if not row.valid:
            raise RuntimeError("RL checkpoint did not produce a valid image")
    except Exception as error:
        (out / "error.json").write_text(json.dumps({
            "checkpoint": config.model, "error_type": type(error).__name__,
            "http_status": getattr(error, "status_code", None),
        }, indent=2) + "\n")
        raise
    finally:
        await renderer.close()
        client.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/forge-gap/rl-serving-smoke"))
    asyncio.run(main(parser.parse_args().output))
