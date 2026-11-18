"""Small direct serving probes; no Weave hooks, generation, or training mutation."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path

from agents import function_tool
from bufogen.agents.agent import instructions
from bufogen.models import load_backends
from bufogen.agents.planning import plan_scene
from bufogen.curation.trajectories import tool_schemas

ROOT = Path(__file__).resolve().parents[2]


def transform(schema, mode):
    if isinstance(schema, list):
        return [transform(item, mode) for item in schema]
    if not isinstance(schema, dict):
        return schema
    omitted = {
        "no_pattern": {"pattern"},
        "no_bounds": {"minimum", "maximum", "exclusiveMinimum", "maxLength", "minLength", "minItems", "maxItems", "pattern"},
    }.get(mode, set())
    out = {key: transform(value, mode) for key, value in schema.items() if key not in omitted}
    if mode == "enum_const" and "const" in out:
        out["enum"] = [out.pop("const")]
    return out


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="original", choices=["original", "fixed", "no_pattern", "no_bounds", "enum_const"])
    args = parser.parse_args()
    config = load_backends()["open_base"]
    tool = tool_schemas([function_tool(plan_scene)])[0]
    tool["function"]["parameters"] = transform(tool["function"]["parameters"], args.mode)
    record = {"model": config.model, "mode": args.mode, "schema": tool}
    client = config.client()
    client.max_retries = 0
    try:
        response = await client.chat.completions.create(
            model=config.model, messages=[{"role": "system", "content": instructions()},
                {"role": "user", "content": "Draw Bufo holding a blue mug by its handle against a pale background."}],
            tools=[tool], tool_choice="required", temperature=0.7, max_tokens=1200)
        record.update(status="ok", finish_reason=response.choices[0].finish_reason,
                      message=response.choices[0].message.model_dump(), usage=response.usage.model_dump())
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__,
                      status_code=getattr(exc, "status_code", None), body=getattr(exc, "body", None))
    finally:
        await client.close()
    output = ROOT / "runs/forge-iteration2/diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"planner-{args.mode}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k:v for k,v in record.items() if k != "schema"}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
