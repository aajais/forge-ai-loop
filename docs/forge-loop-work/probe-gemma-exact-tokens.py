"""Explicit two-call capability probe; not an agent benchmark or training job.

Run only with authorization: this file makes two paid inference requests.
No registration, checkpoint, training, or capture-rewriting operations.
"""
import asyncio
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import art
from openai import AsyncOpenAI

from bufogen.settings import project_path
from bufogen.settings import required


async def main():
    destination = Path("runs/forge-loop/rl-tokenization-diagnosis/two-exchange-probe.json")
    if destination.exists():
        raise RuntimeError("Refusing to overwrite existing probe")
    model = "wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v1"
    defaults = {
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"preserve_thinking": True},
    }
    tools = [{"type": "function", "function": {
        "name": "validate_svg",
        "description": "Parse the supplied SVG locally and report whether its root is svg.",
        "parameters": {"type": "object", "properties": {"svg": {"type": "string"}}, "required": ["svg"], "additionalProperties": False},
    }}]
    messages = [{"role": "user", "content": "Call validate_svg exactly once with <svg xmlns=\"http://www.w3.org/2000/svg\"/>. After its result, reply only OK."}]
    trajectory = art.Trajectory()
    async with AsyncOpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=required("WANDB_API_KEY"), project=project_path(), max_retries=0, timeout=120) as client:
        with trajectory:
            first = await client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="required", max_tokens=512, logprobs=True, extra_body=defaults)
            messages.append(first.choices[0].message.model_dump(exclude_none=True))
            calls = first.choices[0].message.tool_calls or []
            if len(calls) != 1 or calls[0].function.name != "validate_svg":
                raise RuntimeError("Probe did not produce the one requested tool call")
            call = calls[0]
            args = json.loads(call.function.arguments)
            root = ET.fromstring(args["svg"])
            tool_result = {"valid": root.tag.rsplit("}", 1)[-1] == "svg", "parser": "xml.etree.ElementTree"}
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(tool_result)})
            await client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="none", max_tokens=512, logprobs=True, extra_body=defaults)
    destination.write_text(trajectory.model_dump_json(indent=2))
    os.chmod(destination, 0o600)
    print(json.dumps({"saved": str(destination), "exchanges": len(trajectory.exchanges.chat_completions), "tool_result": tool_result}))


if __name__ == "__main__":
    asyncio.run(main())
