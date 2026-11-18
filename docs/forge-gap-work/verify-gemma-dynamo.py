"""Local-only verification of the untouched two-exchange capability capture."""
import json
import os
import sys
from pathlib import Path

if os.getenv("DIAG_TRANSFORMERS_PATH"):
    sys.path.insert(0, os.environ["DIAG_TRANSFORMERS_PATH"])

if os.getenv("DIAG_ART_SOURCE_PATH"):
    sys.path.insert(0, os.environ["DIAG_ART_SOURCE_PATH"])
import art
from art.trajectories._tokenize import _chat_choice_tokens
from transformers import AutoTokenizer

directory = Path("runs/forge-gap/rl-probe")
trajectory = art.Trajectory.model_validate_json(
    (directory / "two-exchange-probe.json").read_text()
)
tokenizer = AutoTokenizer.from_pretrained(
    os.environ.get("DIAG_TOKENIZER_PATH", "/private/tmp/bufo-gemma4-tokenizer")
)
result = {"exchanges": []}
for exchange in trajectory.exchanges.chat_completions:
    choice = exchange.response.choices[0]
    prompt, output, logprobs = _chat_choice_tokens(choice, exchange.response)
    result["exchanges"].append({
        "prompt_exact_count": None if prompt is None else len(prompt),
        "output_exact_count": None if output is None else len(output),
        "logprobs_count": len(logprobs),
        "finish_reason": choice.finish_reason,
        "content_length": len(choice.message.content or ""),
        "tool_calls": len(choice.message.tool_calls or []),
        "request_keys": list(exchange.request),
    })
try:
    value = trajectory.tokenize(
        multi_history=True,
        base_model="google/gemma-4-26B-A4B-it",
        tokenizer=tokenizer,
    )
    result.update(status="ok", histories=len(value.histories))
except Exception as error:
    result.update(status="failed", error_type=type(error).__name__, error=str(error))
(directory / "two-exchange-probe-tokenization.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
