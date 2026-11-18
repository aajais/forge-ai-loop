"""Capture SDK actions without reconstructing tool calls from rendered artifacts.

Version 1 exports text and function tools only. Multimodal SDK inputs are retained
for continuation, with an explicit diagnostic, until a training image contract exists.
"""

import json
import re
from copy import deepcopy

from bufogen.agents.prompting import editable_scene, user_content
from bufogen.generation import BufoGeneration

SCHEMA_VERSION = 1
CAPTURE_SOURCE = "agents.RunResult.to_input_list"
CONTEXT_POLICY = "authentic-cumulative-v1"


def tool_schemas(function_tools) -> list[dict]:
    """Freeze the same SDK schemas used by the running agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.params_json_schema),
                "strict": tool.strict_json_schema,
            },
        }
        for tool in function_tools
    ]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("trajectory contains unsupported message content")
    parts = []
    for part in content:
        if part.get("type") not in {"input_text", "output_text", "text"}:
            raise ValueError("trajectory has unsupported multimodal or non-text content")
        if not isinstance(part.get("text"), str):
            raise ValueError("trajectory contains non-text content")
        parts.append(part["text"])
    return "".join(parts)


def portable_messages(items: list[dict]) -> list[dict]:
    messages = []
    for item in items:
        kind = item.get("type", "message")
        if kind == "reasoning":
            continue
        if kind == "function_call":
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }
            # Parallel calls must precede their results in one assistant message.
            if messages and messages[-1].get("tool_calls"):
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            messages.append(
                {"role": "tool", "tool_call_id": item["call_id"], "content": _text(item["output"])}
            )
        elif kind == "message":
            role = item.get("role")
            if role not in {"user", "assistant"}:
                raise ValueError("trajectory contains unsupported message role")
            messages.append({"role": role, "content": _text(item.get("content"))})
        else:
            raise ValueError(f"trajectory contains unsupported SDK item type: {kind}")
    return messages


def response_items(messages: list[dict]) -> list[dict]:
    """Restore portable history to SDK input, retaining actual call identifiers."""
    items = []
    for message in messages:
        if message["role"] == "system":
            continue
        if message.get("tool_calls"):
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for call in message["tool_calls"]:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
        elif message["role"] == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        else:
            items.append(deepcopy(message))
    return items


def conversation_input(prompt: str, *, history=None, image=None) -> tuple[list[dict], bool]:
    """Use the latest cumulative capture, or explicit non-action legacy context."""
    messages = []
    authentic = True
    if history:
        latest = history[-1].get("trajectory")
        if latest and latest.get("schema_version") == SCHEMA_VERSION:
            if latest.get("source_items") is not None:
                messages = deepcopy(latest["source_items"])
            else:
                messages = response_items(latest["messages"])
            authentic = latest.get("provenance", {}).get("history_authentic") is True
        else:
            authentic = False
            for previous in history:
                messages.append(
                    {
                        "role": "user",
                        "content": user_content(previous["prompt"], previous.get("image")),
                    }
                )
                if previous.get("response_text"):
                    messages.append({"role": "assistant", "content": previous["response_text"]})
            # This is supplied context, never an invented assistant action/result.
            if history[-1].get("svg"):
                messages.append(
                    {
                        "role": "user",
                        "content": "Application context: the previous editable drawing is:\n"
                        + editable_scene(history[-1]["svg"]),
                    }
                )
    messages.append({"role": "user", "content": user_content(prompt, image)})
    return messages, authentic


def capture_trajectory(
    *,
    items,
    system_instructions,
    tools,
    provenance,
    final_output,
    history_authentic=True,
    expected_input=None,
    history_snapshot=None,
) -> dict:
    # Never persist hidden reasoning or encrypted reasoning payloads.
    items = deepcopy([item for item in items if item.get("type") != "reasoning"])
    trajectory = {
        "schema_version": SCHEMA_VERSION,
        "system_instructions": system_instructions,
        "tools": deepcopy(tools),
        "messages": [],
        "provenance": {
            **deepcopy(provenance),
            "capture_source": CAPTURE_SOURCE,
            "context_policy": CONTEXT_POLICY,
            "history_authentic": history_authentic,
            "final_output": final_output,
        },
        "diagnostics": [],
    }
    previous_provenance = (history_snapshot or {}).get("provenance", {})
    source_models = list(previous_provenance.get("source_models", []))
    for model in (previous_provenance.get("model"), provenance.get("model")):
        if model and model not in source_models:
            source_models.append(model)
    trajectory["provenance"]["source_models"] = source_models
    if len(source_models) > 1:
        trajectory["diagnostics"].append("trajectory contains mixed-model history")
    compatible = not history_snapshot or (
        history_snapshot.get("system_instructions") == system_instructions
        and history_snapshot.get("tools") == tools
        and previous_provenance.get("prompt_version") == provenance.get("prompt_version")
        and previous_provenance.get("history_snapshot_compatible", True)
    )
    trajectory["provenance"]["history_snapshot_compatible"] = compatible
    if not compatible:
        trajectory["diagnostics"].append(
            "trajectory history snapshot differs from current settings"
        )
    # A review of the latest PNG does not qualify every older assistant target.
    # Until per-turn target masking/review lineage is supported, keep follow-ups
    # usable in chat but train only single-request captures (including repairs).
    trajectory["provenance"]["prior_conversation"] = bool(history_snapshot)
    if history_snapshot:
        trajectory["diagnostics"].append(
            "trajectory has prior conversation targets without per-turn review qualification"
        )
    if expected_input is not None:
        expected = [item for item in expected_input if item.get("type") != "reasoning"]
        if items[: len(expected)] != expected or len(items) <= len(expected):
            trajectory["diagnostics"].append(
                "incomplete SDK trajectory: submitted input or output missing"
            )
            trajectory["source_items"] = deepcopy(expected)
            trajectory["provenance"]["history_authentic"] = False
            return trajectory
    try:
        trajectory["messages"] = [
            {"role": "system", "content": system_instructions},
            *portable_messages(items),
        ]
    except (ValueError, KeyError, TypeError) as exc:
        # Preserve attachments for normal chat continuation; never silently drop them.
        trajectory["source_items"] = items
        trajectory["diagnostics"].append(str(exc))
    if not history_authentic:
        trajectory["diagnostics"].append("trajectory contains unauthenticated legacy history")
    return trajectory


def _validate_messages(messages, tools):
    names = set()
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name")
        if (
            tool.get("type") != "function"
            or not name
            or name in names
            or not isinstance(function.get("parameters"), dict)
        ):
            raise ValueError("trajectory has invalid or duplicate tool schemas")
        names.add(name)
    if "render_svg" not in names:
        raise ValueError("trajectory is missing render_svg schema")
    pending, seen = {}, set()
    last_render = None
    user_seen = False
    for message in messages[1:]:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("trajectory has orphan tool result")
            name = pending.pop(call_id)
            content = _text(message.get("content"))
            if name == "render_svg":
                try:
                    last_render = json.loads(content)
                except ValueError:
                    # Failed tool invocations can return plain text before repair.
                    last_render = {"valid": False}
        elif role in {"user", "assistant"}:
            if pending:
                raise ValueError("trajectory has unresolved tool calls before next message")
            if role == "user":
                user_seen = True
            content = message.get("content")
            if content is not None:
                content = _text(content)
                if role == "assistant" and "<svg" in content.lower():
                    raise ValueError("trajectory contains assistant raw SVG")
            calls = message.get("tool_calls", [])
            if calls and role != "assistant":
                raise ValueError("trajectory tool calls must be assistant actions")
            for call in calls:
                name = call.get("function", {}).get("name")
                if name not in names:
                    raise ValueError("trajectory refers to unknown tool")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in seen:
                    raise ValueError("trajectory has missing or duplicate tool call IDs")
                if call.get("type") != "function":
                    raise ValueError("trajectory contains unsupported tool call")
                try:
                    arguments = json.loads(call["function"]["arguments"])
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError("trajectory has invalid tool arguments") from exc
                if not isinstance(arguments, dict):
                    raise ValueError("trajectory tool arguments must be an object")
                if name == "render_svg":
                    svg = arguments.get("svg")
                    if not isinstance(svg, str) or not svg.strip() or len(svg) > 16000:
                        raise ValueError("trajectory requires compact editable SVG arguments")
                    if re.search(r"\bid\s*=\s*[\"\'](?:bufo-character|bufo-fullbody)[\"\']", svg):
                        raise ValueError("trajectory contains expanded canonical SVG definitions")
                pending[call_id] = name
                seen.add(call_id)
        else:
            raise ValueError("trajectory contains unsupported message role")
    if pending:
        raise ValueError("trajectory has unresolved tool calls")
    if (
        not isinstance(last_render, dict)
        or last_render.get("valid") is not True
        or last_render.get("composition_warnings")
    ):
        raise ValueError("trajectory has no successful final render")
    if not user_seen or messages[-1].get("role") != "assistant" or not messages[-1].get("content"):
        raise ValueError("trajectory is missing user input or final assistant reply")


def export_training_example(row: BufoGeneration) -> dict:
    """Return authentic tool SFT data; review/split qualification happens in curate."""
    trajectory = row.trajectory
    if not trajectory or trajectory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("missing or unsupported trajectory")
    if trajectory.get("diagnostics"):
        raise ValueError("; ".join(trajectory["diagnostics"]))
    provenance = trajectory.get("provenance", {})
    if (
        provenance.get("capture_source") != CAPTURE_SOURCE
        or provenance.get("history_authentic") is not True
    ):
        raise ValueError("trajectory has unauthenticated provenance")
    if provenance.get("prior_conversation"):
        raise ValueError("trajectory has unqualified prior conversation targets")
    if (
        provenance.get("source_models") != [row.model]
        or provenance.get("history_snapshot_compatible") is not True
    ):
        raise ValueError("trajectory has inconsistent model or history snapshot provenance")
    for field in ("model", "prompt_id", "group_id", "prompt_version"):
        if provenance.get(field) != getattr(row, field) or not provenance.get(field):
            raise ValueError(f"trajectory provenance mismatch: {field}")
    messages = trajectory.get("messages", [])
    if (
        not trajectory.get("system_instructions")
        or not messages
        or messages[0] != {"role": "system", "content": trajectory["system_instructions"]}
    ):
        raise ValueError("trajectory has missing or inconsistent frozen system instructions")
    if not row.valid or row.composition_warnings:
        raise ValueError(
            "trajectory generation is not qualified: render failed or composition warnings"
        )
    tools = trajectory.get("tools", [])
    _validate_messages(messages, tools)
    if messages[-1]["content"] != provenance.get("final_output"):
        raise ValueError("trajectory final reply differs from captured model output")
    return deepcopy(
        {
            "messages": messages,
            "tools": tools,
            "prompt_id": row.prompt_id,
            "prompt": row.prompt,
            "theme": row.theme,
            "source_generation_id": row.id,
            "source_group_id": row.group_id,
            "source_model": row.model,
            "prompt_version": row.prompt_version,
            "trajectory_schema_version": SCHEMA_VERSION,
            "provenance": provenance,
        }
    )
