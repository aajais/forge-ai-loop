import base64
from html import escape
from pathlib import Path

import yaml

from bufogen.agents.agent import BufoAgent
from bufogen.generation import BufoGeneration, Manifest
from bufogen.models import generation_model_presentation, load_backends
from bufogen.rendering.renderer import Renderer
from bufogen.settings import ROOT


def load_prompts(path: Path | None = None) -> list[dict]:
    prompts = yaml.safe_load((path or ROOT / "config/prompts_seed.yaml").read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt file must be a nonempty YAML list of id, prompt, theme objects.")
    ids = set()
    for item in prompts:
        if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
            raise ValueError("Each prompt needs id and prompt.")
        if item["id"] in ids:
            raise ValueError(f"Duplicate prompt id: {item['id']}")
        ids.add(item["id"])
    return prompts


async def run_pipeline(
    prompts: list[dict],
    backend_names: list[str],
    output_dir: Path,
    *,
    backends=None,
    max_turns: int = 100,
    max_tokens: int | None = None,
    disable_thinking: bool | None = None,
) -> list[BufoGeneration]:
    from bufogen.integrations.telemetry import init_weave

    configs = backends or load_backends()
    for name in backend_names:
        configs[name].require_configured()
    init_weave()
    renderer = Renderer()
    rows = []
    manifest = Manifest(output_dir / "manifest.jsonl")
    try:
        for prompt in prompts:
            for name in backend_names:
                row = await BufoAgent(
                    configs[name],
                    renderer,
                    output_dir,
                    max_turns=max_turns,
                    max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ).generate(
                    prompt["prompt"], prompt_id=prompt["id"], theme=prompt.get("theme", "general")
                )
                manifest.append(row)
                rows.append(row)
    finally:
        await renderer.close()
    return rows


def write_demo(rows: list[BufoGeneration], path: Path) -> Path:
    cards = []
    for row in rows:
        presentation = generation_model_presentation(row)
        preview = "<p>No valid render</p>"
        if row.valid and row.png_path:
            image = base64.b64encode(Path(row.png_path).read_bytes()).decode()
            preview = f'<img alt="Generated bufo" src="data:image/png;base64,{image}">'
        link = (
            f'<a href="{escape(row.conversation_url, quote=True)}">View conversation</a>'
            if row.conversation_url
            else ""
        )
        cards.append(
            f"<article><h2>{escape(presentation['model_label'])}</h2><p>{escape(presentation['model_version'] or '')}</p>{preview}<p>{escape(row.prompt)}</p><p>{row.latency_s:.1f}s · {row.tool_calls} checks</p>{link}</article>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bufo comparison</title><style>body{font:17px system-ui;background:#edf3fa;color:#142c47;margin:40px}main{display:flex;gap:24px;flex-wrap:wrap}article{flex:1;min-width:240px;background:white;padding:24px;border-radius:18px}img{width:100%;max-width:512px}a{color:#0541e9}</style><h1>One theme. Three perspectives.</h1><main>'
        + "".join(cards)
        + "</main></html>"
    )
    return path
