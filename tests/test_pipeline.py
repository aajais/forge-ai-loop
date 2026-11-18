from types import SimpleNamespace

import pytest

from bufogen.agents.agent import BufoAgent
from bufogen.generation import BufoGeneration
from bufogen.models import load_backends
from bufogen.rendering.renderer import RenderResult
from bufogen.workflows.pipeline import write_demo

GOOD = '<svg><circle r="10"/></svg>'


class Renderer:
    async def render(self, svg):
        return RenderResult(
            b"PNG" if svg == GOOD else None, svg == GOOD, None if svg == GOOD else "invalid"
        )


class Runner:
    def __init__(self):
        self.calls = []

    async def __call__(self, prompt, group_id, render):
        self.calls.append((prompt, group_id))
        await render(GOOD, ":Bufo-GPU-Gardener:")
        return SimpleNamespace(
            final_output="Here you go—Bufo with a GPU!",
            context_wrapper=SimpleNamespace(usage=SimpleNamespace(input_tokens=5, output_tokens=9)),
            new_items=[],
            to_input_list=lambda: [],
        )


async def test_agent_persists_render_and_preserves_conversation(tmp_path):
    runner = Runner()
    agent = BufoAgent(load_backends()["open_base"], Renderer(), tmp_path, run=runner)
    result = await agent.generate("A bufo with a GPU", prompt_id="p1", group_id="same-conversation")
    assert result.valid and result.input_tokens == 5
    assert result.group_id == "same-conversation"
    assert result.png_path and (tmp_path / f"{result.id}.png").exists()
    assert result.response_text == "Here you go—Bufo with a GPU!"
    assert result.emoji_name == ":bufo-gpu-gardener:"
    assert result.svg == GOOD and result.tool_calls == 1
    assert runner.calls[0][1] == "same-conversation"


async def test_invalid_tool_render_gets_one_repair_attempt(tmp_path):
    runner = Runner()
    original = runner.__call__
    calls = []

    async def bad_then_good(prompt, group_id, render):
        calls.append(group_id)
        r = await original(prompt, group_id, render)
        if len(calls) == 1:
            await render("<svg>")
        return r

    backend = load_backends()["open_base"]
    result = await BufoAgent(backend, Renderer(), tmp_path, run=bad_then_good).generate("frog")
    assert result.valid and len(calls) == 2 and calls[0] == calls[1]
    assert result.input_tokens == 10


async def test_backend_without_tools_fails_explicitly(tmp_path):
    from dataclasses import replace

    backend = replace(load_backends()["open_base"], supports_tools=False)
    with pytest.raises(ValueError, match="tool support"):
        await BufoAgent(backend, Renderer(), tmp_path, run=Runner()).generate("frog")


def test_studio_screenshot_reaches_agent_and_survives_reload_and_followup(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from bufogen.agents.prompting import style_references
    from bufogen.app.ui import create_app
    from bufogen.rendering.image_input import image_data_url

    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    runner = Runner()
    agent = BufoAgent(load_backends()["open_base"], Renderer(), tmp_path, run=runner)
    image = image_data_url(str(style_references()[0]))
    with TestClient(create_app(output_dir=tmp_path, generate=agent.generate)) as client:
        response = client.post(
            "/api/generate", json={"prompt": "Use this reference", "image": image}
        )
        assert response.status_code == 200
        row = response.json()
        assert row["emoji_name"] == ":bufo-gpu-gardener:"
        assert row["input_image_url"] and "input_image_path" not in row
        assert client.get(row["input_image_url"]).content.startswith(b"\x89PNG\r\n\x1a\n")
        assert runner.calls[0][0][-1]["content"][1]["type"] == "input_image"
        supplied = runner.calls[0][0][-1]["content"][1]["image_url"]
        assert supplied.startswith("data:image/png;base64,")
        assert (
            client.post("/api/generate", json={"prompt": "Draw", "image": "bad"}).status_code == 422
        )
        assert len(runner.calls) == 1
    with TestClient(create_app(output_dir=tmp_path, generate=agent.generate)) as client:
        assert client.get("/api/history").json()[0]["input_image_url"] == row["input_image_url"]
        followup = client.post(
            "/api/generate", json={"prompt": "Add a scarf", "previous_id": row["id"]}
        )
        assert followup.status_code == 200
        users = [m for m in runner.calls[1][0] if m.get("role") == "user"]
        assert users[-2]["content"][1]["image_url"] == supplied
        assert users[-1]["content"] == "Add a scarf"


def test_demo_escapes_prompt_and_never_embeds_model_svg(tmp_path):
    row = BufoGeneration(
        prompt_id="p",
        prompt="<script>bad</script>",
        backend="b",
        model="m",
        group_id="g",
        svg='<svg onload="bad"/>',
        valid=False,
        latency_s=1,
    )
    path = write_demo([row], tmp_path / "demo.html")
    html = path.read_text()
    assert "<svg onload" not in html
    assert "<script>bad" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize(
    "backend_name,disable,expected",
    [
        ("open_base", True, {"chat_template_kwargs": {"enable_thinking": False}}),
        ("open_base", False, None),
        ("closed_quality_bar", True, None),
        ("forge_proxy", True, None),
    ],
)
def test_thinking_body_only_for_coreweave(backend_name, disable, expected):
    assert load_backends()[backend_name].thinking_extra_body(disable) == expected


async def test_studio_exposes_render_preview_before_final_reply_and_recovers_result(
    tmp_path, monkeypatch
):
    import asyncio

    import httpx

    from bufogen.app.ui import create_app

    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    rendered = asyncio.Event()
    finish = asyncio.Event()

    async def run(prompt, group_id, render):
        await render(GOOD)
        rendered.set()
        await finish.wait()
        return SimpleNamespace(
            final_output="Done",
            context_wrapper=SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=1)),
            new_items=[],
            to_input_list=lambda: [],
        )

    agent = BufoAgent(load_backends()["qwen_base"], Renderer(), tmp_path, run=run)
    app = create_app(output_dir=tmp_path, generate=agent.generate)
    request_id = "a" * 32
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        task = asyncio.create_task(
            client.post(
                "/api/generate",
                json={"prompt": "frog", "backend": "qwen_base", "request_id": request_id},
            )
        )
        try:
            await asyncio.wait_for(rendered.wait(), 5)
            progress = await client.get(f"/api/generations/{request_id}")
            assert progress.status_code == 200
            preview = progress.json()
            assert preview["status"] == "rendered"
            assert (await client.get(preview["row"]["png_url"])).content == b"PNG"
            assert not task.done()
        finally:
            finish.set()
            result = await task
        assert result.status_code == 200
        complete = (await client.get(f"/api/generations/{request_id}")).json()
        assert complete["status"] == "complete"
        assert complete["row"]["id"] == result.json()["id"]
    restarted = create_app(output_dir=tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=restarted), base_url="http://testserver"
    ) as client:
        recovered = (await client.get(f"/api/generations/{request_id}")).json()
        assert recovered["row"]["id"] == result.json()["id"]
