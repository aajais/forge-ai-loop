import pytest
from fastapi.testclient import TestClient

from bufogen.app.ui import create_app
from bufogen.generation import BufoGeneration


@pytest.fixture(autouse=True)
def studio_credentials(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def test_legacy_reload_import_serves_current_studio(tmp_path):
    from uvicorn.importer import import_from_string

    factory = import_from_string("bufogen.ui:create_app")
    with TestClient(factory(output_dir=tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/compare.html").status_code == 200
        assert client.get("/api/history").json() == []


def test_model_switch_routes_generation_and_keeps_conversation(tmp_path, monkeypatch):
    from bufogen.app import ui

    calls = []

    class Agent:
        def __init__(self, backend, renderer, output_dir, *, studio_mode=False, max_turns=100):
            self.backend = backend

        async def generate(self, prompt, *, group_id, history):
            calls.append((self.backend.name, prompt, history))
            return BufoGeneration(
                prompt_id="p",
                prompt=prompt,
                backend=self.backend.name,
                model=self.backend.model,
                group_id=group_id or "shared-conversation",
                svg="<svg/>",
                valid=False,
                latency_s=1,
                response_text="Drawing reply",
            )

    monkeypatch.setattr(ui, "BufoAgent", Agent)
    monkeypatch.setenv("OPEN_TUNED_MODEL", "")
    monkeypatch.setenv("OPEN_RL_MODEL", "")
    with TestClient(create_app(output_dir=tmp_path)) as client:
        first = client.post("/api/generate", json={"prompt": "Bufo on a cloud"}).json()
        second = client.post(
            "/api/generate",
            json={
                "prompt": "Add a blue scarf",
                "previous_id": first["id"],
                "backend": "closed_quality_bar",
            },
        ).json()
        third = client.post(
            "/api/generate",
            json={
                "prompt": "Make the scarf red",
                "previous_id": second["id"],
            },
        ).json()
        assert [call[0] for call in calls] == [
            "open_base",
            "closed_quality_bar",
            "closed_quality_bar",
        ]
        assert first["group_id"] == second["group_id"] == third["group_id"]
        assert second["parent_id"] == first["id"]
        assert [r["prompt"] for r in calls[-1][2]] == ["Bufo on a cloud", "Add a blue scarf"]
        assert [r["backend"] for r in client.get("/api/history").json()] == [
            "open_base",
            "closed_quality_bar",
            "closed_quality_bar",
        ]
        assert (
            client.post("/api/generate", json={"prompt": "Bufo", "backend": "unknown"}).status_code
            == 422
        )
        assert (
            client.post(
                "/api/generate", json={"prompt": "Bufo", "backend": "open_tuned"}
            ).status_code
            == 409
        )
        assert len(calls) == 3


def test_model_availability_checks_provider_and_shared_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_TUNED_MODEL", "")
    monkeypatch.setenv("OPEN_RL_MODEL", "")
    monkeypatch.delenv("OPENAI_API_KEY")
    with TestClient(create_app(output_dir=tmp_path)) as client:
        response = client.get("/api/status")
        models = {m["id"]: m for m in response.json()["models"]}
        assert models["open_base"]["available"]
        assert not models["open_tuned"]["available"]
        assert not models["open_rl"]["available"]
        assert "OPEN_TUNED_MODEL" in models["open_tuned"]["unavailable_reason"]
        assert not models["closed_quality_bar"]["available"]
        assert "test-wandb-key" not in response.text
        assert (
            client.post(
                "/api/generate", json={"prompt": "Bufo", "backend": "closed_quality_bar"}
            ).status_code
            == 409
        )
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("OPEN_TUNED_MODEL", "test-tuned-model")
        monkeypatch.setenv("OPEN_RL_MODEL", "test-rl-model")
        monkeypatch.setenv("WANDB_ENTITY", "test-team")
        assert all(
            m["available"]
            for m in client.get("/api/status").json()["models"]
            if m["id"] not in {"rl_q3", "rl_n3"}
        )
        monkeypatch.delenv("WANDB_API_KEY")
        assert not any(m["available"] for m in client.get("/api/status").json()["models"])


def test_studio_loads_without_network_and_does_not_leak_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret-test-key")
    with TestClient(create_app(output_dir=tmp_path)) as client:
        response = client.get("/")
        assert response.status_code == 200 and "Bufo studio" in response.text
        status = client.get("/api/status")
        assert status.status_code == 200
        assert "secret-test-key" not in status.text
        assert client.post("/api/generate", json={"prompt": ""}).status_code == 422
        assert (
            client.post(
                "/api/generate", json={"prompt": "frog"}, headers={"Origin": "https://evil.example"}
            ).status_code
            == 403
        )
        assert client.get("/api/files/../../.env").status_code in {400, 404}


def test_generation_preview_download_and_refinement_share_pipeline(tmp_path):
    calls = []

    async def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        row = BufoGeneration(
            prompt_id="p",
            prompt=prompt,
            backend="open_base",
            model="gemma",
            group_id=kwargs.get("group_id") or "conversation1",
            svg="<svg/>",
            valid=True,
            latency_s=1,
            response_text="Here you go—a bufo with a GPU!",
        )
        png = tmp_path / f"{row.id}.png"
        png.write_bytes(b"png")
        svg = tmp_path / f"{row.id}.svg"
        svg.write_text(row.svg)
        row.png_path, row.svg_path = str(png), str(svg)
        return row

    with TestClient(create_app(output_dir=tmp_path, generate=generate)) as client:
        first = client.post("/api/generate", json={"prompt": "frog with GPU"}).json()
        assert first["valid"]
        assert first["response_text"] == "Here you go—a bufo with a GPU!"
        assert "svg" not in first
        assert client.get(first["png_url"]).content == b"png"
        download = client.get(first["svg_url"])
        assert "attachment" in download.headers["content-disposition"]
        second = client.post(
            "/api/generate", json={"prompt": "add a hat", "previous_id": first["id"]}
        )
        assert second.status_code == 200
        assert calls[1][1]["group_id"] == "conversation1"
        assert calls[1][0] == "add a hat"
        assert calls[1][1]["history"][0]["svg"] == "<svg/>"
        assert calls[1][1]["history"][0]["prompt"] == "frog with GPU"
        assert second.json()["parent_id"] == first["id"]
        third = client.post(
            "/api/generate",
            json={"prompt": "Make that hat blue", "previous_id": second.json()["id"]},
        )
        assert third.status_code == 200
        assert [r["prompt"] for r in calls[2][1]["history"]] == ["frog with GPU", "add a hat"]
        fresh = client.post("/api/generate", json={"prompt": "A fresh scene"})
        assert fresh.status_code == 200 and calls[3][1]["history"] == []
        assert len(client.get("/api/history").json()) == 4


def test_generation_failure_is_actionable_and_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "very-secret")

    async def fail(*args, **kwargs):
        raise RuntimeError("Connection failed: very-secret")

    with TestClient(create_app(output_dir=tmp_path, generate=fail)) as client:
        result = client.post("/api/generate", json={"prompt": "frog"})
        assert result.status_code == 502
        assert "very-secret" not in result.text


async def test_independent_conversations_run_concurrently_but_same_thread_is_guarded(tmp_path):
    import asyncio

    import httpx

    from bufogen.generation import Manifest

    seed = BufoGeneration(
        prompt_id="seed",
        prompt="first",
        backend="open_base",
        model="gemma",
        group_id="existing",
        svg="<svg/>",
        valid=False,
        latency_s=0,
    )
    Manifest(tmp_path / "manifest.jsonl").append(seed)
    started = asyncio.Queue()
    release = asyncio.Event()

    async def generate(prompt, **kwargs):
        await started.put(prompt)
        await release.wait()
        return BufoGeneration(
            prompt_id=prompt,
            prompt=prompt,
            backend="open_base",
            model="gemma",
            group_id=kwargs.get("group_id") or prompt,
            svg="<svg/>",
            valid=False,
            latency_s=0,
        )

    app = create_app(output_dir=tmp_path, generate=generate)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        first = asyncio.create_task(
            client.post("/api/generate", json={"prompt": "slow", "previous_id": seed.id})
        )
        second = None
        try:
            assert await asyncio.wait_for(started.get(), 1) == "slow"
            duplicate = await client.post(
                "/api/generate", json={"prompt": "duplicate", "previous_id": seed.id}
            )
            assert duplicate.status_code == 409
            second = asyncio.create_task(
                client.post("/api/generate", json={"prompt": "independent"})
            )
            assert await asyncio.wait_for(started.get(), 1) == "independent"
            assert not first.done()
        finally:
            release.set()
            results = await asyncio.gather(*[task for task in [first, second] if task])
        assert all(r.status_code == 200 for r in results)
        assert {r.json()["prompt"] for r in results} == {"slow", "independent"}
        assert not (await client.get("/api/status")).json()["busy"]


def test_reload_watches_all_packages_and_resolves_app_factory(monkeypatch):
    import importlib
    import sys
    from pathlib import Path

    import uvicorn

    from bufogen.app import ui

    calls = []
    monkeypatch.setattr(sys, "argv", ["bufo-studio", "--reload", "--port", "8766"])
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append((app, options)))
    ui.main()
    target, options = calls[0]
    module_name, factory = target.split(":")
    assert getattr(importlib.import_module(module_name), factory) is ui.create_app
    assert options["factory"] and options["reload"]
    assert options["port"] == 8766
    package_root = Path(ui.__file__).resolve().parents[1]
    assert options["reload_dirs"] == [str(package_root)]
    assert (package_root / "agents/agent.py").is_file()


def test_demo_rl_default_resolves_to_evaluated_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("RL_Q3_MODEL", raising=False)
    with TestClient(create_app(output_dir=tmp_path)) as client:
        model = next(m for m in client.get("/api/status").json()["models"] if m["id"] == "rl_q3")
        assert model["available"]
        assert (
            model["model"] == "wandb-artifact:///wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v6"
        )
