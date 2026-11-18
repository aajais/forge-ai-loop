"""Local UI using the same baseline agent and manifest as the pipeline."""

import asyncio
import importlib.util
import os
import re
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from bufogen.agents.agent import BufoAgent
from bufogen.generation import Manifest
from bufogen.models import generation_model_presentation, load_backends
from bufogen.rendering.image_input import MAX_IMAGE_DATA_URL, image_data_url, normalize_image
from bufogen.rendering.renderer import Renderer
from bufogen.runtime import resolve_inference_policy
from bufogen.settings import ROOT, project_path


class GenerationRequest(BaseModel):
    request_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    prompt: str = Field(default="", max_length=4000)
    image: str | None = Field(default=None, max_length=MAX_IMAGE_DATA_URL)
    previous_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    backend: str | None = Field(default=None, max_length=100)

    @field_validator("prompt")
    @classmethod
    def nonempty(cls, value):
        return value.strip()

    @field_validator("image")
    @classmethod
    def screenshot(cls, value):
        return normalize_image(value) if value is not None else None

    @model_validator(mode="after")
    def has_content(self):
        if not self.prompt and not self.image:
            raise ValueError("Enter a message or attach an image first.")
        if not self.prompt:
            self.prompt = "Create a Bufo scene inspired by this image."
        return self


def public_error(exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError):
        return "A required package is missing. Run uv sync, then restart Bufo studio."
    text = str(exc)
    for name, value in os.environ.items():
        if any(part in name for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and value:
            text = text.replace(value, "[redacted]")
    return text[:800] or "Generation failed. Check server logs and try again."


def backend_unavailable_reason(backend) -> str | None:
    if not backend.model:
        return f"Set {backend.model_env} in .env to enable this model."
    try:
        backend.require_configured()
        resolve_inference_policy(backend)
    except ValueError as exc:
        return str(exc)
    if not backend.supports_tools:
        return "This model needs render-tool support."
    keys = [backend.api_key_env, "WANDB_API_KEY"]
    if backend.provider == "forge":
        keys.append("WANDB_ENTITY")
    for key in dict.fromkeys(keys):
        if not os.getenv(key, "").strip():
            return f"Set {key} in .env to enable this model."
    return None


def create_app(*, output_dir: Path | None = None, generate=None) -> FastAPI:
    output_dir = output_dir or ROOT / "runs/studio"
    manifest = Manifest(output_dir / "manifest.jsonl")
    renderer = Renderer()
    active_groups = set()
    jobs = {}

    @asynccontextmanager
    async def generation_slot(request):
        previous = (
            next((r for r in manifest.read() if r.id == request.previous_id), None)
            if request.previous_id
            else None
        )
        key = previous.group_id if previous else uuid4().hex
        if key in active_groups:
            raise HTTPException(
                409, "This conversation is already generating. You can start another conversation."
            )
        active_groups.add(key)
        try:
            yield
        finally:
            active_groups.discard(key)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await renderer.close()

    app = FastAPI(title="Bufo studio", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        if (
            request.method == "POST"
            and origin
            and urlsplit(origin).netloc != request.headers.get("host")
        ):
            return JSONResponse(
                {"detail": "Only requests from this studio are allowed."}, status_code=403
            )
        if request.method == "POST":
            body = await request.body()
            if len(body) > MAX_IMAGE_DATA_URL + 30_000:
                return JSONResponse({"detail": "Request is too large."}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    def serialize(row):
        value = row.to_dict()
        config = load_backends().get(row.backend)
        value.update(generation_model_presentation(row, config=asdict(config) if config else None))
        value.pop("png_path", None)
        value.pop("svg_path", None)
        value.pop("svg", None)
        value.pop("input_image_path", None)
        value["input_image_url"] = (
            f"/api/files/{row.id}.input.png" if row.input_image_path else None
        )
        if value["prompt"].startswith("Revise this SVG according to the request."):
            value["prompt"] = value["prompt"].rsplit("\nRequest: ", 1)[-1]
        value["png_url"] = f"/api/files/{row.id}.png" if row.png_path and row.valid else None
        value["svg_url"] = f"/api/files/{row.id}.svg" if row.svg_path else None
        return value

    @app.get("/api/status")
    async def status():
        backends = load_backends()
        backend = backends["open_base"]
        missing = [
            name
            for name in ["agents", "cwsandbox", "weave", "wandb"]
            if importlib.util.find_spec(name) is None
        ]
        models = []
        for item in backends.values():
            reason = (
                "Run uv sync, then restart the studio."
                if missing
                else backend_unavailable_reason(item)
            )
            try:
                policy = resolve_inference_policy(item, studio_mode=True).to_dict()
            except ValueError as exc:
                policy = None
                reason = f"Invalid inference configuration: {exc}"
            models.append(
                {
                    "id": item.name,
                    "label": item.display_name,
                    **item.presentation,
                    "tuning_method": item.tuning_method,
                    "model": item.model,
                    "runtime_policy": policy,
                    "available": reason is None,
                    "unavailable_reason": reason,
                }
            )
        return {
            "model": backend.model,
            "default_backend": "open_base",
            "models": models,
            "project": project_path(),
            "credentials_configured": bool(os.getenv("WANDB_API_KEY")),
            "missing_packages": missing,
            "busy": bool(active_groups),
            "agents_url": f"https://wandb.ai/{project_path()}/weave/agents",
        }

    @app.get("/api/history")
    async def history():
        return [serialize(row) for row in manifest.read()[-100:]]

    @app.get("/api/generations/{request_id}")
    async def generation_status(request_id: str):
        job = jobs.get(request_id)
        if job:
            if job.get("result"):
                return {"status": "complete", "row": serialize(job["result"])}
            if job.get("error"):
                return {"status": "failed", "error": job["error"]}
            partial = job.get("preview")
            if partial and partial.valid and partial.png_path:
                value = serialize(partial)
                value["png_url"] = f"/api/generations/{request_id}/preview.png"
                value["svg_url"] = None
                return {"status": "rendered", "row": value}
            return {"status": "running"}
        # Recover completed requests after a connection loss or server restart.
        row = next(
            (
                r
                for r in reversed(manifest.read())
                if r.execution_metadata.get("studio_request_id") == request_id
            ),
            None,
        )
        if row:
            return {"status": "complete", "row": serialize(row)}
        raise HTTPException(404, "Generation not found.")

    @app.get("/api/generations/{request_id}/preview.png")
    async def generation_preview(request_id: str):
        partial = jobs.get(request_id, {}).get("preview")
        if not partial or not partial.valid or not partial.png_path:
            raise HTTPException(404, "Preview not ready.")
        return FileResponse(
            partial.png_path, media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/generate")
    async def generate_bufo(request: GenerationRequest):
        request_id = request.request_id or uuid4().hex
        if request_id in jobs or any(
            r.execution_metadata.get("studio_request_id") == request_id for r in manifest.read()
        ):
            raise HTTPException(409, "This request already exists. Check its generation status.")
        async with generation_slot(request):
            prompt, group_id, history = request.prompt, None, []
            if request.previous_id:
                saved = manifest.read()
                previous = next((r for r in saved if r.id == request.previous_id), None)
                if previous is None:
                    raise HTTPException(404, "Previous bufo was not found. Start a new drawing.")
                group_id = previous.group_id
                # Follow parent links for new threads. Older manifests without
                # parent links retain their chronological conversation history.
                chain, cursor = [], previous
                by_id = {r.id: r for r in saved}
                while cursor is not None:
                    chain.append(cursor)
                    cursor = by_id.get(cursor.parent_id) if cursor.parent_id else None
                if len(chain) == 1 and previous.response_text == "":
                    chain = list(
                        reversed(
                            [
                                r
                                for r in saved[: saved.index(previous) + 1]
                                if r.group_id == group_id
                            ]
                        )
                    )
                for r in reversed(chain):
                    value = r.to_dict()
                    value["prompt"] = serialize(r)["prompt"]
                    if r.input_image_path:
                        value["image"] = image_data_url(r.input_image_path)
                    history.append(value)
            backend_name = request.backend or (history[-1]["backend"] if history else "open_base")
            backend = load_backends().get(backend_name)
            if backend is None:
                raise HTTPException(422, "Unknown model. Choose a model from the studio selector.")
            reason = backend_unavailable_reason(backend)
            if reason:
                raise HTTPException(409, reason)
            job = {}
            jobs[request_id] = job
            # Retain active jobs, with only the latest 100 completed jobs in memory.
            finished = [
                key for key, value in jobs.items() if value.get("result") or value.get("error")
            ]
            for key in finished[:-100]:
                del jobs[key]
            try:
                agent = (
                    getattr(generate, "__self__", None)
                    if generate
                    else BufoAgent(backend, renderer, output_dir, studio_mode=True, max_turns=100)
                )
                if agent is not None:
                    agent.on_render = lambda partial: job.update(preview=partial)
                generate_fn = generate or agent.generate
                async with asyncio.timeout(1800):
                    options = {"group_id": group_id, "history": history}
                    if request.image:
                        options["image"] = request.image
                    row = await generate_fn(prompt, **options)
                row.parent_id = request.previous_id
                row.execution_metadata["studio_request_id"] = request_id
                manifest.append(row)
                job["result"] = row
                return serialize(row)
            except TimeoutError:
                job["error"] = "Generation timed out after thirty minutes. Try a simpler theme."
                raise HTTPException(
                    504, "Generation timed out after thirty minutes. Try a simpler theme."
                ) from None
            except Exception as exc:
                job["error"] = public_error(exc)
                raise HTTPException(502, public_error(exc)) from None

    @app.get("/api/files/{filename}")
    async def file(filename: str):
        match = re.fullmatch(r"([a-f0-9]{32})\.(png|svg|input\.png)", filename)
        if not match:
            raise HTTPException(404, "File not found.")
        row = next((r for r in manifest.read() if r.id == match[1]), None)
        path = output_dir / filename
        if (
            row is None
            or not path.is_file()
            or (match[2] == "png" and not row.valid)
            or (match[2] == "input.png" and not row.input_image_path)
        ):
            raise HTTPException(404, "File not found.")
        return FileResponse(
            path,
            media_type="image/svg+xml" if match[2] == "svg" else "image/png",
            filename=filename,
            content_disposition_type="attachment" if match[2] == "svg" else "inline",
        )

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "static/index.html")

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Local baseline-agent studio")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reload", action="store_true", help="Restart automatically on Python edits"
    )
    args = parser.parse_args()
    uvicorn.run(
        "bufogen.app.ui:create_app",
        factory=True,
        host="127.0.0.1",
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).parent.parent)] if args.reload else None,
    )


if __name__ == "__main__":
    main()
