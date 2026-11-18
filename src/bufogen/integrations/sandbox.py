"""The only module allowed to talk to cwsandbox."""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path

import httpx

from bufogen.settings import ROOT, required

log = logging.getLogger(__name__)


def _sandbox_auth():
    """Select native per-instance auth; let the SDK resolve credentials."""
    from cwsandbox import AuthStrategy

    if os.getenv("CWSANDBOX_API_KEY", "").strip():
        return AuthStrategy.COREWEAVE_API_KEY
    return AuthStrategy.WANDB


async def sandbox_status(sandbox_id: str) -> dict:
    """Read startup details through the SDK's public status API."""
    from uuid import UUID

    from cwsandbox import Sandbox

    sandbox_id = str(UUID(sandbox_id))
    result = await Sandbox.from_id(sandbox_id, auth=_sandbox_auth(), timeout_seconds=10)
    # Select only diagnostic fields; the full spec can include credentials.
    return {
        "sandbox_id": result.sandbox_id,
        "state": f"STATE_{result.status.value.upper()}" if result.status else None,
        "state_reason": result.status_reason,
        "runner_id": result.runner_id,
        "exit_code": result.returncode,
        "images": [container.image for container in result.containers],
    }


def _retryable_create_error(exc: Exception) -> bool:
    from cwsandbox.exceptions import (
        SandboxError,
        SandboxRequestTimeoutError,
        SandboxUnavailableError,
    )

    if isinstance(exc, (SandboxUnavailableError, SandboxRequestTimeoutError)):
        return True
    # These provisioning failures currently arrive as generic SDK SandboxErrors.
    # Do not retry auth, validation, quota, or arbitrary application errors.
    return type(exc) is SandboxError and any(
        message in str(exc)
        for message in (
            "owning gateway could not complete the CreateSandbox",
            "selected runner failed to place the sandbox",
        )
    )


async def renderer_wheel() -> Path:
    """Fetch a pinned Linux wheel on the host; render sandboxes never need egress."""
    folder = ROOT / ".cache/render"
    folder.mkdir(parents=True, exist_ok=True)
    for cached in sorted(folder.glob("*.whl")):
        sidecar = cached.with_name(cached.name + ".sha256")
        try:
            if sidecar.read_text().strip() == hashlib.sha256(cached.read_bytes()).hexdigest():
                return cached
        except OSError:
            continue
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get("https://pypi.org/pypi/resvg-py/0.5.0/json")
        response.raise_for_status()
        candidates = [
            u
            for u in response.json()["urls"]
            if "manylinux" in u["filename"]
            and "x86_64" in u["filename"]
            and "abi3" in u["filename"]
        ]
        if not candidates:
            raise RuntimeError(
                "No Linux x86_64 resvg wheel available; set BUFO_RENDER_IMAGE to a prebuilt image."
            )
        item = candidates[0]
        target = folder / item["filename"]
        if (
            not target.exists()
            or hashlib.sha256(target.read_bytes()).hexdigest() != item["digests"]["sha256"]
        ):
            response = await client.get(item["url"])
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != item["digests"]["sha256"]:
                raise RuntimeError("Renderer wheel checksum mismatch.")
            target.write_bytes(response.content)
        target.with_name(target.name + ".sha256").write_text(item["digests"]["sha256"])
        return target


class SandboxJob:
    def __init__(self, *, gpu: bool = False, image: str | None = None, lifetime: int = 1800):
        self.gpu = gpu
        self.image = image or os.getenv("BUFO_SFT_IMAGE" if gpu else "BUFO_RENDER_IMAGE")
        self.lifetime = lifetime
        self._sandbox = None
        self._lock = asyncio.Lock()

    async def _start(self):
        from cwsandbox import Sandbox

        if self.gpu:
            required("CWSANDBOX_API_KEY")
        files = []
        if not self.gpu:
            files = [
                {
                    "mount_path": "/app/render_worker.py",
                    "file_content": (ROOT / "scripts/render_worker.py").read_bytes(),
                },
                {
                    "mount_path": "/app/DejaVuSans.ttf",
                    "file_content": (ROOT / "assets/fonts/DejaVuSans.ttf").read_bytes(),
                },
                {
                    "mount_path": "/app/svg.py",
                    "file_content": (ROOT / "src/bufogen/rendering/svg.py").read_bytes(),
                },
            ]
            if not self.image:
                wheel = await renderer_wheel()
                files.append({"mount_path": "/app/resvg.whl", "file_content": wheel.read_bytes()})
        if self.gpu and not self.image:
            raise ValueError(
                "Set BUFO_SFT_IMAGE to the image built from scripts/train_entrypoints/Dockerfile."
            )
        options = dict(
            auth=_sandbox_auth(),
            container_image=self.image or "python:3.12-slim",
            request_timeout_seconds=30,
            max_lifetime_seconds=self.lifetime,
            placement_mode=None if self.gpu else "serverless",
            tags=["bufo-gen", "sft" if self.gpu else "render"],
            network={"deny_egress": not self.gpu, "deny_ingress": True},
            resources={"cpu": "8", "memory": "64Gi", "gpu": {"count": 1, "memory_gb": 80}}
            if self.gpu
            else {"cpu": "1", "memory": "1Gi"},
        )
        # Only this adapter selects the sandbox credential provider.
        # Keep one SDK object: start() reuses its idempotency key after ambiguous
        # gateway failures. Calling Sandbox.run() again would create a new key.
        self._sandbox = Sandbox(**options)
        for attempt in range(1, 4):
            try:
                await self._sandbox.start()
                break
            except Exception as exc:
                if not _retryable_create_error(exc):
                    raise
                if attempt == 3:
                    raise RuntimeError(
                        "The remote sandbox gateway could not create the renderer or training "
                        "sandbox after 3 attempts. Try again shortly; if it persists, share "
                        "this error with your sandbox service administrator. "
                        f"Service error: {exc}"
                    ) from exc
                hint = getattr(exc, "retry_delay", None)
                delay = min(10.0, max(2**attempt, hint.total_seconds() if hint else 0))
                log.warning(
                    "Sandbox creation attempt %d/3 failed (%s); retrying in %.1fs",
                    attempt,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
        if not self.gpu:
            # Keep binary assets out of create-time Kubernetes resources. exec()
            # waits for RUNNING; only then transfer assets through the file API.
            result = await self._sandbox.exec(["mkdir", "-p", "/app"], timeout_seconds=30)
            if result.returncode:
                raise RuntimeError(f"Renderer directory setup failed: {result.stderr}")
            for file in files:
                await self._sandbox.write_file(
                    file["mount_path"], file["file_content"], timeout_seconds=30
                )
        if not self.gpu and not self.image:
            result = await self._sandbox.exec(
                [
                    "python",
                    "-c",
                    'import zipfile; zipfile.ZipFile("/app/resvg.whl").extractall("/app")',
                ],
                timeout_seconds=30,
            )
            if result.returncode:
                raise RuntimeError(f"Renderer setup failed: {result.stderr}")

    async def execute(self, entrypoint: Path, payload: dict, *, stream_logs: bool = False) -> dict:
        from cwsandbox.exceptions import SandboxFailedError

        async with self._lock:
            try:
                if self._sandbox is None:
                    await self._start()
                if self.gpu:
                    await self._sandbox.write_file("/tmp/job.py", Path(entrypoint).read_bytes())
                    command = ["python", "/tmp/job.py"]
                else:
                    command = ["python", "/app/render_worker.py", "/tmp/render-result.json"]
                process = self._sandbox.exec(
                    command, stdin=True, timeout_seconds=self.lifetime - 30 if self.gpu else 20
                )
                await process.stdin.write(json.dumps(payload).encode())
                await process.stdin.close()
                if stream_logs:
                    async for line in process.stdout:
                        log.info("%s", line.rstrip())
                result = await process
                if result.returncode:
                    raise RuntimeError(
                        f"Sandbox job exited {result.returncode}: {result.stderr[-2000:]}"
                    )
                if not self.gpu:
                    # PNG/base64 responses can exceed reliable stdout stream sizes.
                    data = await self._sandbox.read_file(
                        "/tmp/render-result.json", timeout_seconds=30
                    )
                    return json.loads(data)
                lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
                return json.loads(lines[-1])
            except BaseException as exc:
                startup_error = None
                if isinstance(exc, SandboxFailedError) and self._sandbox is not None:
                    sandbox_id = self._sandbox.sandbox_id
                    try:
                        details = await sandbox_status(sandbox_id)
                        reason = details["state_reason"] or "Server supplied no startup reason."
                        startup_error = RuntimeError(
                            f"Sandbox {sandbox_id} failed to start: {reason} "
                            f"(runner: {details['runner_id'] or 'unknown'})."
                        )
                    except Exception as diagnostic_error:
                        log.warning(
                            "Could not read sandbox startup details (%s)",
                            type(diagnostic_error).__name__,
                        )
                try:
                    await self.close()
                except Exception as cleanup_error:
                    # Keep the original create/execute failure visible to the user.
                    log.warning("Sandbox cleanup also failed (%s)", type(cleanup_error).__name__)
                if startup_error is not None:
                    raise startup_error from exc
                raise

    async def close(self):
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            await sandbox.stop()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()
