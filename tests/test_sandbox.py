import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from bufogen.integrations.sandbox import SandboxJob


class Writer:
    def __init__(self):
        self.data = b""
        self.closed = False

    async def write(self, data):
        self.data += data

    async def close(self):
        self.closed = True


class Process:
    def __init__(self, returncode=0):
        self.stdin = Writer()
        self.returncode = returncode

    def __await__(self):
        async def done():
            return SimpleNamespace(
                returncode=self.returncode, stdout='{"valid":true}\n', stderr="worker failed"
            )

        return done().__await__()


class Remote:
    def __init__(self, returncode=0):
        self.calls = []
        self.stopped = False
        self.returncode = returncode
        self.files = {}
        self.events = []

    async def start(self):
        self.events.append("start")

    async def write_file(self, path, contents, **kwargs):
        self.events.append(("write", path))
        self.files[path] = contents

    def exec(self, command, **kwargs):
        self.events.append(("exec", command))
        process = Process(self.returncode)
        self.calls.append((command, kwargs, process))
        return process

    async def stop(self):
        self.stopped = True

    async def read_file(self, path, **kwargs):
        return self.files.get(path, b'{"valid":true}')


async def test_payload_is_stdin_data_and_warm_job_is_reused():
    remote = Remote()
    job = SandboxJob()
    job._sandbox = remote
    payload = {"svg": "<svg><text>$(touch /tmp/unsafe)</text></svg>"}
    for _ in range(2):
        assert await job.execute(Path("worker.py"), payload) == {"valid": True}
    assert len(remote.calls) == 2
    command, kwargs, process = remote.calls[0]
    assert command == ["python", "/app/render_worker.py", "/tmp/render-result.json"]
    assert kwargs["stdin"] is True and process.stdin.closed
    assert json.loads(process.stdin.data) == payload
    await job.close()
    assert remote.stopped


async def test_large_render_response_is_read_from_file_instead_of_stdout():
    remote = Remote()
    expected = {"valid": True, "png_base64": "a" * 200_000}
    remote.files["/tmp/render-result.json"] = json.dumps(expected).encode()
    job = SandboxJob()
    job._sandbox = remote
    assert await job.execute(Path("worker.py"), {"svg": "<svg/>"}) == expected


async def test_remote_failure_stops_sandbox_and_preserves_error():
    remote = Remote(returncode=1)
    job = SandboxJob()
    job._sandbox = remote
    with pytest.raises(RuntimeError, match="worker failed"):
        await job.execute(Path("worker.py"), {})
    assert remote.stopped and job._sandbox is None


@pytest.mark.parametrize("file_output", [False, True])
def test_worker_renders_a_trusted_fixture_to_real_png(monkeypatch, capsys, tmp_path, file_output):
    # This fixed fixture exercises resvg; generated model content is never run locally.
    import base64
    import importlib.util
    import io
    import resource
    import sys

    from PIL import Image

    import bufogen.rendering.svg
    from bufogen.settings import ROOT

    pytest.importorskip("resvg_py")
    monkeypatch.setitem(sys.modules, "svg", bufogen.rendering.svg)
    spec = importlib.util.spec_from_file_location(
        "render_worker", ROOT / "scripts/render_worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker, "__file__", str(ROOT / "assets/fonts/render_worker.py"))
    monkeypatch.setattr(resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><circle cx="256" cy="256" r="160" fill="#76b85c"/></svg>'
                }
            )
        ),
    )
    output = tmp_path / "result.json" if file_output else None
    worker.main(output)
    stdout = capsys.readouterr().out
    if file_output:
        assert not stdout
    result = json.loads(output.read_text() if file_output else stdout)
    assert result["valid"], result["error"]
    with Image.open(io.BytesIO(base64.b64decode(result["png_base64"]))) as png:
        assert png.size == (512, 512)
        assert png.getpixel((256, 256))[:3] == (118, 184, 92)


@pytest.mark.parametrize("coreweave_key", [False, True])
async def test_render_sandbox_has_no_egress_or_credentials_and_uploads_a_font(
    monkeypatch, coreweave_key
):
    import sys

    import cwsandbox

    options = {}
    remote = Remote()

    def start(**kwargs):
        options.update(kwargs)
        return remote

    if coreweave_key:
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-coreweave-key")
    else:
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setitem(sys.modules, "wandb.sandbox", None)
    monkeypatch.setattr(cwsandbox, "Sandbox", start)
    monkeypatch.setattr(cwsandbox, "set_auth_mode", Mock(side_effect=AssertionError("global auth")))
    job = SandboxJob(image="python-with-resvg")
    await job._start()
    assert options["network"]["deny_egress"] is True
    assert options["placement_mode"] == "serverless"
    assert "environment_variables" not in options
    assert not options.get("mounted_files")
    assert "/app/DejaVuSans.ttf" in remote.files
    assert remote.events[0] == "start"
    assert remote.events[1] == ("exec", ["mkdir", "-p", "/app"])
    if coreweave_key:
        assert options["auth"] is cwsandbox.AuthStrategy.COREWEAVE_API_KEY
    else:
        assert options["auth"] is cwsandbox.AuthStrategy.WANDB
    await job.close()


def install_sandbox_factory(monkeypatch, failures):
    import sys

    import cwsandbox

    remote = Remote()
    remote.start = AsyncMock(side_effect=[*failures, None])
    created = []

    def create(**options):
        created.append(options)
        return remote

    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "wandb.sandbox", None)
    monkeypatch.setattr(cwsandbox, "Sandbox", create)
    monkeypatch.setattr(cwsandbox, "set_auth_mode", lambda *args: None)
    sleep = AsyncMock()
    monkeypatch.setattr("bufogen.integrations.sandbox.asyncio.sleep", sleep)
    return remote, created, sleep


@pytest.mark.parametrize(
    "message",
    [
        "the owning gateway could not complete the CreateSandbox",
        "the selected runner failed to place the sandbox",
    ],
)
async def test_gateway_create_failure_retries_same_sandbox_then_executes(monkeypatch, message):
    from cwsandbox.exceptions import SandboxError

    failure = SandboxError(f"Create sandbox failed: {message}")
    remote, created, sleep = install_sandbox_factory(monkeypatch, [failure, failure])
    job = SandboxJob(image="python-with-resvg")
    assert await job.execute(Path("worker.py"), {}) == {"valid": True}
    assert len(created) == 1  # A new SDK object would generate a different create request ID.
    assert created[0]["request_timeout_seconds"] == 30
    assert remote.start.await_count == 3 and sleep.await_count == 2
    assert remote.calls[-1][0] == ["python", "/app/render_worker.py", "/tmp/render-result.json"]
    await job.close()


async def test_large_renderer_files_are_uploaded_after_creation(monkeypatch, tmp_path):
    wheel = tmp_path / "renderer.whl"
    wheel.write_bytes(b"wheel" * 300_000)
    monkeypatch.delenv("BUFO_RENDER_IMAGE", raising=False)
    monkeypatch.setattr("bufogen.integrations.sandbox.renderer_wheel", AsyncMock(return_value=wheel))
    remote, created, _ = install_sandbox_factory(monkeypatch, [])
    job = SandboxJob()
    await job._start()
    assert len(created) == 1 and not created[0].get("mounted_files")
    assert remote.files["/app/resvg.whl"] == wheel.read_bytes()
    assert set(remote.files) == {
        "/app/resvg.whl",
        "/app/DejaVuSans.ttf",
        "/app/render_worker.py",
        "/app/svg.py",
    }
    assert remote.events[0] == ("exec", ["mkdir", "-p", "/app"])
    assert remote.events[-1][0] == "exec"
    assert "extractall" in remote.events[-1][1][-1]
    await job.close()


async def test_upload_failure_stops_sandbox_before_rendering(monkeypatch):
    remote, _, _ = install_sandbox_factory(monkeypatch, [])
    remote.write_file = AsyncMock(side_effect=RuntimeError("upload failed"))
    job = SandboxJob(image="python-with-resvg")
    with pytest.raises(RuntimeError, match="upload failed"):
        await job.execute(Path("worker.py"), {})
    assert remote.stopped and job._sandbox is None
    assert len(remote.calls) == 1 and remote.calls[0][0] == ["mkdir", "-p", "/app"]


async def test_gateway_retry_exhaustion_explains_failure_and_cleans_up(monkeypatch):
    from cwsandbox.exceptions import SandboxUnavailableError

    failure = SandboxUnavailableError("gateway unavailable")
    remote, created, sleep = install_sandbox_factory(monkeypatch, [failure] * 3)
    job = SandboxJob(image="python-with-resvg")
    with pytest.raises(RuntimeError, match="after 3 attempts") as error:
        await job.execute(Path("worker.py"), {})
    assert error.value.__cause__ is failure
    assert len(created) == 1 and remote.start.await_count == 3
    assert sleep.await_count == 2 and not remote.calls
    assert remote.stopped and job._sandbox is None


@pytest.mark.parametrize("kind", ["authentication", "validation", "quota"])
async def test_nontransient_creation_errors_are_not_retried(monkeypatch, kind):
    from cwsandbox.exceptions import (
        CWSandboxAuthenticationError,
        SandboxResourceExhaustedError,
        SandboxValidationError,
    )

    cls = {
        "authentication": CWSandboxAuthenticationError,
        "validation": SandboxValidationError,
        "quota": SandboxResourceExhaustedError,
    }[kind]
    failure = cls("cannot create")
    remote, created, sleep = install_sandbox_factory(monkeypatch, [failure])
    job = SandboxJob(image="python-with-resvg")
    with pytest.raises(cls) as error:
        await job.execute(Path("worker.py"), {})
    assert error.value is failure
    assert len(created) == 1 and remote.start.await_count == 1
    sleep.assert_not_awaited()
    assert remote.stopped and not remote.calls


async def test_cleanup_failure_does_not_hide_creation_error(monkeypatch):
    from cwsandbox.exceptions import CWSandboxAuthenticationError

    failure = CWSandboxAuthenticationError("invalid credentials")
    remote, _, _ = install_sandbox_factory(monkeypatch, [failure])
    remote.stop = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    job = SandboxJob(image="python-with-resvg")
    with pytest.raises(CWSandboxAuthenticationError) as error:
        await job.execute(Path("worker.py"), {})
    assert error.value is failure
    assert job._sandbox is None


async def test_gpu_still_requires_coreweave_credentials(monkeypatch):
    remote, created, _ = install_sandbox_factory(monkeypatch, [])
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    with pytest.raises(ValueError, match="CWSANDBOX_API_KEY"):
        await SandboxJob(gpu=True, image="training-image")._start()
    assert not created
    remote.start.assert_not_awaited()


def test_wandb_auth_can_resolve_login_credentials_without_env_key(monkeypatch):
    from cwsandbox import AuthStrategy

    from bufogen.integrations.sandbox import _sandbox_auth

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    assert _sandbox_auth() is AuthStrategy.WANDB


async def test_startup_failure_reports_server_reason_before_cleanup(monkeypatch):
    from cwsandbox.exceptions import SandboxFailedError

    remote, _, _ = install_sandbox_factory(monkeypatch, [])
    remote.sandbox_id = "2b14917b-4b94-4931-99f1-b3864023add1"
    remote.exec = Mock(side_effect=SandboxFailedError("Sandbox failed to start"))

    async def status(sandbox_id):
        assert sandbox_id == remote.sandbox_id and not remote.stopped
        return {"state_reason": "ImagePullBackOff: registry unavailable", "runner_id": "runner-1"}

    monkeypatch.setattr("bufogen.integrations.sandbox.sandbox_status", status, raising=False)
    job = SandboxJob(image="python-with-resvg")
    with pytest.raises(RuntimeError, match="ImagePullBackOff") as error:
        await job.execute(Path("worker.py"), {})
    assert remote.sandbox_id in str(error.value)
    assert isinstance(error.value.__cause__, SandboxFailedError)
    assert remote.stopped


async def test_status_diagnostic_failure_keeps_original_startup_error(monkeypatch):
    from cwsandbox.exceptions import SandboxFailedError

    remote, _, _ = install_sandbox_factory(monkeypatch, [])
    remote.sandbox_id = "2b14917b-4b94-4931-99f1-b3864023add1"
    failure = SandboxFailedError("Sandbox failed to start")
    remote.exec = Mock(side_effect=failure)
    monkeypatch.setattr(
        "bufogen.integrations.sandbox.sandbox_status", AsyncMock(side_effect=TimeoutError()), raising=False
    )
    job = SandboxJob(image="python-with-resvg")
    with pytest.raises(SandboxFailedError) as error:
        await job.execute(Path("worker.py"), {})
    assert error.value is failure and remote.stopped


async def test_status_reads_server_reason_without_exposing_secrets(monkeypatch):
    from cwsandbox import AuthStrategy, Sandbox, SandboxStatus

    from bufogen.integrations.sandbox import sandbox_status

    response = SimpleNamespace(
        sandbox_id="2b14917b-4b94-4931-99f1-b3864023add1",
        status=SandboxStatus.FAILED,
        status_reason="Mount setup failed",
        runner_id="runner-1",
        returncode=None,
        containers=[
            SimpleNamespace(
                image="python:3.12-slim", environment_variables={"SECRET": "must-not-appear"}
            )
        ],
    )
    rpc = AsyncMock(return_value=response)
    monkeypatch.setattr(Sandbox, "from_id", rpc)
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("CWSANDBOX_BASE_URL", raising=False)
    result = await sandbox_status(response.sandbox_id)
    assert result["state_reason"] == "Mount setup failed"
    assert result["state"] == "STATE_FAILED" and result["runner_id"] == "runner-1"
    assert result["exit_code"] is None
    assert "must-not-appear" not in json.dumps(result)
    assert "test-wandb-key" not in json.dumps(result)
    rpc.assert_awaited_once_with(response.sandbox_id, auth=AuthStrategy.WANDB, timeout_seconds=10)


async def test_status_command_returns_diagnostics_without_running_agent(monkeypatch, capsys):
    from bufogen.cli import dispatch, parser

    details = {"state": "STATE_FAILED", "state_reason": "Image pull failed"}
    status = AsyncMock(return_value=details)
    monkeypatch.setattr("bufogen.integrations.sandbox.sandbox_status", status)
    args = parser().parse_args(
        ["--stage", "sandbox-status", "--sandbox-id", "2b14917b-4b94-4931-99f1-b3864023add1"]
    )
    assert await dispatch(args) == 0
    assert json.loads(capsys.readouterr().out) == details
    status.assert_awaited_once_with(args.sandbox_id)


class ExplodingClient:
    def __init__(self, *a, **k):
        raise AssertionError("network used")


async def test_cached_verified_wheel_skips_pypi(tmp_path, monkeypatch):
    import hashlib

    from bufogen.integrations import sandbox

    monkeypatch.setattr(sandbox, "ROOT", tmp_path)
    wheel = tmp_path / ".cache/render/resvg_py-0.5.0-cp38-abi3-manylinux_x86_64.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"w")
    wheel.with_name(wheel.name + ".sha256").write_text(hashlib.sha256(b"w").hexdigest())
    monkeypatch.setattr(sandbox.httpx, "AsyncClient", ExplodingClient)
    assert await sandbox.renderer_wheel() == wheel


async def test_mismatched_sidecar_falls_back_to_network_and_writes_sidecar(tmp_path, monkeypatch):
    import hashlib

    from bufogen.integrations import sandbox

    monkeypatch.setattr(sandbox, "ROOT", tmp_path)
    name = "resvg_py-0.5.0-cp38-abi3-manylinux_x86_64.whl"
    wheel = tmp_path / ".cache/render" / name
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"stale")
    wheel.with_name(name + ".sha256").write_text("0" * 64)
    good = hashlib.sha256(b"fresh").hexdigest()

    def handler(request):
        if request.url.path.endswith("/json"):
            return httpx.Response(
                200,
                json={"urls": [{"filename": name, "url": "https://x/w", "digests": {"sha256": good}}]},
            )
        return httpx.Response(200, content=b"fresh")

    real = httpx.AsyncClient
    monkeypatch.setattr(
        sandbox.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    assert await sandbox.renderer_wheel() == wheel
    assert wheel.read_bytes() == b"fresh"
    assert wheel.with_name(name + ".sha256").read_text() == good
