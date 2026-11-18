import base64
import hashlib
from collections import OrderedDict
from dataclasses import dataclass

from bufogen.integrations.sandbox import SandboxJob
from bufogen.rendering.svg import validate_svg
from bufogen.settings import ROOT


@dataclass(frozen=True)
class RenderResult:
    png_bytes: bytes | None
    valid: bool
    error: str | None


class Renderer:
    def __init__(self, job: SandboxJob | None = None):
        self.job = job or SandboxJob()
        self._cache = OrderedDict()

    async def render(self, svg: str) -> RenderResult:
        valid, error = validate_svg(svg)
        if not valid:
            return RenderResult(None, False, error)
        key = hashlib.sha256(svg.encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = await self.job.execute(ROOT / "scripts/render_worker.py", {"svg": svg})
        png = (
            base64.b64decode(data["png_base64"], validate=True) if data.get("png_base64") else None
        )
        result = RenderResult(png, bool(data["valid"] and png), data.get("error"))
        self._cache[key] = result
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return result

    async def close(self):
        await self.job.close()
