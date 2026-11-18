import io
import json

import httpx
from PIL import Image

from bufogen.curation import backup


async def test_reference_fetch_intersects_slack_names_and_retains_unmatched_names(
    tmp_path, monkeypatch
):
    image = io.BytesIO()
    Image.new("RGB", (16, 16), "green").save(image, format="PNG")

    def response(request):
        if "/git/trees/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "tree": [
                        {"type": "blob", "path": "bufos/bufo-garden.png"},
                        {"type": "blob", "path": "bufos/bufo-cloud.png"},
                    ]
                },
            )
        if "raw.githubusercontent" in request.url.host:
            return httpx.Response(200, content=image.getvalue())
        return httpx.Response(200, json={"default_branch": "main"})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        backup.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(response), **kw),
    )
    monkeypatch.setattr(backup, "ROOT", tmp_path)
    names = tmp_path / "names.txt"
    names.write_text("bufo-garden\namps_bufo\n")
    downloaded = await backup.fetch_reference_bufos(names_file=names)
    assert [p.name for p in downloaded] == ["bufo-garden.png"]
    assert json.loads((tmp_path / "assets/reference_bufos/prompt_only_names.json").read_text()) == [
        "amps_bufo"
    ]
