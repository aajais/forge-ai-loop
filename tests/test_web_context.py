import json
from types import SimpleNamespace

import pytest

from bufogen.agents.web_context import WebContext
from bufogen.rendering.emoji import emoji_shortcode


@pytest.mark.parametrize(
    "proposed,fallback,expected",
    [
        (":Bufo-Pitbull-Vibes:", "", ":bufo-pitbull-vibes:"),
        ("GPU Gardener!", "", ":bufo-gpu-gardener:"),
        ("", "Snow day", ":bufo-snow-day:"),
        ("🐸", "🐸", ":bufo-drawing:"),
        ("bufo_café", "", ":bufo_cafe:"),
    ],
)
def test_shortcodes(proposed, fallback, expected):
    assert emoji_shortcode(proposed, fallback) == expected
    assert len(emoji_shortcode("x" * 100)) <= 66


def mock_service(monkeypatch, *, error=None):
    requests = []

    class Client:
        def __init__(self, **kwargs):
            self.responses = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def create(self, **kwargs):
            requests.append(kwargs)
            if error:
                raise error
            source = {"type": "url_citation", "url": "https://example.com/event", "title": "Event"}
            return SimpleNamespace(
                id="response-id",
                output_text="Blue stage. \ue200cite\ue202opaque\ue201",
                model_dump=lambda: {
                    "usage": {"input_tokens": 12},
                    "output": [
                        {"type": "web_search_call", "status": "completed"},
                        {
                            "type": "message",
                            "content": [
                                {
                                    "annotations": [
                                        source,
                                        source,
                                        {"type": "url_citation", "url": "javascript:alert(1)"},
                                    ]
                                }
                            ],
                        },
                    ],
                },
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("BUFO_WEB_SEARCH", "true")
    monkeypatch.setattr("bufogen.agents.web_context.AsyncOpenAI", Client)
    return requests


async def test_search_returns_real_citations_and_bounds_external_calls(monkeypatch):
    requests = mock_service(monkeypatch)
    context = WebContext()
    result = json.loads(await context.search_web("Example event"))
    assert result["status"] == "ok"
    assert result["summary"] == "Blue stage."
    assert result["sources"] == [{"url": "https://example.com/event", "title": "Event"}]
    assert requests[0]["tool_choice"] == "required"
    assert requests[0]["tools"][0]["external_web_access"] is True
    await context.search_web("Other event")
    assert json.loads(await context.search_web("Again"))["status"] == "unavailable"
    assert len(requests) == 2
    assert len(context.records) == 2
    assert WebContext().records == []


async def test_search_failure_redacts_credentials_and_does_not_raise(monkeypatch):
    mock_service(monkeypatch, error=RuntimeError("Bearer private-api-key"))
    result = await WebContext().search_web("event")
    assert json.loads(result)["status"] == "unavailable"
    assert "private-api-key" not in result


@pytest.mark.parametrize("setting", ["disabled", "missing-key", "bad-query"])
async def test_search_unavailable_without_a_network_request(monkeypatch, setting):
    requests = mock_service(monkeypatch)
    if setting == "disabled":
        monkeypatch.setenv("BUFO_WEB_SEARCH", "false")
    if setting == "missing-key":
        monkeypatch.delenv("OPENAI_API_KEY")
    result = await WebContext().search_web("" if setting == "bad-query" else "event")
    assert json.loads(result)["status"] == "unavailable"
    assert requests == []
