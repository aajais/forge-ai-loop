import base64
import io

import pytest
from PIL import Image

from bufogen.app.ui import GenerationRequest
from bufogen.rendering.image_input import normalize_image


@pytest.mark.parametrize("format,mime", [("PNG", "png"), ("JPEG", "jpeg"), ("WEBP", "webp")])
def test_screenshot_is_decoded_resized_and_normalized(format, mime):
    output = io.BytesIO()
    Image.new("RGB", (2000, 1000), "blue").save(output, format=format)
    value = f"data:image/{mime};base64," + base64.b64encode(output.getvalue()).decode()
    normalized = normalize_image(value)
    with Image.open(io.BytesIO(base64.b64decode(normalized.split(",", 1)[1]))) as image:
        assert image.format == "PNG" and image.size == (1600, 800)
    request = GenerationRequest(image=value)
    assert request.prompt == "Create a Bufo scene inspired by this image."


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/image.png",
        "data:image/svg+xml;base64,PHN2Zy8+",
        "data:image/png;base64,AAAA",
        "data:image/png;base64,invalid!",
    ],
)
def test_invalid_screenshot_is_rejected(value):
    with pytest.raises(ValueError):
        GenerationRequest(prompt="Draw", image=value)


def test_screenshot_size_limit(monkeypatch):
    from bufogen.rendering import image_input

    monkeypatch.setattr(image_input, "MAX_IMAGE_BYTES", 2)
    with pytest.raises(ValueError, match="5 MB"):
        normalize_image("data:image/png;base64,AAAA")
