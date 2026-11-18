"""Decode screenshot attachments into bounded, metadata-free PNG images."""

import base64
import binascii
import io
import re
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DATA_URL = 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 100


def normalize_image(value: str) -> str:
    match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=\s]+)", value)
    if not match:
        raise ValueError("Attach a PNG, JPEG, or WebP image.")
    try:
        data = base64.b64decode(re.sub(r"\s", "", match[2]), validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Images must be 5 MB or smaller.")
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Attach a PNG, JPEG, or WebP image.")
            if source.width * source.height > 20_000_000:
                raise ValueError("Images must contain no more than 20 million pixels.")
            picture = ImageOps.exif_transpose(source).convert("RGBA")
            picture.thumbnail((1600, 1600))
            picture.info.clear()
            output = io.BytesIO()
            picture.save(output, format="PNG")
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not read this image. Try a PNG, JPEG, or WebP screenshot.") from exc
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def image_data_url(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
