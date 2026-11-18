"""Copy a complete benchmark into neutral, hash-bound visual review packets."""

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont
from bufogen.evaluation import benchmark
from bufogen.generation import BufoGeneration


def freeze(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen packet differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def save(path, value):
    freeze(path, (json.dumps(value, indent=2) + "\n").encode())


def create(directory, *, reviewers=3):
    directory = Path(directory).resolve()
    settings = json.loads((directory / "benchmark.json").read_text())
    values = [
        json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()
    ]
    expected = {
        (p["id"], a["name"]) for p in settings["configuration"]["prompts"]
        for a in settings["configuration"]["arms"]
    }
    if (not 1 <= reviewers <= 3 or len(values) != len(expected)
            or {(r["prompt_id"], r["backend"]) for r in values} != expected):
        raise ValueError("Require every planned attempt before creating review packets")
    benchmark._validate_rows([BufoGeneration(**row) for row in values], settings)
    values.sort(key=lambda row: hashlib.sha256(("neutral-review-v1:" + row["id"]).encode()).hexdigest())
    root = directory / "visual-review"
    key, inputs = [], []
    for index, row in enumerate(values):
        image_path, image_hash = None, None
        if row.get("png_path"):
            data = Path(row["png_path"]).read_bytes()
            image_hash = hashlib.sha256(data).hexdigest()
            if image_hash != row["benchmark_metadata"].get("png_sha256"):
                raise ValueError("Image differs from the original generation hash")
            image_path = root / "images" / f"image-{index + 1:02d}.png"
            freeze(image_path, data)
        inputs.append({
            "index": index, "label": f"Image {index + 1:02d}",
            "prompt": row["prompt"], "requirements": row["benchmark_metadata"]["requirements"],
            "png_path": str(image_path) if image_path else None, "png_sha256": image_hash,
        })
        key.append({
            "index": index, "id": row["id"], "prompt_id": row["prompt_id"],
            "backend": row["backend"], "model": row["model"],
        })
    save(root / "unblinding-key.json", key)
    save(root / "protocol.json", {
        "version": "neutral-visual-review-v1", "attempts": len(values), "reviewers": reviewers,
        "benchmark_sha256": hashlib.sha256((directory / "benchmark.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((directory / "manifest.jsonl").read_bytes()).hexdigest(),
        "packet_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "blinding": "Model identities and original image paths omitted from reviewer inputs.",
    })
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    for reviewer in range(reviewers):
        folder = root / f"reviewer-{reviewer}"
        batch = inputs[reviewer::reviewers]
        save(folder / "inputs.json", batch)
        for offset in range(0, len(batch), 4):
            sheet = Image.new("RGB", (1080, 1800), "#eef0ed")
            draw = ImageDraw.Draw(sheet)
            for n, item in enumerate(batch[offset:offset + 4]):
                x, y = n % 2 * 540, n // 2 * 900
                draw.text((x + 10, y + 5), item["label"], font=font, fill="black")
                if item["png_path"]:
                    with Image.open(item["png_path"]) as im:
                        rgba = im.convert("RGBA")
                        background = Image.new("RGBA", rgba.size, "white")
                        background.alpha_composite(rgba)
                        sheet.paste(background.convert("RGB"), (x + 10, y + 30))
                else:
                    draw.text((x + 10, y + 180), "No image generated", font=font, fill="black")
                lines = textwrap.wrap(item["prompt"], width=62)
                for requirement in item["requirements"]:
                    lines.extend(textwrap.wrap("- " + requirement, width=62))
                if len(lines) > 20:
                    raise ValueError("Review text exceeds sheet space; inspect packet layout")
                draw.multiline_text((x + 10, y + 550), "\n".join(lines), font=font,
                                    fill="black", spacing=2)
            path = folder / f"sheet-{offset // 4 + 1:02d}.png"
            import io
            data = io.BytesIO()
            sheet.save(data, format="PNG")
            freeze(path, data.getvalue())
    return {"attempts": len(values), "root": str(root)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reviewers", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(create(args.directory, reviewers=args.reviewers)))
