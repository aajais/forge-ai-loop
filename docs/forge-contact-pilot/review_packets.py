"""Freeze every pilot image into three identity-blinded assistant review packets."""
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont
from bufogen.generation import BufoGeneration
from bufogen.evaluation import benchmark
from bufogen.settings import ROOT


def main():
    root = ROOT / "runs/forge-contact-pilot/comparison"
    protocol = json.loads((root / "pilot-protocol.json").read_text())
    rows = []
    for interface in ("svg", "contact"):
        directory = root / interface
        settings = json.loads((directory / "benchmark.json").read_text())
        if settings["configuration"] != protocol["configurations"][interface]:
            raise ValueError("Pilot configuration changed")
        values = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
        expected = {
            (p["id"], a["name"])
            for p in settings["configuration"]["prompts"]
            for a in settings["configuration"]["arms"]
        }
        if len(values) != 32 or {(r["prompt_id"], r["backend"]) for r in values} != expected:
            raise ValueError("Every attempted generation, including errors, must be present")
        benchmark._validate_rows([BufoGeneration(**r) for r in values], settings)
        rows.extend(dict(row, interface=interface) for row in values)
    rows.sort(key=lambda r: hashlib.sha256(("contact-pilot-review-v1:" + r["id"]).encode()).hexdigest())
    out = root / "blind-review"
    out.mkdir(exist_ok=True)
    inputs, key = [], []
    for i, row in enumerate(rows):
        path = Path(row["png_path"]) if row.get("png_path") else None
        inputs.append({
            "index": i, "label": f"Image {i+1:02d}", "prompt": row["prompt"],
            "requirements": row["benchmark_metadata"]["requirements"],
            "png_path": str(path) if path else None,
            "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
        })
        key.append({k: row[k] for k in ("id", "prompt_id", "interface", "backend", "model")} | {"index": i})
    def save(path, value):
        data = json.dumps(value, indent=2) + "\n"
        if path.exists() and path.read_text() != data:
            raise ValueError("Existing review sample changed")
        path.write_text(data)
    save(out / "unblinding-key.json", key)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    for reviewer in range(3):
        folder = out / f"reviewer-{reviewer}"
        folder.mkdir(exist_ok=True)
        batch = inputs[reviewer::3]
        save(folder / "inputs.json", batch)
        for start in range(0, len(batch), 4):
            sheet = Image.new("RGB", (1080, 1480), "#eef0ed")
            draw = ImageDraw.Draw(sheet)
            for n, row in enumerate(batch[start:start+4]):
                x, y = n % 2 * 540, n // 2 * 740
                draw.text((x+10, y+5), row["label"], font=font, fill="black")
                if row["png_path"]:
                    with Image.open(row["png_path"]) as source:
                        image = source.convert("RGBA")
                    image.thumbnail((512, 512))
                    sheet.paste(image, (x+14, y+32), image)
                else:
                    draw.text((x+20, y+220), "NO IMAGE", font=font, fill="red")
                for line, content in enumerate(textwrap.wrap(" | ".join(row["requirements"]), 60)):
                    draw.text((x+10, y+550+line*19), content, font=font, fill="black")
            sheet.save(folder / f"sheet-{start//4:02d}.png")
    print(json.dumps({"images": len(rows), "directory": str(out), "batches": [22, 21, 21]}))


if __name__ == "__main__":
    main()
