"""Freeze all 32 diagnostic images for intervention-blinded assistant review."""
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def main():
    root = ROOT / "runs/forge-iteration2/plan-continuation"
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    protocol = json.loads((root / "diagnostic-protocol.json").read_text())
    expected = {
        (prompt["id"], arm)
        for prompt in protocol["prompts"]
        for arm in ("ordinary_plan", "teacher_plan_continuation")
    }
    if len(rows) != 32 or {(r["prompt_id"], r["backend"]) for r in rows} != expected:
        raise ValueError("Require every requested prompt/arm, including failed generations")
    rows.sort(key=lambda r: hashlib.sha256(("plan-blind-v1:" + r["id"]).encode()).hexdigest())
    out = root / "blind-review"
    out.mkdir(exist_ok=True)
    inputs, key = [], []
    for i, row in enumerate(rows):
        png = Path(row["png_path"]) if row.get("png_path") else None
        inputs.append({
            "index": i, "label": f"Image {i+1:02d}", "prompt": row["prompt"],
            "requirements": row["benchmark_metadata"]["requirements"],
            "png_path": str(png) if png else None,
            "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest() if png else None,
        })
        key.append({k: row[k] for k in ("id", "prompt_id", "backend", "model")} | {"index": i})
    for filename, value in (("inputs.json", inputs), ("unblinding-key.json", key)):
        path, data = out / filename, json.dumps(value, indent=2) + "\n"
        if path.exists() and path.read_text() != data:
            raise ValueError("Existing blinded packet changed")
        path.write_text(data)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    for start in range(0, len(inputs), 4):
        sheet = Image.new("RGB", (1080, 1480), "#eef0ed")
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(inputs[start:start + 4]):
            x, y = n % 2 * 540, n // 2 * 740
            draw.text((x + 10, y + 5), row["label"], font=font, fill="black")
            if row["png_path"]:
                with Image.open(row["png_path"]) as source:
                    image = source.convert("RGBA")
                image.thumbnail((512, 512))
                sheet.paste(image, (x + 14, y + 32), image)
            else:
                draw.text((x + 20, y + 220), "NO IMAGE", font=font, fill="red")
            for line, content in enumerate(textwrap.wrap(" | ".join(row["requirements"]), 60)):
                draw.text((x + 10, y + 550 + line * 19), content, font=font, fill="black")
        sheet.save(out / f"sheet-{start // 4:02d}.png")
    print(json.dumps({"images": len(inputs), "directory": str(out)}))


if __name__ == "__main__":
    main()
