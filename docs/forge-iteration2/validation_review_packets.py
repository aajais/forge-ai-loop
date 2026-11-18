"""Prepare a fixed, model-blinded visual diagnostic; never write review decisions.

Uses the '-base' development prompt from each of the four validation families,
across all five arms. This sample is defined before candidate evaluation.
"""
import hashlib
import json
from pathlib import Path
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import validated_evidence

ROOT = Path(__file__).resolve().parents[2]


def main():
    root = ROOT / "runs/forge-iteration2"
    _, generations, _, _ = validated_evidence(root, "validation")
    selected = [r.to_dict() for r in generations if r.prompt_id.endswith("-base")]
    assert len(selected) == 20
    assert len({r["benchmark_metadata"]["family"] for r in selected}) == 4
    selected.sort(key=lambda row: hashlib.sha256(("blind-v1:" + row["id"]).encode()).hexdigest())
    out = root / "validation-blind-review"
    out.mkdir(parents=True, exist_ok=True)
    inputs, key = [], []
    for i, row in enumerate(selected):
        path = Path(row["png_path"]) if row.get("png_path") else None
        inputs.append({"index": i, "label": f"Image {i+1:02d}", "prompt": row["prompt"],
            "requirements": row["benchmark_metadata"]["requirements"],
            "png_path": str(path) if path else None,
            "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None})
        key.append({"index": i, "generation_id": row["id"], "prompt_id": row["prompt_id"],
                    "backend": row["backend"], "model": row["model"]})
    for name, value in (("inputs.json", inputs), ("unblinding-key.json", key)):
        data = json.dumps(value, indent=2) + "\n"
        target = out / name
        if target.exists() and target.read_text() != data:
            raise ValueError("Refusing to replace a fixed blind-review sample")
        target.write_text(data)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    for start in range(0, len(inputs), 4):
        canvas = Image.new("RGB", (1080, 1480), "#eef0ed")
        draw = ImageDraw.Draw(canvas)
        for n, row in enumerate(inputs[start:start+4]):
            x, y = n % 2 * 540, n // 2 * 740
            draw.text((x+10, y+5), row["label"], font=font, fill="black")
            if row["png_path"]:
                im = Image.open(row["png_path"]).convert("RGBA")
                im.thumbnail((512, 512))
                canvas.paste(im, (x+14, y+32), im)
            else:
                draw.text((x+20, y+220), "NO IMAGE", font=font, fill="red")
            for line, content in enumerate(textwrap.wrap(" | ".join(row["requirements"]), 60)):
                draw.text((x+10, y+550+line*19), content, font=font, fill="black")
        canvas.save(out / f"sheet-{start//4:02d}.png")
    print(json.dumps({"images": len(inputs), "directory": str(out)}))


if __name__ == "__main__":
    main()
