"""Prepare actual image sheets and provenance; writes no review decisions."""

import hashlib
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT / "runs/forge-iteration2"
    rows = [json.loads(line) for line in (directory / "teacher/manifest.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["prompt_id"] for r in rows}) == 104
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    for number in range(3):
        selected = sorted(rows, key=lambda r: r["prompt_id"])[number::3]
        out = directory / "reviews" / f"reviewer-{number}"
        out.mkdir(parents=True, exist_ok=True)
        inputs = []
        for i, row in enumerate(selected):
            path = Path(row["png_path"]) if row.get("png_path") else None
            inputs.append({
                "index": i, "source_generation_id": row["id"], "prompt_id": row["prompt_id"],
                "prompt": row["prompt"], "family": row["benchmark_metadata"]["family"],
                "partition": row["benchmark_metadata"]["split"],
                "requirements": row["benchmark_metadata"]["requirements"],
                "png_path": str(path) if path else None,
                "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
                "valid": row["valid"], "conversation_url": row.get("conversation_url"),
            })
        data = json.dumps(inputs, indent=2) + "\n"
        target = out / "inputs.json"
        if target.exists() and target.read_text() != data:
            raise ValueError("Cannot change a prepared image review packet")
        target.write_text(data)
        for start in range(0, len(inputs), 4):
            batch = inputs[start:start+4]
            canvas = Image.new("RGB", (1080, 1480), "#eef0ed")
            draw = ImageDraw.Draw(canvas)
            for n, item in enumerate(batch):
                x, y = (n % 2) * 540, (n // 2) * 740
                label = f"{item['index']} {item['prompt_id']}"
                draw.text((x+10, y+5), label[-62:], font=font, fill="black")
                if item["png_path"]:
                    im = Image.open(item["png_path"]).convert("RGBA")
                    im.thumbnail((512, 512))
                    canvas.paste(im, (x+14, y+32), im)
                else:
                    draw.text((x+20, y+220), "NO IMAGE — reject", font=font, fill="red")
                for line, content in enumerate(textwrap.wrap(" | ".join(item["requirements"]), 60)):
                    draw.text((x+10, y+550+line*19), content, font=font, fill="black")
            canvas.save(out / f"sheet-{start//4:02d}.png")
        print(json.dumps({"directory": str(out), "images": len(inputs)}))


if __name__ == "__main__":
    main()
