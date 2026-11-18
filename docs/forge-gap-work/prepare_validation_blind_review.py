"""Prepare a fixed, score-independent 16-image diagnostic review after generation.

Only manifest provenance and frozen prompts are read. No score/selection files are
consulted. Reviewer material contains opaque IDs, images, prompts and requirements;
the separate private mapping retains arm identities and original error details.
"""

import argparse
import hashlib
import json
import random
import secrets
import shutil
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/forge-gap/validation-candidates"
DESTINATION = ROOT / "runs/forge-gap/validation-blind-review"
FAMILIES = ("airport_marshaller", "florist_arranging", "ice_cream_cart", "locksmith")
ARMS = {"open_base", "closed_quality_bar", "open_tuned_epoch1", "open_tuned_epoch3"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fixed_sample():
    benchmark_data = (SOURCE / "benchmark.json").read_bytes()
    config = json.loads(benchmark_data)["configuration"]
    if {arm["name"] for arm in config["arms"]} != ARMS:
        raise ValueError("Expected the four fixed validation arms")
    prompts = {
        p["id"]: p
        for p in config["prompts"]
        if p["id"] in {f"gap-validation-{family}-base" for family in FAMILIES}
    }
    if len(prompts) != 4:
        raise ValueError("The four predefined base prompts must exist")
    expected = {(prompt_id, arm) for prompt_id in prompts for arm in ARMS}
    selected = {}
    for line in (SOURCE / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["prompt_id"], row["backend"])
        if key not in expected:
            continue
        if key in selected:
            raise ValueError("Duplicate predefined scenario attempt; do not select a substitute")
        # Deliberately project away scores, generation metadata, and judge outputs.
        selected[key] = {
            field: row.get(field)
            for field in ("id", "prompt_id", "backend", "model", "png_path", "error")
        }
    return prompts, selected, expected - set(selected), digest(benchmark_data)


def prepare(check=False):
    prompts, selected, missing, benchmark_sha = fixed_sample()
    if missing:
        print(
            json.dumps({"ready": False, "completed_fixed_attempts": len(selected), "required": 16})
        )
        return False
    if check:
        print(json.dumps({"ready": True, "completed_fixed_attempts": 16, "required": 16}))
        return True
    if DESTINATION.exists():
        raise FileExistsError("Refusing to overwrite a blinded assignment or review")
    temporary = Path(tempfile.mkdtemp(prefix="validation-blind-review-", dir=DESTINATION.parent))
    try:
        public = temporary / "reviewer"
        (public / "images").mkdir(parents=True)
        (public / "sheets").mkdir()
        seed = secrets.randbits(128)
        ordered = list(sorted(selected))
        random.Random(seed).shuffle(ordered)
        inputs, mapping = [], []
        font = ImageFont.truetype(str(ROOT / "assets/fonts/DejaVuSans.ttf"), 18)
        for key in ordered:
            source = selected[key]
            prompt = prompts[key[0]]
            opaque_id = secrets.token_hex(6)
            source_path = Path(source["png_path"]) if source.get("png_path") else None
            image_exists = (
                not source.get("error") and source_path is not None and source_path.is_file()
            )
            image_path = public / "images" / f"{opaque_id}.png"
            png_sha = None
            if image_exists:
                original = source_path.read_bytes()
                png_sha = digest(original)
                # Re-encode pixels to avoid exposing embedded model/provenance metadata.
                try:
                    with Image.open(source_path) as image:
                        image.convert("RGBA").save(image_path)
                except (OSError, ValueError):
                    image_exists = False
            if not image_exists:
                image = Image.new("RGB", (768, 768), "#eef0ed")
                draw = ImageDraw.Draw(image)
                draw.text(
                    (70, 350), "NO GENERATED IMAGE — REVIEW AS FAILURE", font=font, fill="black"
                )
                image.save(image_path)
            item = {
                "review_id": opaque_id,
                "prompt": prompt["prompt"],
                "requirements": prompt["requirements"],
                "image_path": f"images/{opaque_id}.png",
                "image_available": image_exists,
            }
            inputs.append(item)
            mapping.append({**source, "review_id": opaque_id, "original_png_sha256": png_sha})
        dump(public / "inputs.json", inputs)
        (public / "instructions.md").write_text(
            "# Blinded diagnostic image review\n\n"
            "Review every opaque ID independently against its prompt and requirements. "
            "Inspect the actual image, enlarging contacts when needed. Require a recognizable, "
            "connected Bufo plus every visible requirement for a pass. Missing images fail. "
            "Record requirement-level booleans, character/connectivity judgments, overall pass, "
            "and concrete visual evidence. Do not infer model identity or consult scores. "
            "Use only this reviewer directory; keep decisions keyed by review_id.\n\n"
            "This fixed diagnostic sample does not replace the preregistered selection rule.\n"
        )
        for start in range(0, 16, 4):
            canvas = Image.new("RGB", (1200, 1580), "#eef0ed")
            draw = ImageDraw.Draw(canvas)
            for index, item in enumerate(inputs[start : start + 4]):
                x, y = (index % 2) * 600, (index // 2) * 790
                draw.text((x + 20, y + 10), item["review_id"], font=font, fill="black")
                with Image.open(public / item["image_path"]) as original:
                    image = original.convert("RGBA")
                    image.thumbnail((560, 560))
                    canvas.paste(image, (x + 20, y + 40), image)
                lines = textwrap.wrap(" | ".join(item["requirements"]), width=55)
                for number, line in enumerate(lines):
                    draw.text((x + 20, y + 615 + number * 23), line, font=font, fill="black")
            canvas.save(public / "sheets" / f"sheet-{start // 4 + 1:02d}.png")
        dump(temporary / "PRIVATE-mapping.json", {"shuffle_seed": seed, "mapping": mapping})
        dump(
            temporary / "receipt.json",
            {
                "schema_version": 1,
                "benchmark_sha256": benchmark_sha,
                "selection_rule": "All four arms for the base variant of each predefined validation family",
                "families": list(FAMILIES),
                "attempt_count": 16,
                "source": str(SOURCE),
                "reviewer_inputs_sha256": digest((public / "inputs.json").read_bytes()),
                "private_mapping_sha256": digest((temporary / "PRIVATE-mapping.json").read_bytes()),
            },
        )
        temporary.rename(DESTINATION)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(
        json.dumps(
            {
                "ready": True,
                "reviewer_directory": str(DESTINATION / "reviewer"),
                "attempts": 16,
                "sheets": 4,
            }
        )
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    prepare(parser.parse_args().check)
