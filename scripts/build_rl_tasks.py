"""Build the Stage B RL environment task set (train split, each with its Sol reference).

Local files only; no paid calls.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml

from bufogen.evaluation.references import load_reference_manifest
from bufogen.settings import ROOT
from bufogen.training.rl_contract import write_environment_tasks

PROMPTS = ROOT / "config" / "distill_prompts.yaml"
REFERENCES = ROOT / "runs" / "references" / "distill-v2" / "references.json"
OUTPUT = ROOT / "runs" / "distill" / "stage-b" / "rl" / "tasks-train.jsonl"


def select_rows(prompts: list[dict], manifest: dict) -> list[dict]:
    """Train-split prompts with the reference attached as the Astra judge reads it."""
    rows = []
    for prompt in prompts:
        row = dict(prompt)
        if row.get("split") == "train":
            entry = manifest["rows"].get(row["id"])
            if not entry or entry["png_path"] is None:
                raise ValueError(f"No reference image for train prompt {row['id']}")
            row["reference"] = {
                "png_path": str((Path(manifest["base_dir"]) / entry["png_path"]).resolve()),
                "png_sha256": entry["png_sha256"],
            }
        rows.append(row)
    return rows


def build(prompts_path: Path, references_path: Path, output: Path) -> dict:
    prompts = yaml.safe_load(Path(prompts_path).read_text())
    rows = select_rows(prompts, load_reference_manifest(references_path))
    with tempfile.TemporaryDirectory() as scratch:
        metadata = write_environment_tasks(rows, Path(scratch))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(scratch) / "train.jsonl", output)
        sidecar = output.with_suffix(".metadata.json")
        shutil.copyfile(Path(scratch) / "metadata.json", sidecar)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    metadata = build(args.prompts, args.references, args.output)
    print(f"{metadata['count']} tasks -> {args.output} (sha256 {metadata['tasks_sha256'][:12]})")


if __name__ == "__main__":
    main()
