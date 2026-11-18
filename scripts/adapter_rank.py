"""Read-only: adapter rank `r` from a W&B LoRA artifact's adapter_config.json (small JSON only)."""

import json
import sys
import tempfile
from pathlib import Path

import wandb
from dotenv import load_dotenv

from bufogen.settings import ROOT


def adapter_rank(reference: str) -> int:
    load_dotenv(ROOT / ".env")
    artifact = wandb.Api().artifact(reference.removeprefix("wandb-artifact:///"), type="lora")
    with tempfile.TemporaryDirectory() as tmp:
        path = artifact.get_entry("adapter_config.json").download(root=tmp)
        return int(json.loads(Path(path).read_text())["r"])


if __name__ == "__main__":
    print(adapter_rank(sys.argv[1]))
