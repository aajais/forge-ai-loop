import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.getenv("BUFO_ROOT", Path(__file__).resolve().parents[2]))
load_dotenv(ROOT / ".env")


def project_path() -> str:
    return f"{os.getenv('WANDB_ENTITY', 'wg-fc-infra-workshops')}/{os.getenv('WANDB_PROJECT', 'bufo-gen')}"


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} in {ROOT / '.env'} before running this stage.")
    return value
