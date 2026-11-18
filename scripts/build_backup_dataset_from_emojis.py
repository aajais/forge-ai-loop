import asyncio

from bufogen.curation.backup import build_backup_dataset
from bufogen.settings import ROOT

if __name__ == "__main__":
    print(asyncio.run(build_backup_dataset(ROOT / "runs/backup")))
