"""Publish every Stage B derived record that exists to Weave + W&B (idempotent; see stage_records)."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bufogen.settings import ROOT
from bufogen.workflows import stage_records

SB = "runs/distill/stage-b"
PAIRED_DIRS = ("anchoring", "sweep", "rl", "rl-eval", "confirmation")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root: Path) -> list[tuple[str, Path, dict]]:
    sb = root / SB
    records: list[tuple[str, Path, dict]] = []

    def add(kind, rel_or_path, metadata=None):
        path = rel_or_path if isinstance(rel_or_path, Path) else root / rel_or_path
        if path.is_file():
            meta = {"source_path": str(path.relative_to(root)), **(metadata or {})}
            records.append((kind, path, meta))
        else:
            records.append((kind, path, None))

    add("funnel", f"{SB}/export/funnel.json")
    for path in sorted((sb / "decisions").glob("attempt-*.jsonl")):
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        counts = {d: sum(1 for r in rows if r.get("decision") == d) for d in ("accept", "reject")}
        add(f"decisions-{path.stem}", path, {**counts, "rows": len(rows)})
    add("astra-calibration", f"{SB}/rl/astra-calibration.json")
    add("anchoring-probe", f"{SB}/anchoring/report.json")
    add("student-probe", f"{SB}/student-probe.json")
    add("rl-tasks", f"{SB}/rl/tasks-train.metadata.json")
    add("checkpoints", f"{SB}/checkpoints.json")
    lowconf = sb / "lowconf"
    for name in ("summary.json", "NONE.txt"):  # never the key directory or packet contents
        if (lowconf / name).is_file():
            add("lowconf", lowconf / name)
            break
    else:
        raise FileNotFoundError(f"lowconf: neither summary.json nor NONE.txt in {lowconf}")
    add("sweep-selection", f"{SB}/sweep/selection.json")
    add("sweep-distinctness", f"{SB}/sweep/distinctness.json")
    # Scoring summaries only; never confirmation/packet or confirmation/key (blinding).
    add("confirmation-summary", f"{SB}/confirmation/astra-v5-k3/summary.json")
    add("heldout-sol-summary", f"{SB}/heldout-sol-arm/astra-v5-k3/summary.json")
    for sub in PAIRED_DIRS:
        for path in sorted((sb / sub).glob("**/paired-*.json")):
            if "key" in path.relative_to(sb).parts:
                continue
            add(f"{sub}-{path.stem}", path)
    prompts = root / "config/distill_prompts.yaml"
    aug = root / "docs/distill/augmented-prompts.md"
    meta = {"prompt_sha256": _sha(prompts)} if prompts.is_file() else {}
    add("prompt-set", prompts, meta)
    add("prompt-set-doc", aug, meta)
    add("progress-doc", "docs/distill/stage-b-progress.md")
    add("stage-log", f"{SB}/stage-b-log.md")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    ledger = ROOT / SB / "publications.json"
    published = 0
    for kind, path, metadata in collect(ROOT):
        rel = path.relative_to(ROOT)
        if metadata is None:
            print(f"SKIP missing {kind} {rel}")
        elif args.dry_run:
            print(f"WOULD PUBLISH {kind} {rel}")
        else:
            entry = stage_records.publish_record(kind, path, metadata=metadata, ledger=ledger)
            published += entry["published"]
            state = "published" if entry["published"] else "unchanged"
            print(f"{state} {kind} {entry['weave_ref']} {entry['wandb_artifact']}")
    print(f"records newly published: {published}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
