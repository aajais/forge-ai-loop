"""Stage B funnel: concatenate per-attempt reviewed examples and count the pipeline stages.

Pure functions over row dicts (no network); ``main`` reads the attempt directories under
runs/distill/stage-b and writes export/accepted.jsonl and export/funnel.json.
"""

import json
import re
from pathlib import Path

from bufogen.curation.judge_reviews import is_low_confidence

CONFIG = Path("config/distill_prompts.yaml")
STAGE_B = Path("runs/distill/stage-b")
SCORED = "astra-v5-k3"


def _judge(row: dict) -> dict:
    return (row.get("scores") or {}).get("astra_judge") or {}


def train_families(config: Path) -> list[str]:
    """Sorted family names of the train split in the prompt config."""
    import yaml

    rows = yaml.safe_load(Path(config).read_text())
    return sorted({r["family"] for r in rows if r["split"] == "train"})


def concat_unique(example_lists: list[list[dict]]) -> list[dict]:
    """Concatenate reviewed examples; a repeated source_generation_id is an error."""
    out, seen = [], set()
    for examples in example_lists:
        for example in examples:
            gid = example["source_generation_id"]
            if gid in seen:
                raise ValueError(f"duplicate source_generation_id {gid}")
            seen.add(gid)
            out.append(example)
    return out


def attempt_funnel(
    manifest_rows: list[dict],
    scored_rows: list[dict],
    decisions: list[dict],
    exported: list[dict],
    families: list[str],
) -> dict:
    counts = dict.fromkeys(families, 0)
    for d in decisions:
        if d["decision"] == "accept":
            family = d.get("prompt_family")
            if family not in counts:
                raise ValueError(f"accepted decision has unknown prompt_family {family!r}")
            counts[family] += 1
    return {
        "prompts": len(manifest_rows),
        "completed_generations": sum(
            (r.get("benchmark_metadata") or {}).get("status") == "completed" for r in manifest_rows
        ),
        "judged_ok": sum(_judge(r).get("status") == "ok" for r in scored_rows),
        "judge_errors": sum(_judge(r).get("status") == "judge_error" for r in scored_rows),
        "accepted": sum(d["decision"] == "accept" for d in decisions),
        "rejected": sum(d["decision"] == "reject" for d in decisions),
        "low_confidence": sum(is_low_confidence(r) for r in scored_rows),
        "accepted_per_family": dict(sorted(counts.items())),
        "exported_examples": len(exported),
    }


def total_funnel(per_attempt: dict[str, dict]) -> dict:
    total = {k: 0 for k in next(iter(per_attempt.values())) if k != "accepted_per_family"}
    families: dict[str, int] = {}
    for stats in per_attempt.values():
        for key in total:
            total[key] += stats[key]
        for family, n in stats["accepted_per_family"].items():
            families[family] = families.get(family, 0) + n
    total["accepted_per_family"] = dict(sorted(families.items()))
    return total


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    root = STAGE_B
    families = train_families(CONFIG)
    per_attempt, example_lists = {}, []
    dirs = sorted(
        (p for p in (root / "sol-data").glob("attempt-*") if re.fullmatch(r"attempt-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[1]),
    )
    for d in dirs:
        decisions_path = root / "decisions" / f"{d.name}.jsonl"
        if not decisions_path.exists():
            continue
        exported = _jsonl(root / "export" / d.name / "reviewed-examples.jsonl")
        example_lists.append(exported)
        per_attempt[d.name] = attempt_funnel(
            _jsonl(d / "manifest.jsonl"),
            _jsonl(d / SCORED / "scored.jsonl"),
            _jsonl(decisions_path),
            exported,
            families,
        )
    if not per_attempt:
        raise SystemExit("no attempt with a decisions file")
    accepted = concat_unique(example_lists)
    out = root / "export"
    out.mkdir(parents=True, exist_ok=True)
    (out / "accepted.jsonl").write_text("".join(json.dumps(e) + "\n" for e in accepted))
    funnel = {"attempts": per_attempt, "total": total_funnel(per_attempt)}
    (out / "funnel.json").write_text(json.dumps(funnel, indent=2) + "\n")
    print(json.dumps(funnel["total"], indent=2))


if __name__ == "__main__":
    main()
