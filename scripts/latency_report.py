"""Per-arm latency/truncation table for benchmark run directories.

Usage: uv run python scripts/latency_report.py RUN_DIR...

Each RUN_DIR holds benchmark.json and manifest.jsonl (and scored.jsonl once
`bufogen --stage benchmark-score` has run). Row latency statistics use rows with a
recorded latency (latency_s > 0); failed rows without a partial generation record 0.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from bufogen.evaluation.benchmark import _summary
from bufogen.generation import BufoGeneration

NOT_REPORTED = "not reported"
NOT_SCORED = "not scored"


def _median(values):
    return statistics.median(values) if values else None


def _p90(values):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=10, method="inclusive")[8]


def _rows(dicts):
    return [BufoGeneration(**d) for d in dicts]


def strict_passes(scored, settings):
    """Strict passes per the benchmark module's own `_summary` ("provisional_passes")."""
    if scored is None:
        return NOT_SCORED
    arms = _summary(_rows(scored), settings)["arms"]
    return sum(arm["provisional_passes"] for arm in arms.values())


def summarize(manifest, scored, settings) -> dict:
    calls = [c for r in manifest for c in r.get("model_calls") or []]
    latencies = [r["latency_s"] for r in manifest if r["latency_s"] > 0]
    reasoning_tokens = [c["reasoning_tokens"] for c in calls if c["reasoning_tokens"] is not None]
    median_reasoning = _median(reasoning_tokens)
    return {
        "n": len(manifest),
        "generation_failures": sum(
            (r.get("benchmark_metadata") or {}).get("status") != "completed" for r in manifest
        ),
        "rows_truncated": sum((r.get("truncated_calls") or 0) > 0 for r in manifest),
        "length_calls": sum(c["finish_reason"] == "length" for c in calls),
        "total_calls": len(calls),
        "median_latency_s": _median(latencies),
        "p90_latency_s": _p90(latencies),
        "median_output_tokens_per_call": _median([c["output_tokens"] for c in calls]),
        "median_reasoning_tokens": NOT_REPORTED if median_reasoning is None else median_reasoning,
        "median_reasoning_chars": _median([c["reasoning_chars"] for c in calls]),
        "median_call_latency_s": _median([c["latency_s"] for c in calls]),
        "strict_passes": strict_passes(scored, settings),
    }


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / "benchmark.json").read_text())
    scored_path = run_dir / "scored.jsonl"
    scored = _read_jsonl(scored_path) if scored_path.exists() else None
    return {"arm": run_dir.name} | summarize(
        _read_jsonl(run_dir / "manifest.jsonl"), scored, settings
    )


COLUMNS = (
    ("arm", "arm"),
    ("n", "n"),
    ("gen failures", "generation_failures"),
    ("rows truncated", "rows_truncated"),
    ("length / calls", None),
    ("median latency s", "median_latency_s"),
    ("p90 latency s", "p90_latency_s"),
    ("median out tok/call", "median_output_tokens_per_call"),
    ("median reasoning tok", "median_reasoning_tokens"),
    ("median reasoning chars", "median_reasoning_chars"),
    ("median call latency s", "median_call_latency_s"),
    ("strict passes", "strict_passes"),
)


def _cell(stats, key):
    if key is None:
        return f"{stats['length_calls']}/{stats['total_calls']}"
    value = stats[key]
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    if key == "strict_passes" and isinstance(value, int):
        return f"{value}/{stats['n']}"
    return str(value)


def table(arms) -> str:
    lines = [
        "| " + " | ".join(name for name, _ in COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for stats in arms:
        lines.append("| " + " | ".join(_cell(stats, key) for _, key in COLUMNS) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    paths = (argv if argv is not None else sys.argv[1:]) or []
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2
    print(table([load(Path(p)) for p in paths]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
