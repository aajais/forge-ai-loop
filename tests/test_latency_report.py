"""Offline tests for scripts/latency_report.py on synthetic benchmark rows."""

import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/latency_report.py"
SPEC = importlib.util.spec_from_file_location("latency_report_tests", PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def call(finish="stop", out=100, latency=1.0, reasoning_tokens=None, reasoning_chars=0):
    return {
        "index": 0,
        "finish_reason": finish,
        "input_tokens": 10,
        "output_tokens": out,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_chars": reasoning_chars,
        "latency_s": latency,
        "served_model": "m",
    }


METRICS = dict.fromkeys(
    ("similarity", "anatomy", "composition", "readability", "constraint_coverage"), 0.5
)


def row(prompt_id, latency, calls, *, status="completed", valid=True, quality_failure=False):
    return {
        "id": f"id-{prompt_id}",
        "prompt_id": prompt_id,
        "prompt": "p",
        "backend": "open_base",
        "model": "m",
        "group_id": "g",
        "svg": "",
        "valid": valid,
        "latency_s": latency,
        "model_calls": calls,
        "truncated_calls": sum(c["finish_reason"] == "length" for c in calls),
        "benchmark_metadata": {"status": status},
        "scores": {"scene_quality": {"status": "ok", "quality_failure": quality_failure} | METRICS},
    }


def settings(n):
    return {
        "run_id": "r",
        "configuration_hash": "h",
        "comparison_fingerprint": "f",
        "configuration": {
            "arms": [{"name": "open_base"}],
            "prompts": [{"id": f"p{i}", "split": "train"} for i in range(n)],
        },
    }


def test_median_p90_and_length_counts():
    rows = [
        row(f"p{i}", float(i), [call(finish="length" if i < 3 else "stop", out=10 * i), call()])
        for i in range(1, 11)
    ]
    stats = report.summarize(rows, rows, settings(10))
    assert stats["n"] == 10
    assert stats["median_latency_s"] == 5.5
    assert stats["p90_latency_s"] == 9.1
    assert stats["length_calls"] == 2
    assert stats["total_calls"] == 20
    assert stats["rows_truncated"] == 2


def test_reasoning_tokens_not_reported_when_always_none():
    rows = [row("p0", 1.0, [call(reasoning_chars=40), call(reasoning_chars=60)])]
    stats = report.summarize(rows, rows, settings(1))
    assert stats["median_reasoning_tokens"] == "not reported"
    assert stats["median_reasoning_chars"] == 50
    reported = [row("p0", 1.0, [call(reasoning_tokens=7)])]
    assert report.summarize(reported, reported, settings(1))["median_reasoning_tokens"] == 7


def test_strict_pass_uses_benchmark_definition():
    rows = [
        row("p0", 1.0, [call()]),
        row("p1", 1.0, [call()], quality_failure=True),
        row("p2", 1.0, [call()], valid=False),
        row("p3", 0, [], status="generation_error"),
    ]
    rows[3]["scores"]["scene_quality"] = {"status": "missing_image"}
    stats = report.summarize(rows, rows, settings(4))
    assert stats["strict_passes"] == 1
    assert stats["generation_failures"] == 1
    # Failed rows with no recorded latency are excluded from latency statistics.
    assert stats["median_latency_s"] == 1.0


def test_unscored_run_and_table(tmp_path):
    rows = [row("p0", 2.0, [call()])]
    stats = report.summarize(rows, None, settings(1))
    assert stats["strict_passes"] == "not scored"
    run = tmp_path / "latency-x"
    run.mkdir()
    (run / "benchmark.json").write_text(json.dumps(settings(1)))
    (run / "manifest.jsonl").write_text(json.dumps(rows[0]) + "\n")
    table = report.table([report.load(run)])
    assert "latency-x" in table and "not scored" in table
