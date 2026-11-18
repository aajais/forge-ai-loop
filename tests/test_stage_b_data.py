import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


funnel = _load("data_funnel")
lowconf = _load("lowconf_queue")


def scored(status="ok", confidence=0.9):
    return {"scores": {"astra_judge": {"status": status, "judge_confidence": confidence}}}


def lowconf_rows():
    return [
        SimpleNamespace(id="a", **scored("ok", 0.59)),
        SimpleNamespace(id="b", **scored("ok", 0.6)),
        SimpleNamespace(id="c", **scored("error", 0.1)),
        SimpleNamespace(id="d", **scored("ok", None)),
        SimpleNamespace(id="e", scores={}),
    ]


def test_lowconf_selection_requires_ok_and_below_floor():
    assert [r.id for r in lowconf.select_lowconf(lowconf_rows())] == ["a"]


def test_both_consumers_share_the_predicate_on_the_same_rows():
    rows = lowconf_rows()
    dicts = [
        scored(r.scores["astra_judge"]["status"], r.scores["astra_judge"]["judge_confidence"])
        if r.scores
        else {"scores": {}}
        for r in rows
    ]
    assert [funnel.is_low_confidence(d) for d in dicts] == [
        r in lowconf.select_lowconf(rows) for r in rows
    ]


def test_funnel_low_confidence_matches_selection():
    assert funnel.is_low_confidence(scored("ok", 0.3))
    assert not funnel.is_low_confidence(scored("ok", 0.6))
    assert not funnel.is_low_confidence(scored("error", 0.3))


def test_concat_unique_rejects_duplicates():
    a = [{"source_generation_id": "1"}, {"source_generation_id": "2"}]
    assert len(funnel.concat_unique([a, [{"source_generation_id": "3"}]])) == 3
    with pytest.raises(ValueError, match="duplicate"):
        funnel.concat_unique([a, [{"source_generation_id": "2"}]])


def test_attempt_and_total_funnel():
    manifest = [
        {"benchmark_metadata": {"status": "completed"}},
        {"benchmark_metadata": {"status": "completed"}},
        {"benchmark_metadata": {"status": "error"}},
    ]
    rows = [scored("ok", 0.9), scored("ok", 0.4), scored("judge_error", None)]
    decisions = [
        {"decision": "accept", "prompt_family": "x"},
        {"decision": "reject", "prompt_family": "x"},
        {"decision": "reject", "prompt_family": "y"},
    ]
    stats = funnel.attempt_funnel(manifest, rows, decisions, [{"a": 1}], families=["x", "y"])
    assert stats == {
        "prompts": 3,
        "completed_generations": 2,
        "judged_ok": 2,
        "judge_errors": 1,
        "accepted": 1,
        "rejected": 2,
        "low_confidence": 1,
        "accepted_per_family": {"x": 1, "y": 0},
        "exported_examples": 1,
    }
    total = funnel.total_funnel({"attempt-1": stats, "attempt-2": stats})
    assert total["accepted"] == 2 and total["accepted_per_family"] == {"x": 2, "y": 0}


def test_family_table_has_all_sixteen_rows_with_zero_family():
    families = [f"f{i:02d}" for i in range(16)]
    decisions = [{"decision": "accept", "prompt_family": "f00"}]
    stats = funnel.attempt_funnel([], [], decisions, [], families=families)
    assert list(stats["accepted_per_family"]) == families
    assert stats["accepted_per_family"]["f00"] == 1 and stats["accepted_per_family"]["f15"] == 0
    total = funnel.total_funnel({"a": stats, "b": stats})
    assert len(total["accepted_per_family"]) == 16 and total["accepted_per_family"]["f00"] == 2


def test_train_families_from_config_are_sixteen():
    assert (
        len(funnel.train_families(Path(__file__).parents[1] / "config/distill_prompts.yaml")) == 16
    )


def _stage(tmp_path, n_attempts=1):
    for i in range(1, n_attempts + 1):
        (tmp_path / "sol-data" / f"attempt-{i}" / "astra-v5-k3").mkdir(parents=True)
        (tmp_path / "sol-data" / f"attempt-{i}" / "astra-v5-k3" / "scored.jsonl").write_text("")
    (tmp_path / "sol-data" / "attempt-9").mkdir()  # no scored rows: not counted as scanned
    return tmp_path


def test_main_none_branch_counts_scanned_and_skips_queue(tmp_path, monkeypatch):
    root = _stage(tmp_path)
    monkeypatch.setattr(lowconf, "STAGE_B", root)
    monkeypatch.setattr(
        lowconf, "load_scored_rows", lambda d, s: [SimpleNamespace(**scored("ok", 0.9), id="x")]
    )
    monkeypatch.setattr(lowconf, "create_packet", lambda *a, **k: pytest.fail("no packet"))
    lowconf.main()
    text = (root / "lowconf" / "NONE.txt").read_text()
    assert "attempts scanned: 1" in text and "low-confidence rows: 0" in text
    assert json.loads((root / "lowconf" / "summary.json").read_text()) == {
        "count": 0,
        "attempts_scanned": 1,
    }


def test_main_queue_branch_removes_stale_none(tmp_path, monkeypatch):
    root = _stage(tmp_path)
    (root / "lowconf").mkdir()
    (root / "lowconf" / "NONE.txt").write_text("stale")
    monkeypatch.setattr(lowconf, "STAGE_B", root)
    monkeypatch.setattr(
        lowconf, "load_scored_rows", lambda d, s: [SimpleNamespace(**scored("ok", 0.2), id="x")]
    )
    calls = {}

    def fake_packet(rows, out, **k):
        calls.update(rows=rows, key=k["key_dir"])
        out.mkdir(parents=True)
        (out / "queue.jsonl").write_text("neutral")

    monkeypatch.setattr(lowconf, "create_packet", fake_packet)
    import bufogen.curation.review_queue as rq

    monkeypatch.setattr(rq, "create_queue", lambda p, k, name: {"queue_id": "q1", "url": "u"})
    lowconf.main()
    assert not (root / "lowconf" / "NONE.txt").exists()
    assert len(calls["rows"]) == 1 and calls["key"] == root / "lowconf" / "key"
    summary = json.loads((root / "lowconf" / "summary.json").read_text())
    assert set(summary) == {"count", "attempts_scanned", "queue_id", "queue_url", "packet_sha256"}
    assert summary["count"] == 1 and summary["attempts_scanned"] == 1
    assert summary["queue_id"] == "q1" and summary["queue_url"] == "u"
    assert len(summary["packet_sha256"]) == 64


@pytest.mark.parametrize("family", [None, "nope"])
def test_unknown_or_missing_family_raises(family):
    with pytest.raises(ValueError, match="prompt_family"):
        funnel.attempt_funnel(
            [], [], [{"decision": "accept", "prompt_family": family}], [], families=["x"]
        )
