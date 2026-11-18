import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/paired_across_runs.py"
spec = importlib.util.spec_from_file_location("paired_across_runs", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

JUDGE = {"rubric_version": "bufo-astra-v1", "max_completion_tokens": 3200}
REFS = {"sha256": "abc"}


def row(prompt_id, backend, family, passed, index):
    return {
        "id": f"{backend}-{index}",
        "prompt_id": prompt_id,
        "prompt": "p",
        "backend": backend,
        "model": "m",
        "group_id": "g",
        "svg": "<svg/>",
        "valid": True,
        "latency_s": 1.0,
        "benchmark_metadata": {"family": family, "split": "validation", "requirements": ["a"]},
        "scores": {"astra_judge": {"strict_pass": passed}},
    }


def write_dir(tmp_path, name, rows, judge=JUDGE, refs=REFS, self_reference=False):
    d = tmp_path / name
    d.mkdir()
    (d / "scorer.json").write_text(
        json.dumps({"judge": judge, "references": refs, "self_reference": self_reference})
    )
    (d / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def grid(backend, passes):
    rows = []
    for index, (family, passed) in enumerate(passes):
        rows.append(row(f"p{index}", backend, family, passed, index))
    return rows


def test_pairs_rows_across_directories(tmp_path):
    fams = ["f0", "f0", "f1", "f1"]
    left = write_dir(tmp_path, "students", grid("student", zip(fams, [False, False, True, False])))
    right = write_dir(tmp_path, "sol", grid("teacher", zip(fams, [True, True, True, True])))
    report = mod.paired_across(left, "student", right, "teacher")
    assert report["left"]["strict_pass"] == 1 and report["right"]["strict_pass"] == 4
    assert report["paired"]["prompts"] == 4 and report["paired"]["families"] == 2
    assert report["paired"]["difference"] == pytest.approx(-0.75)
    assert report["judge"] == JUDGE and report["references"] == REFS


def test_refuses_mismatched_judge_or_references_or_self_reference(tmp_path):
    fams = ["f0", "f1"]
    left = write_dir(tmp_path, "a", grid("student", zip(fams, [False, True])))
    other_judge = write_dir(
        tmp_path, "b", grid("teacher", zip(fams, [True, True])), judge=JUDGE | {"threshold": 0.5}
    )
    other_refs = write_dir(
        tmp_path, "c", grid("teacher", zip(fams, [True, True])), refs={"sha256": "x"}
    )
    selfref = write_dir(
        tmp_path, "d", grid("teacher", zip(fams, [True, True])), self_reference=True
    )
    with pytest.raises(SystemExit, match="Judge configuration"):
        mod.paired_across(left, "student", other_judge, "teacher")
    with pytest.raises(SystemExit, match="Reference manifest"):
        mod.paired_across(left, "student", other_refs, "teacher")
    with pytest.raises(SystemExit, match="Self-reference"):
        mod.paired_across(left, "student", selfref, "teacher")
    with pytest.raises(SystemExit, match="No rows"):
        mod.paired_across(left, "missing", other_refs, "teacher")


@pytest.mark.parametrize("field", ["prompt", "requirements"])
def test_refuses_pairs_whose_prompt_or_requirements_differ(tmp_path, field):
    fams = ["f0", "f0", "f1", "f1"]
    left_rows = grid("student", zip(fams, [False, False, True, False]))
    right_rows = grid("teacher", zip(fams, [True, True, True, True]))
    if field == "prompt":
        right_rows[2]["prompt"] = "a different request"
    else:
        right_rows[2]["benchmark_metadata"]["requirements"] = ["a", "b"]
    left = write_dir(tmp_path, "students", left_rows)
    right = write_dir(tmp_path, "sol", right_rows)
    with pytest.raises(SystemExit, match="p2.*differs"):
        mod.paired_across(left, "student", right, "teacher")


def test_different_manifests_pair_only_when_paired_reference_drawings_are_identical(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    fams = ["f0", "f1"]

    def manifest(name, rows):
        data = json.dumps({"rows": rows}).encode()
        (tmp_path / f"{name}.json").write_bytes(data)
        return {"path": f"{name}.json", "sha256": __import__("hashlib").sha256(data).hexdigest()}

    narrow = manifest("narrow", {"p0": {"png_sha256": "a"}, "p1": {"png_sha256": "b"}})
    wide = manifest(
        "wide", {"p0": {"png_sha256": "a"}, "p1": {"png_sha256": "b"}, "p2": {"png_sha256": "c"}}
    )
    changed = manifest("changed", {"p0": {"png_sha256": "a"}, "p1": {"png_sha256": "X"}})
    left = write_dir(tmp_path, "l", grid("student", zip(fams, [False, True])), refs=narrow)
    superset = write_dir(tmp_path, "r", grid("teacher", zip(fams, [True, True])), refs=wide)
    assert mod.paired_across(left, "student", superset, "teacher")["paired"]["prompts"] == 2
    diverged = write_dir(tmp_path, "x", grid("teacher", zip(fams, [True, True])), refs=changed)
    with pytest.raises(SystemExit, match="Reference manifest"):
        mod.paired_across(left, "student", diverged, "teacher")


def _manifests(tmp_path, left_rows, right_rows, *, bad_digest=False):
    refs = []
    for name, rows in (("l", left_rows), ("r", right_rows)):
        data = json.dumps({"rows": rows}).encode()
        (tmp_path / f"{name}.json").write_bytes(data)
        digest = __import__("hashlib").sha256(data).hexdigest()
        refs.append({"path": f"{name}.json", "sha256": "0" * 64 if bad_digest else digest})
    return refs


def test_relaxed_check_refuses_missing_prompt_and_unverified_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    fams = ["f0", "f1"]
    both = {"p0": {"png_sha256": "a"}, "p1": {"png_sha256": "b"}}
    left_refs, right_refs = _manifests(tmp_path, both, {"p0": {"png_sha256": "a"}})
    left = write_dir(tmp_path, "l1", grid("student", zip(fams, [False, True])), refs=left_refs)
    right = write_dir(tmp_path, "r1", grid("teacher", zip(fams, [True, True])), refs=right_refs)
    with pytest.raises(SystemExit, match="Reference manifest"):
        mod.paired_across(left, "student", right, "teacher")
    left_refs, right_refs = _manifests(tmp_path, both, both, bad_digest=True)
    left = write_dir(tmp_path, "l2", grid("student", zip(fams, [False, True])), refs=left_refs)
    right = write_dir(tmp_path, "r2", grid("teacher", zip(fams, [True, True])), refs=right_refs)
    with pytest.raises(SystemExit, match="Reference manifest"):
        mod.paired_across(left, "student", right, "teacher")


def test_report_records_both_manifests_and_whether_rule_was_used(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    fams = ["f0", "f1"]
    both = {"p0": {"png_sha256": "a"}, "p1": {"png_sha256": "b"}}
    wide = {**both, "p2": {"png_sha256": "c"}}
    left_refs, right_refs = _manifests(tmp_path, both, wide)
    left = write_dir(tmp_path, "l", grid("student", zip(fams, [False, True])), refs=left_refs)
    right = write_dir(tmp_path, "r", grid("teacher", zip(fams, [True, True])), refs=right_refs)
    report = mod.paired_across(left, "student", right, "teacher")
    assert report["references_left"] == left_refs and report["references_right"] == right_refs
    assert report["per_prompt_reference_digest_rule"] is True
    same = write_dir(tmp_path, "s", grid("teacher", zip(fams, [True, True])), refs=left_refs)
    assert (
        mod.paired_across(left, "student", same, "teacher")["per_prompt_reference_digest_rule"]
        is False
    )
