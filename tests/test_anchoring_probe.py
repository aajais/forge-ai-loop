import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/anchoring_probe.py"
spec = importlib.util.spec_from_file_location("anchoring_probe", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SVG = '<svg><text x="1">READY</text><text x="2">OTHER</text></svg>'


def test_mutate_text_replaces_exact_content():
    out = mod.mutate_text(SVG, "READY", "LATER")
    assert '<text x="1">LATER</text>' in out and "READY" not in out
    assert "OTHER" in out


def test_mutate_text_absent():
    with pytest.raises(ValueError, match="absent"):
        mod.mutate_text(SVG, "FRESH", "STALE")


def test_mutate_text_ambiguous():
    with pytest.raises(ValueError, match="ambiguous"):
        mod.mutate_text(SVG + "<text>READY</text>", "READY", "LATER")


def test_mutate_text_ignores_substring_matches():
    with pytest.raises(ValueError, match="absent"):
        mod.mutate_text("<svg><text>READY NOW</text></svg>", "READY", "LATER")


def test_required_text():
    assert mod.required_text(["x", "A yellow board reads exactly GATE 4"]) == (
        "A yellow board reads exactly GATE 4",
        "GATE 4",
    )
    assert mod.required_text(["nothing"]) is None


def row(pid, split, status="completed"):
    return {
        "prompt_id": pid,
        "split": split,
        "status": status,
        "requirements": ["A top sign reads exactly READY"],
    }


def test_select_rows_takes_validation_and_first_train():
    rows = [row("t3", "train"), row("t1", "train"), row("v1", "validation"), row("h1", "heldout")]
    rows.append(row("t0", "train", status="failed"))
    picked = mod.select_rows(rows, train_n=1)
    assert [r["prompt_id"] for r in picked] == ["v1", "t1"]


def test_manifest_entry_shape():
    e = mod.manifest_entry(
        "p", "validation", "READY", "LATER", "A top sign reads exactly READY", "a", "b"
    )
    assert e == {
        "prompt_id": "p",
        "split": "validation",
        "original_word": "READY",
        "replacement": "LATER",
        "altered_requirement": "A top sign reads exactly READY",
        "reference_sha256": "a",
        "near_copy_sha256": "b",
    }


def verdict(sat, vs, strict=False, status="ok"):
    return {
        "status": status,
        "vs_reference": vs,
        "strict_pass": strict,
        "prompt_checks": [
            {"requirement": "R", "satisfied": sat},
            {"requirement": "other", "satisfied": True},
        ],
    }


def test_build_report_counts_and_gates():
    entries = [{"prompt_id": f"p{i}", "altered_requirement": "R"} for i in range(12)]
    verdicts = {f"p{i}": verdict(False, "worse") for i in range(12)}
    verdicts["p0"] = verdict(True, "equal", strict=True)
    verdicts["p1"] = verdict(False, "equal")
    rep = mod.build_report(entries, verdicts)
    assert rep["rows"] == 12
    assert rep["altered_unsatisfied"] == 11
    assert rep["vs_reference_worse"] == 10
    assert rep["strict_pass"] == 1
    assert rep["judge_errors"] == 0
    assert rep["gate_unsatisfied"] is True and rep["gate_worse"] is True
    assert rep["gate_passed"] is True


def test_build_report_errors_fail_gate():
    entries = [{"prompt_id": f"p{i}", "altered_requirement": "R"} for i in range(12)]
    verdicts = {f"p{i}": verdict(False, "worse") for i in range(12)}
    verdicts["p0"] = {"status": "judge_error", "prompt_checks": [], "strict_pass": False}
    verdicts["p1"] = verdict(True, "worse")
    rep = mod.build_report(entries, verdicts)
    assert rep["judge_errors"] == 1
    assert rep["altered_unsatisfied"] == 10
    assert rep["gate_unsatisfied"] is False and rep["gate_passed"] is False


def test_forbidden_words_and_replacement_collision():
    prompts = [
        {"requirements": ["A chalkboard menu reads exactly HOT", "x"]},
        {"requirements": ["A top sign reads exactly READY"]},
    ]
    forbidden = mod.forbidden_words(prompts)
    assert forbidden == {"HOT", "READY"}
    with pytest.raises(ValueError, match="collides"):
        mod.check_replacement("ICE", "HOT", forbidden)
    with pytest.raises(ValueError, match="collides"):
        mod.check_replacement("ICE", "ICE", forbidden)
    mod.check_replacement("ICE", "COLD", forbidden)


def test_configured_replacements_do_not_collide_with_real_prompts():
    from bufogen.evaluation.benchmark import load_benchmark

    forbidden = mod.forbidden_words(load_benchmark(mod.ROOT / "config/distill_prompts.yaml"))
    for old, new in mod.REPLACEMENTS.items():
        mod.check_replacement(old, new, forbidden)
