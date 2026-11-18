import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from bufogen.evaluation.statistics import astra_pass, paired_family_interval

DOCS = Path(__file__).resolve().parents[1] / "docs/forge-gap-work/paired_statistics.py"
spec = importlib.util.spec_from_file_location("docs_paired_statistics", DOCS)
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


def row(prompt_id, backend, family, passed):
    return SimpleNamespace(
        prompt_id=prompt_id,
        backend=backend,
        valid=True,
        benchmark_metadata={"family": family, "status": "completed"},
        scores={
            "scene_quality": {"quality_failure": not passed},
            "astra_judge": {"strict_pass": passed},
        },
    )


def grid(families=4, per_family=3):
    rows = []
    for f in range(families):
        for v in range(per_family):
            index = f * per_family + v
            rows.append(row(f"p{index}", "student", f"fam{f}", index % 2 == 0))
            rows.append(row(f"p{index}", "teacher", f"fam{f}", index % 3 != 0))
    return rows


def test_inferred_clusters_match_docs_implementation():
    rows = grid()
    expected = docs.paired_family_interval(rows, "student", "teacher", families=4, per_family=3)
    actual = paired_family_interval(rows, "student", "teacher", pass_fn=astra_pass)
    assert actual["balanced"] is True
    assert actual["prompts_per_family"] == {f"fam{f}": 3 for f in range(4)}
    assert actual["prompt_difference"] == actual["difference"]
    for key in ("balanced", "prompts_per_family", "prompt_difference"):
        actual.pop(key)
    assert actual == expected
    assert actual["difference"] == expected["difference"]
    assert actual["ci95"] == expected["ci95"]
    assert actual["replicates"] == 20000 and actual["seed"] == 20261015


def test_uneven_families_weight_each_family_equally():
    rows = [r for r in grid() if r.prompt_id != "p0"]
    result = paired_family_interval(rows, "student", "teacher", pass_fn=astra_pass)
    assert result["balanced"] is False
    assert result["prompts_per_family"] == {"fam0": 2, "fam1": 3, "fam2": 3, "fam3": 3}
    assert result["prompts"] == 11 and result["families"] == 4
    # Family means are unweighted; the per-prompt mean is reported beside them.
    assert result["difference"] == pytest.approx(sum(result["family_differences"].values()) / 4)
    assert result["prompt_difference"] == pytest.approx(
        sum(astra_pass(r) for r in rows if r.backend == "student") / 11
        - sum(astra_pass(r) for r in rows if r.backend == "teacher") / 11
    )
    assert result["prompt_difference"] != result["difference"]


def test_incomplete_pairs_rejected():
    rows = [r for r in grid() if not (r.prompt_id == "p0" and r.backend == "student")]
    with pytest.raises(ValueError, match="paired"):
        paired_family_interval(rows, "student", "teacher", pass_fn=astra_pass)


def test_astra_pass_requires_literal_true():
    assert astra_pass(SimpleNamespace(scores={"astra_judge": {"strict_pass": True}})) == 1
    assert astra_pass(SimpleNamespace(scores={"astra_judge": {"strict_pass": 1}})) == 0
    assert astra_pass(SimpleNamespace(scores={})) == 0
