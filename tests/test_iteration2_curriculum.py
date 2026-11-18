"""Frozen curriculum must preserve split isolation and bind correction evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "runs/forge-gap/training-fit-selected/scored.jsonl").exists(),
    reason="requires local experiment evidence under runs/forge-gap/training-fit-selected",
)


def module():
    path = ROOT / "docs/forge-iteration2/build_curriculum.py"
    assert path.exists(), "Iteration-two curriculum builder has not been implemented"
    spec = importlib.util.spec_from_file_location("iteration2_curriculum", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frozen_curriculum_counts_lineage_and_new_confirmation_families():
    m = module()
    rows, manifest = m.build(ROOT)
    assert manifest["counts"] == {"train": 80, "validation": 24, "heldout": 48}
    assert len({r["id"] for r in rows}) == 152
    old = [
        *yaml.safe_load((ROOT / "config/forge_prompts.yaml").read_text()),
        *yaml.safe_load((ROOT / "config/forge_gap_prompts.yaml").read_text()),
    ]
    assert not {r["family"] for r in rows if r["split"] == "heldout"} & {r["family"] for r in old}
    assert not {r["id"] for r in rows} & {r["id"] for r in old}
    by_id = {r["id"]: r for r in old}
    for row in rows:
        if row["split"] == "validation":
            original = by_id[row["curriculum_metadata"]["source_prompt_id"]]
            assert (row["prompt"], row["requirements"], row["family"]) == (
                original["prompt"],
                original["requirements"],
                original["family"],
            )


def test_corrections_are_bound_untrusted_train_only_and_have_no_privileged_answers():
    rows, manifest = module().build(ROOT)
    corrections = [r for r in rows if r["curriculum_metadata"]["kind"] == "student_correction"]
    assert len(corrections) == 16
    for r in corrections:
        meta = r["curriculum_metadata"]
        assert r["split"] == "train" and meta["source_prompt_id"].startswith("gap-train-")
        assert meta["critic_model"] == "gpt-6.1-sol"
        assert meta["review_status"] == "unreviewed_automatic_critique"
        assert all(
            len(meta[k]) == 64
            for k in ("source_svg_sha256", "source_png_sha256", "source_row_sha256")
        )
        assert "Automatic critique" in r["prompt"] and "<editable_scene>" in r["prompt"]
        assert "system" not in r and not r.get("turns")
        assert meta["history"] is None and meta["image"] is None
    assert manifest["confirmation_outputs_generated"] is False
    for r in corrections:
        assert 'id="bufo-character"' not in r["prompt"]
        assert 'id="bufo-fullbody"' not in r["prompt"]


def test_freeze_is_idempotent_and_refuses_modified_curriculum_or_source(tmp_path):
    m = module()
    rows, manifest = m.build(ROOT)
    m.freeze(rows, manifest, tmp_path)
    m.freeze(rows, manifest, tmp_path)
    before = (tmp_path / "config/forge_iteration2_prompts.yaml").read_bytes()
    changed = copy.deepcopy(rows)
    changed[0]["prompt"] += " changed"
    with pytest.raises(RuntimeError, match="frozen"):
        m.freeze(changed, manifest, tmp_path)
    assert (tmp_path / "config/forge_iteration2_prompts.yaml").read_bytes() == before
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["sources"][next(iter(manifest["sources"]))] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen"):
        m.freeze(rows, changed_manifest, tmp_path)


def test_rejects_correction_cross_split_and_tampered_image():
    m = module()
    row = m.read_corrections(ROOT)[0]
    bad = copy.deepcopy(row)
    bad["benchmark_metadata"]["split"] = "heldout"
    with pytest.raises(ValueError, match="train"):
        m.correction_row(bad, ROOT, m.gap_rows(ROOT))
    bad = copy.deepcopy(row)
    bad["benchmark_metadata"]["png_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="PNG"):
        m.correction_row(bad, ROOT, m.gap_rows(ROOT))


def test_rejects_duplicate_and_missing_correction_family():
    m = module()
    source = m.read_corrections(ROOT)
    with pytest.raises(ValueError, match="one correction"):
        m.build(ROOT, correction_sources=source[:-1])
    with pytest.raises(ValueError, match="one correction"):
        m.build(ROOT, correction_sources=[*source, source[0]])
