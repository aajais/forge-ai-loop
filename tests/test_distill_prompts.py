import re
from pathlib import Path

import yaml

from bufogen.evaluation.benchmark import load_benchmark

ROOT = Path(__file__).resolve().parents[1]
DISTILL = ROOT / "config" / "distill_prompts.yaml"
ORIGINAL = ROOT / "config" / "forge_gap_prompts.yaml"
FAMILIES = {"airport_marshaller", "florist_arranging", "ice_cream_cart", "locksmith"}
VARIANTS = ("twohands", "nightscene", "caption", "closeup", "pair", "weather")


def _rows(path):
    # Stage A set only: the Stage B augmented block is covered by tests/test_augment_prompts.py.
    return [r for r in yaml.safe_load(path.read_text()) if not r["id"].startswith("aug-train-")]


def test_distill_prompts_load_and_counts():
    rows = [r for r in load_benchmark(DISTILL) if not r["id"].startswith("aug-train-")]
    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
    assert {k: len(v) for k, v in by_split.items()} == {
        "train": 128,
        "validation": 48,
        "heldout": 80,
    }
    assert len({row["family"] for row in rows}) == 36
    assert {row["family"] for row in by_split["validation"]} == FAMILIES


def test_original_entries_unchanged():
    new_rows = _rows(DISTILL)
    new = {row["id"]: row for row in new_rows}
    old = _rows(ORIGINAL)
    assert len(old) == 232
    splits = {}
    for row in old:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    assert splits == {"train": 128, "validation": 24, "heldout": 80}
    assert len(new_rows) == 232 + 24
    assert [r["id"] for r in new_rows][:232] == [r["id"] for r in old]
    for row in old:
        assert new[row["id"]] == row


def test_new_validation_rows_have_concrete_requirements():
    old_ids = {row["id"] for row in _rows(ORIGINAL)}
    added = [row for row in _rows(DISTILL) if row["id"] not in old_ids]
    assert {row["id"] for row in added} == {
        f"gap-validation-{family}-{variant}" for family in FAMILIES for variant in VARIANTS
    }
    assert len(added) == 24
    assert len({row["id"] for row in added}) == 24
    all_ids = [row["id"] for row in _rows(DISTILL)]
    assert len(all_ids) == 256 and len(set(all_ids)) == 256
    for row in added:
        assert row["split"] == "validation"
        assert 3 <= len(row["requirements"]) <= 5
        assert row["prompt"].endswith("Visible details: " + "; ".join(row["requirements"]) + ".")
    for row in _rows(DISTILL):
        assert not any(";" in req for req in row["requirements"]), row["id"]
    captions = [row for row in added if row["id"].endswith("-caption")]
    assert len(captions) == 4
    for row in captions:
        literals = re.findall(r'"([^"]+)"', row["prompt"])
        assert len(literals) == 1, row["id"]
        assert len(literals[0].split()) <= 3
        assert any(literals[0] in req for req in row["requirements"]), row["id"]


def test_no_requirement_copied_from_heldout_or_validation():
    rows = _rows(DISTILL)
    old_ids = {row["id"] for row in _rows(ORIGINAL)}
    for split in ("heldout", "validation"):
        held = {r for row in rows if row["split"] == split for r in row["requirements"]}
        for row in rows:
            if row["id"] not in old_ids and row["split"] != split:
                assert not held & set(row["requirements"]), (split, row["id"])
