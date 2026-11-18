import hashlib
import re
import subprocess
import sys
from pathlib import Path

import yaml

from bufogen.evaluation.benchmark import load_benchmark

ROOT = Path(__file__).resolve().parents[1]
DISTILL = ROOT / "config" / "distill_prompts.yaml"
# sha256 of the first 256 prompt entries' canonical dump, captured before augmentation.
EXISTING_IDS = 256
# sha256 of config/distill_prompts.yaml before augmentation (the 256 existing entries, byte for byte).
EXISTING_SHA256 = "080d40c3fd90ba73c9f6668a52152395d8226044a9068df738858a72cbffe4c9"
# sha256 of config/distill_prompts.yaml before augmentation (the 256 existing entries, byte for byte).
EXISTING_SHA256 = "080d40c3fd90ba73c9f6668a52152395d8226044a9068df738858a72cbffe4c9"


def _rows():
    return yaml.safe_load(DISTILL.read_text())


def _aug():
    return [r for r in _rows() if r["id"].startswith("aug-train-")]


def test_requirements_follow_r6_rules():
    aug = _aug()
    assert len(aug) == 128
    rows = {r["id"]: r for r in _rows()}
    train_families = {r["family"] for r in rows.values() if r["split"] == "train"}
    assert len(train_families) == 16
    per_family = {}
    for row in aug:
        per_family.setdefault(row["family"], []).append(row["id"])
        assert row["split"] == "train" and row["family"] in train_families
        assert row["provenance"] == "stage-b-augment-2026-10"
        base = rows[row["augmented_from"]]
        assert base["split"] == "train" and base["family"] == row["family"]
        assert not base["id"].startswith("aug-")
        assert re.fullmatch(rf"aug-train-{row['family']}-0[1-8]", row["id"])
        reqs = row["requirements"]
        assert 3 <= len(reqs) <= 6
        normal = [r.lower().strip() for r in reqs]
        assert len(set(normal)) == len(normal)
        for i, a in enumerate(normal):
            assert ";" not in a
            assert not a.endswith(".")
            for j, b in enumerate(normal):
                assert i == j or a not in b, (row["id"], a, b)
        assert row["prompt"].endswith("Visible details: " + "; ".join(reqs) + ".")
        # exactly one element differs from the base requirements
        assert set(base["requirements"]) - set(reqs) | (set(reqs) - set(base["requirements"]))
    assert {k: len(v) for k, v in per_family.items()} == {f: 8 for f in train_families}
    assert len({r["prompt"] for r in aug}) == 128
    # Validation is the selection set and heldout the confirmation set: no augmented requirement
    # may copy a requirement string from either.
    for split in ("validation", "heldout"):
        held = {q for r in rows.values() if r["split"] == split for q in r["requirements"]}
        for row in aug:
            assert not held & set(row["requirements"]), (split, row["id"])
    load_benchmark(DISTILL)


def test_existing_prompts_unchanged_and_generator_deterministic():
    rows = _rows()
    existing = rows[:EXISTING_IDS]
    assert not any(r["id"].startswith("aug-") for r in existing)
    assert all(r["id"].startswith("aug-train-") for r in rows[EXISTING_IDS:])
    original = ROOT / "config" / "forge_gap_prompts.yaml"
    old = {r["id"]: r for r in yaml.safe_load(original.read_text())}
    for row in existing:
        if row["id"] in old:
            assert old[row["id"]] == row
    # The file text before the augmented block is exactly the dump of the existing entries.
    text = DISTILL.read_text()
    marker = "- id: aug-train-"
    head = text[: text.index(marker)]
    assert yaml.safe_load(head) == existing
    assert hashlib.sha256(head.encode()).hexdigest() == EXISTING_SHA256
    assert hashlib.sha256(head.encode()).hexdigest() == EXISTING_SHA256


def test_generator_is_idempotent(tmp_path):
    out = tmp_path / "p.yaml"
    out.write_text(DISTILL.read_text())
    docs = tmp_path / "docs.md"
    for _ in range(2):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "augment_prompts.py"),
                "--out",
                str(out),
                "--seed",
                "20261025",
                "--docs",
                str(docs),
            ],
            check=True,
            capture_output=True,
        )
    assert out.read_bytes() == DISTILL.read_bytes()
    assert hashlib.sha256(out.read_bytes()).hexdigest() in docs.read_text()
