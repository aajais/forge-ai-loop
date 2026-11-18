import json
from pathlib import Path

import pytest
from PIL import Image

from bufogen.evaluation.references import (
    REFERENCES_VERSION,
    attach_references,
    build_reference_manifest,
    load_reference_manifest,
)
from bufogen.generation import BufoGeneration, Manifest


def png(path: Path, color: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)
    return str(path)


def row(prompt_id, backend, png_path, **meta):
    return BufoGeneration(
        prompt_id=prompt_id,
        prompt="p",
        backend=backend,
        model="m",
        group_id="g",
        svg="",
        valid=png_path is not None,
        latency_s=0.0,
        png_path=png_path,
        benchmark_metadata={"status": "ok", "split": "dev", "family": "f", **meta},
    )


def make_run(tmp_path, rows=None):
    run = tmp_path / "run"
    run.mkdir()
    (run / "benchmark.json").write_text(
        json.dumps({"run_id": "r1", "configuration_hash": "abc", "configuration": {}})
    )
    if rows is None:
        rows = [
            row("a", "sol", png(run / "pngs/a.png", "red")),
            row("b", "sol", png(run / "pngs/b.png", "blue")),
            row("a", "other", png(run / "pngs/a2.png", "green")),
        ]
    manifest = Manifest(run / "manifest.jsonl")
    for r in rows:
        manifest.append(r)
    return run


def test_build_manifest_copies_pngs_and_records_hashes(tmp_path):
    run = make_run(tmp_path)
    out = tmp_path / "out"
    data = build_reference_manifest(run, out, backend="sol")
    assert data["version"] == REFERENCES_VERSION
    assert data["source_run_id"] == "r1"
    assert data["source_configuration_hash"] == "abc"
    assert data["backend"] == "sol" and data["model"] == "m"
    assert set(data["rows"]) == {"a", "b"}
    entry = data["rows"]["a"]
    assert entry["png_path"] == "images/a.png"
    assert (out / "images/a.png").read_bytes() == (run / "pngs/a.png").read_bytes()
    assert len(entry["png_sha256"]) == 64
    assert entry["status"] == "ok" and entry["split"] == "dev" and entry["family"] == "f"
    assert json.loads((out / "references.json").read_text()) == data


def test_build_manifest_rejects_duplicate_backend_rows(tmp_path):
    run = tmp_path / "dup"
    run.mkdir()
    (run / "benchmark.json").write_text(json.dumps({"run_id": "r", "configuration_hash": "h"}))
    manifest = Manifest(run / "manifest.jsonl")
    manifest.append(row("a", "sol", png(run / "1.png", "red")))
    manifest.append(row("a", "sol", png(run / "2.png", "blue")))
    with pytest.raises(ValueError, match="duplicate reference row for a"):
        build_reference_manifest(run, tmp_path / "out", backend="sol")


def test_build_manifest_refuses_to_change_existing(tmp_path):
    run = make_run(tmp_path)
    out = tmp_path / "out"
    first = build_reference_manifest(run, out, backend="sol")
    assert build_reference_manifest(run, out, backend="sol") == first
    second = tmp_path / "second"
    second.mkdir()
    other = make_run(second, [row("a", "sol", png(second / "x.png", "black"))])
    with pytest.raises(FileExistsError):
        build_reference_manifest(other, out, backend="sol")


def test_build_manifest_records_missing_png_as_null(tmp_path):
    run = make_run(tmp_path, [row("a", "sol", None)])
    data = build_reference_manifest(run, tmp_path / "out", backend="sol")
    assert data["rows"]["a"]["png_path"] is None
    assert data["rows"]["a"]["png_sha256"] is None
    attached = attach_references(
        [row("a", "x", None)], load_reference_manifest(tmp_path / "out/references.json")
    )
    assert attached[0].benchmark_metadata["reference"] is None


def test_attach_references_verifies_png_hash(tmp_path):
    run = make_run(tmp_path)
    out = tmp_path / "out"
    build_reference_manifest(run, out, backend="sol")
    manifest = load_reference_manifest(out / "references.json")
    rows = attach_references([row("a", "cand", None)], manifest)
    ref = rows[0].benchmark_metadata["reference"]
    assert set(ref) == {"png_path", "png_sha256", "generation_id", "is_self"}
    assert Path(ref["png_path"]) == (out / "images/a.png").resolve()
    assert ref["is_self"] is False
    Image.new("RGB", (4, 4), "yellow").save(out / "images/a.png")
    with pytest.raises(ValueError, match="reference image changed: a"):
        load_reference_manifest(out / "references.json")


def test_self_reference_uses_own_png(tmp_path):
    own = png(tmp_path / "own.png", "red")
    rows = attach_references([row("a", "sol", own)], None, self_reference=True)
    ref = rows[0].benchmark_metadata["reference"]
    assert ref["png_path"] == own and ref["is_self"] is True
    assert ref["generation_id"] == rows[0].id and len(ref["png_sha256"]) == 64


def test_missing_prompt_gets_none_reference(tmp_path):
    run = make_run(tmp_path)
    build_reference_manifest(run, tmp_path / "out", backend="sol")
    manifest = load_reference_manifest(tmp_path / "out/references.json")
    rows = attach_references([row("zzz", "cand", None)], manifest)
    assert rows[0].benchmark_metadata["reference"] is None


def test_build_manifest_raises_when_png_file_missing(tmp_path):
    run = make_run(tmp_path)
    (run / "pngs/a.png").unlink()
    with pytest.raises(FileNotFoundError, match="a"):
        build_reference_manifest(run, tmp_path / "out", backend="sol")


def test_attach_requires_loaded_manifest():
    with pytest.raises(ValueError, match="load_reference_manifest"):
        attach_references([row("a", "x", None)], {"rows": {}})


def _make_manifest(tmp_path, name, ids, run_id, colour="red"):
    run = tmp_path / name
    run.mkdir()
    (run / "benchmark.json").write_text(
        json.dumps({"run_id": run_id, "configuration_hash": "abc", "configuration": {}})
    )
    manifest = Manifest(run / "manifest.jsonl")
    for i in ids:
        manifest.append(row(i, "sol", png(run / f"pngs/{i}.png", colour)))
    out = tmp_path / f"refs-{name}"
    build_reference_manifest(run, out, backend="sol")
    return out / "references.json"


def test_merge_reference_manifests_union_and_duplicate_refusal(tmp_path):
    from bufogen.evaluation.references import merge_reference_manifests

    one = _make_manifest(tmp_path, "one", ["a", "b"], "r1")
    two = _make_manifest(tmp_path, "two", ["c"], "r2")
    out = tmp_path / "merged"
    merged = merge_reference_manifests([one, two], out)
    assert sorted(merged["rows"]) == ["a", "b", "c"]
    assert merged["version"] == REFERENCES_VERSION
    assert merged["source_run_id"] == "r1+r2"
    assert [s["source_run_id"] for s in merged["sources"]] == ["r1", "r2"]
    assert (
        merged["sources"][0]["sha256"] == __import__("hashlib").sha256(one.read_bytes()).hexdigest()
    )
    loaded = load_reference_manifest(out / "references.json")
    assert sorted(loaded["rows"]) == ["a", "b", "c"]
    assert all((out / "images" / f"{i}.png").exists() for i in "abc")
    dup = _make_manifest(tmp_path, "dup", ["b"], "r3")
    with pytest.raises(ValueError, match="duplicate"):
        merge_reference_manifests([one, dup], tmp_path / "bad")
    assert not (tmp_path / "bad" / "references.json").exists()


def test_merge_reference_manifests_replace_mode(tmp_path):
    from bufogen.evaluation.references import merge_reference_manifests

    one = _make_manifest(tmp_path, "one", ["a", "b"], "r1")
    two = _make_manifest(tmp_path, "two", ["c"], "r2")
    fix = _make_manifest(tmp_path, "fix", ["b"], "r3", colour="blue")
    fix_sha = load_reference_manifest(fix)["rows"]["b"]["png_sha256"]
    old_sha = load_reference_manifest(one)["rows"]["b"]["png_sha256"]
    # Default stays strict.
    with pytest.raises(ValueError, match="duplicate"):
        merge_reference_manifests([one, two, fix], tmp_path / "strict")
    out = tmp_path / "merged"
    merged = merge_reference_manifests([one, two, fix], out, replace=True)
    assert sorted(merged["rows"]) == ["a", "b", "c"]
    assert merged["rows"]["b"]["png_sha256"] == fix_sha != old_sha
    assert load_reference_manifest(out / "references.json")["rows"]["b"]["png_sha256"] == fix_sha
    assert [s.get("replaced_prompt_ids") for s in merged["sources"]] == [None, None, ["b"]]
    # Replacing within replace mode needs something to replace only when duplicated; no-dup is fine.
    ok = merge_reference_manifests([one, two], tmp_path / "plain", replace=True)
    assert all("replaced_prompt_ids" not in s for s in ok["sources"])
