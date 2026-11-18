import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from bufogen.generation import BufoGeneration, Manifest
from bufogen.settings import ROOT

REFERENCES_VERSION = "bufo-references-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative_to_root(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(ROOT).resolve()))
    except ValueError:
        return str(path)


def build_reference_manifest(run_dir: Path, out_dir: Path, *, backend: str) -> dict:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    settings = json.loads((run_dir / "benchmark.json").read_text())
    selected: dict[str, BufoGeneration] = {}
    for generation in Manifest(run_dir / "manifest.jsonl").read():
        if generation.backend != backend:
            continue
        if generation.prompt_id in selected:
            raise ValueError(f"duplicate reference row for {generation.prompt_id}")
        selected[generation.prompt_id] = generation

    target = out_dir / "references.json"
    existing = json.loads(target.read_text()) if target.exists() else None

    rows = {}
    pending_copies = []
    for prompt_id, generation in selected.items():
        metadata = generation.benchmark_metadata or {}
        source = Path(generation.png_path) if generation.png_path else None
        if source is not None and not source.exists():
            raise FileNotFoundError(f"reference PNG missing for {prompt_id}: {source}")
        has_png = source is not None
        rows[prompt_id] = {
            "png_path": f"images/{prompt_id}.png" if has_png else None,
            "png_sha256": _sha256(source) if has_png else None,
            "generation_id": generation.id,
            "valid": generation.valid,
            "status": metadata.get("status"),
            "split": metadata.get("split"),
            "family": metadata.get("family"),
        }
        if has_png:
            pending_copies.append((source, out_dir / "images" / f"{prompt_id}.png"))

    data = {
        "version": REFERENCES_VERSION,
        "source_run_id": settings["run_id"],
        "source_configuration_hash": settings["configuration_hash"],
        "source_run_dir": _relative_to_root(run_dir),
        "backend": backend,
        "model": next((g.model for g in selected.values()), None),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    if existing is not None:
        if {k: v for k, v in existing.items() if k != "created_at"} != {
            k: v for k, v in data.items() if k != "created_at"
        }:
            raise FileExistsError(f"Reference manifest differs: {target}")
        return existing

    for source, destination in pending_copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n")
    return data


def merge_reference_manifests(paths: list[Path], out_dir: Path, *, replace: bool = False) -> dict:
    """Union several verified reference manifests into one.

    A duplicate prompt id is an error unless replace=True, where the later source's row wins and
    its source entry records the ids as replaced_prompt_ids.
    """
    paths = [Path(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("merge needs at least two manifests")
    out_dir = Path(out_dir)
    rows, sources, copies = {}, [], {}
    first = None
    for path in paths:
        data = load_reference_manifest(path)
        first = first or data
        if data.get("backend") != first.get("backend") or data.get("model") != first.get("model"):
            raise ValueError("cannot merge manifests from different backends or models")
        replaced = []
        for prompt_id, entry in data["rows"].items():
            if prompt_id in rows:
                if not replace:
                    raise ValueError(f"duplicate reference row for {prompt_id}")
                replaced.append(prompt_id)
            rows[prompt_id] = dict(entry)
            copies.pop(prompt_id, None)
            if entry["png_path"] is not None:
                copies[prompt_id] = (
                    Path(data["base_dir"]) / entry["png_path"],
                    out_dir / entry["png_path"],
                )
        source = {
            "path": _relative_to_root(path),
            "source_run_id": data["source_run_id"],
            "sha256": _sha256(path),
        }
        if replaced:
            source["replaced_prompt_ids"] = replaced
        sources.append(source)
    target = out_dir / "references.json"
    if target.exists():
        raise FileExistsError(f"Reference manifest exists: {target}")
    merged = {
        "version": REFERENCES_VERSION,
        "source_run_id": "+".join(s["source_run_id"] for s in sources),
        "source_configuration_hash": digest_of(
            [json.loads(p.read_text())["source_configuration_hash"] for p in paths]
        ),
        "source_run_dir": None,
        "backend": first["backend"],
        "model": first["model"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "rows": rows,
    }
    for source, destination in copies.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=2) + "\n")
    return merged


def digest_of(values: list) -> str:
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def load_reference_manifest(path: Path) -> dict:
    """Load and verify a reference manifest.

    The returned dict carries an in-memory "base_dir" key (not stored on disk)
    that attach_references uses to resolve image paths.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("version") != REFERENCES_VERSION:
        raise ValueError(f"unsupported reference manifest version: {data.get('version')}")
    base = path.parent.resolve()
    for prompt_id, entry in data["rows"].items():
        if entry["png_path"] is None:
            continue
        image = base / entry["png_path"]
        if not image.exists() or _sha256(image) != entry["png_sha256"]:
            raise ValueError(f"reference image changed: {prompt_id}")
    data["base_dir"] = str(base)
    return data


def attach_references(
    rows: list[BufoGeneration], manifest: dict | None, *, self_reference: bool = False
) -> list[BufoGeneration]:
    if manifest is not None and "base_dir" not in manifest:
        raise ValueError("manifest must come from load_reference_manifest")
    for row in rows:
        if row.benchmark_metadata is None:
            row.benchmark_metadata = {}
        reference = None
        if self_reference:
            if row.png_path and Path(row.png_path).exists():
                reference = {
                    "png_path": row.png_path,
                    "png_sha256": _sha256(Path(row.png_path)),
                    "generation_id": row.id,
                    "is_self": True,
                }
        else:
            entry = (manifest or {}).get("rows", {}).get(row.prompt_id)
            if entry and entry["png_path"] is not None:
                reference = {
                    "png_path": str((Path(manifest["base_dir"]) / entry["png_path"]).resolve()),
                    "png_sha256": entry["png_sha256"],
                    "generation_id": entry["generation_id"],
                    "is_self": False,
                }
        row.benchmark_metadata["reference"] = reference
    return rows
