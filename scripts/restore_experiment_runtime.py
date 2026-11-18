"""Restore the exact code/config an experiment run was frozen with.

Usage: python scripts/restore_experiment_runtime.py RUN_DIR TARGET [--sync | --refresh-manifest]
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCAFFOLD = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".env.example",
    "src",
    "scripts",
    "config",
    "assets",
    "tests",
    "docs",
)
EXCLUDE = (
    "__pycache__",
    ".venv",
    "runs",
    "wandb",
    ".env",
    ".env.*",
    ".pytest_cache",
    ".ruff_cache",
    ".superpowers",
)
KEEP = {".env.example"}

WANDB_PACKAGES = ("openai-agents", "weave", "cwsandbox", "openpipe-art", "openai")

README = """# Restored experiment runtime

Restored from `{run_dir}` (run `{run_id}`). See RUNTIME.json for what came from
the frozen copy vs the current tree, and env.frozen for judge/arm model settings.

Build the environment (done by `--sync`; RUNTIME.json records installed packages
and their differences from the recorded ones):

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --locked --extra rl

Run tests and re-score against this tree:

    set -a; . ./env.frozen; set +a
    BUFO_ROOT=$PWD .venv/bin/python -m pytest -q <tests>
    BUFO_ROOT=$PWD .venv/bin/bufogen --stage benchmark-score --output ./rescore-copy
"""


class RestoreError(ValueError):
    """Raised when a run cannot be faithfully restored."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _excluded(name: str) -> bool:
    if name in KEEP:
        return False
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in EXCLUDE)


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if _excluded(child.name):
            continue
        if child.is_dir():
            _copy_tree(child, dst / child.name)
        else:
            shutil.copy2(child, dst / child.name)


def _scaffold_files(project_root: Path) -> set[str]:
    """Relative paths of every file the scaffold copy takes from the current tree."""

    def walk(path: Path):
        for child in path.iterdir():
            if _excluded(child.name):
                continue
            if child.is_dir():
                yield from walk(child)
            else:
                yield child

    files: set[str] = set()
    for name in SCAFFOLD:
        src = project_root / name
        if src.is_dir():
            files.update(p.relative_to(project_root).as_posix() for p in walk(src))
        elif src.is_file():
            files.add(name)
    return files


def _frozen_files(frozen: Path) -> set[str]:
    if not frozen.is_dir():
        return set()
    return {p.relative_to(frozen).as_posix() for p in frozen.rglob("*") if p.is_file()}


def _from_current_tree(project_root: Path, frozen: Path, skip=()) -> list[str]:
    """Every copied scaffold file that no frozen (or reconstructed) file replaced."""
    return sorted(_scaffold_files(project_root) - _frozen_files(frozen) - set(skip))


def _hash_target(target: Path, listed) -> dict[str, str]:
    """sha256 of each listed file as it exists in the target (not in today's project tree)."""
    return {rel: _sha256(target / rel) for rel in listed if (target / rel).is_file()}


def refresh_manifest(run_dir: Path, target: Path, *, project_root: Path) -> dict:
    """Recompute RUNTIME.json from_current_tree for an existing target; nothing else changes."""
    target = Path(target)
    runtime = json.loads((target / "RUNTIME.json").read_text())
    listed = _from_current_tree(
        Path(project_root), Path(run_dir) / "frozen", runtime.get("reconstructed") or {}
    )
    runtime["from_current_tree"] = _hash_target(target, listed)
    (target / "RUNTIME.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    return runtime


def _env_lines(config: dict) -> list[str]:
    scorer = config.get("scorer") or {}
    lines = [
        f"SCENE_JUDGE_BACKEND={scorer.get('judge_backend', '')}",
        f"SCENE_JUDGE_MODEL={scorer.get('judge_model', '')}",
        "BUFO_WEB_SEARCH=false",
    ]
    for arm in config.get("arms") or []:
        if arm.get("model_env") and arm.get("model"):
            lines.append(f"{arm['model_env']}={arm['model']}")
    return lines


def _copy_all(src: Path, dst: Path) -> None:
    """Unfiltered copy: frozen evidence must land verbatim, whatever the dir names."""
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.is_dir():
            _copy_all(child, dst / child.name)
        else:
            shutil.copy2(child, dst / child.name)


def restore(run_dir: Path, target: Path, *, project_root: Path) -> dict:
    run_dir, target, project_root = Path(run_dir), Path(target), Path(project_root)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise RestoreError(f"target {target} must be an empty directory")
    benchmark = json.loads((run_dir / "benchmark.json").read_text())
    config = benchmark["configuration"]
    hashes: dict[str, str] = config.get("source_hashes") or {}
    frozen = run_dir / "frozen"

    # Frozen runs contain selected sources, so their remaining scaffold must use
    # the same import layout. Overlaying old flat modules on today's packages
    # would verify the stored hashes but produce a broken hybrid checkout.
    legacy_layout = "src/bufogen/svg.py"
    packaged_layout = "src/bufogen/rendering/svg.py"
    legacy_agent = "src/bufogen/agent.py"
    packaged_agent = "src/bufogen/agents/agent.py"
    for old, new in ((legacy_layout, packaged_layout), (legacy_agent, packaged_agent)):
        for recorded, incompatible in ((old, new), (new, old)):
            if recorded in hashes and (project_root / incompatible).is_file():
                raise RestoreError(
                    "Source layout mismatch: restore with --project-root pointing to a "
                    "source checkout matching this experiment. See README's legacy "
                    "experiment restoration instructions."
                )

    # Verify before copying anything so failures leave target untouched.
    restored: dict[str, str] = {}
    for rel, expected in hashes.items():
        path = frozen / rel
        if not path.is_file():
            raise RestoreError(f"missing frozen file: {rel}")
        actual = _sha256(path)
        if actual != expected:
            raise RestoreError(f"sha256 mismatch for {rel}: expected {expected}, got {actual}")
        restored[rel] = actual

    target.mkdir(parents=True, exist_ok=True)
    for name in SCAFFOLD:
        src = project_root / name
        if src.is_dir():
            _copy_tree(src, target / name)
        elif src.is_file():
            shutil.copy2(src, target / name)
    if frozen.is_dir():
        _copy_all(frozen, target)
    for rel, expected in hashes.items():
        copied = target / rel
        actual = _sha256(copied) if copied.is_file() else "<missing>"
        if actual != expected:
            raise RestoreError(
                f"sha256 mismatch after overlay for {rel}: expected {expected}, got {actual}"
            )

    from_current = _hash_target(target, _from_current_tree(project_root, frozen))
    runtime = {
        "run_dir": str(run_dir),
        "run_id": benchmark.get("run_id"),
        "configuration_hash": benchmark.get("configuration_hash"),
        "restored": restored,
        "from_current_tree": from_current,
        "recorded_packages": (config.get("runtime") or {}).get("packages"),
        "installed_packages": None,
        "package_differences": None,
    }
    (target / "RUNTIME.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    (target / "env.frozen").write_text("\n".join(_env_lines(config)) + "\n")
    (target / "README.md").write_text(README.format(run_dir=run_dir, run_id=runtime["run_id"]))
    return runtime


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_freeze(text: str) -> dict[str, str]:
    """Parse `pip freeze` output into {normalized-name: version-or-"@ url"}."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if "==" in line:
            name, _, version = line.partition("==")
            out[_norm(name.strip())] = version.strip()
        elif " @ " in line:
            name, _, url = line.partition(" @ ")
            out[_norm(name.strip())] = f"@ {url.strip()}"
    return out


def package_differences(recorded: dict | None, installed: dict[str, str]) -> dict:
    diff = {}
    for name, version in (recorded or {}).items():
        have = installed.get(_norm(name))
        if have != version:
            diff[name] = {"recorded": version, "installed": have}
    return diff


def _first_created_at(run_dir: Path) -> datetime | None:
    manifest = run_dir / "manifest.jsonl"
    if not manifest.is_file():
        return None
    times = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        try:
            stamp = json.loads(line).get("created_at")
            if stamp:
                times.append(datetime.fromisoformat(stamp))
        except ValueError:
            continue
    return min(times) if times else None


def _wandb_started_at(run: Path) -> datetime | None:
    """UTC start of a wandb run; its directory name is in the machine's local time."""
    try:
        stamp = json.loads((run / "files/wandb-metadata.json").read_text())["startedAt"]
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    m = re.match(r"run-(\d{8}_\d{6})-", run.name)
    if not m:
        return None
    local = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").astimezone()
    return local.astimezone(timezone.utc)


def _wandb_requirements(run_dir: Path, project_root: Path, recorded, installed) -> dict:
    """Best-effort diff vs a wandb requirements.txt captured before the run began."""
    try:
        first = _first_created_at(run_dir)
        if first is None:
            return {"packages": None, "reason": "no created_at in manifest.jsonl"}
        candidates = []
        for req in (project_root / "wandb").glob("run-*/files/requirements.txt"):
            started = _wandb_started_at(req.parent.parent)
            if started is not None and started < first:
                candidates.append((started, req))
        if not candidates:
            return {"packages": None, "reason": "no wandb requirements.txt predates the run"}
        started, req = max(candidates)
        theirs: dict[str, set[str]] = {}
        for name, version in parse_freeze(req.read_text()).items():
            theirs.setdefault(name, set()).add(version)
        # parse_freeze keeps the last duplicate; re-scan to keep every pinned version
        for line in req.read_text().splitlines():
            if "==" in line:
                name, _, version = line.partition("==")
                theirs.setdefault(_norm(name.strip()), set()).add(version.strip())
        packages = {}
        for name in WANDB_PACKAGES:
            versions = sorted(theirs.get(name, ()))
            have = installed.get(name)
            if versions and have in versions:
                continue
            if not versions and have is None:
                continue
            packages[name] = {
                "wandb": versions[0] if len(versions) == 1 else (versions or None),
                "recorded": (recorded or {}).get(name),
                "installed": have,
            }
        return {"source": str(req), "run_started_utc": started.isoformat(), "packages": packages}
    except Exception as exc:  # best effort; never fail the restore
        return {"packages": None, "reason": f"{type(exc).__name__}: {exc}"}


def sync_runtime(target: Path, *, project_root: Path, run_dir: Path | None = None) -> dict:
    """Build TARGET/.venv from the restored lock and record package differences."""
    target = Path(target).resolve()
    venv = target / ".venv"
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}
    subprocess.run(["uv", "sync", "--locked", "--extra", "rl"], cwd=target, env=env, check=True)
    frozen = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(venv / "bin/python")],
        cwd=target,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    installed = parse_freeze(frozen.stdout)
    runtime = json.loads((target / "RUNTIME.json").read_text())
    runtime["installed_packages"] = installed
    runtime["package_differences"] = package_differences(runtime["recorded_packages"], installed)
    run_dir = Path(run_dir or runtime["run_dir"])
    runtime["wandb_requirements"] = _wandb_requirements(
        run_dir, Path(project_root), runtime["recorded_packages"], installed
    )
    (target / "RUNTIME.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    return runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--sync", action="store_true", help="uv sync the target and record packages"
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Recompute from_current_tree in an existing TARGET's RUNTIME.json (no copy, no sync)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Current tree to copy scaffolding from (default: this repository)",
    )
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    if args.refresh_manifest:
        runtime = refresh_manifest(args.run_dir, args.target, project_root=project_root)
        print(f"from_current_tree: {len(runtime['from_current_tree'])} files")
        return 0
    try:
        runtime = restore(args.run_dir, args.target, project_root=project_root)
    except RestoreError as exc:
        print(f"restore failed: {exc}", file=sys.stderr)
        return 1
    print(f"restored {len(runtime['restored'])} frozen files into {args.target}")
    if args.sync:
        try:
            runtime = sync_runtime(args.target, project_root=project_root, run_dir=args.run_dir)
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"sync failed: {exc}", file=sys.stderr)
            return 2
        print(f"package_differences: {json.dumps(runtime['package_differences'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
