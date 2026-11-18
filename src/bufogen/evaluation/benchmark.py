"""Frozen paired generation and pixel scoring with durable local checkpoints."""

import asyncio
import fcntl
import hashlib
import html
import json
import os
import platform
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from types import CodeType
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import yaml

from bufogen.agents.agent import BufoAgent, instructions
from bufogen.agents.prompting import prompt_version
from bufogen.evaluation import calibration
from bufogen.evaluation.references import (
    _relative_to_root,
    attach_references,
    load_reference_manifest,
)
from bufogen.evaluation.scorers.astra_judge import (
    CRAFT,
    RUBRIC_VERSION,
    RUBRICS,
    VS_REFERENCE,
    AstraReferenceJudge,
    config_key,
    judge_failure,
)
from bufogen.evaluation.scorers.efficiency import SERVING_PATHS, EfficiencyScorer
from bufogen.evaluation.scorers.scene_quality import (
    SceneQuality,
    failure_result,
    validate_requirements,
)
from bufogen.evaluation.statistics import astra_pass, paired_family_interval
from bufogen.evaluation.weave_replay import (
    ReplayArm,
    shared_dataset_name,
    shared_dataset_rows,
    split_of,
)
from bufogen.generation import BufoGeneration
from bufogen.models import frozen_model_presentations, generation_model_presentation
from bufogen.rendering.renderer import Renderer
from bufogen.runtime import resolve_inference_policy
from bufogen.settings import ROOT

SCHEMA_VERSION = 1
EFFICIENCY_CONFIGURATION = {"scorer": "efficiency-v1"}
ACCOUNTED_STATUSES = {"ok", "judge_error", "missing_reference", "reference_mismatch"}
# Splits reserved for the single confirmation run; judging them needs confirmation=True.
# review_packet.CONFIRMATION_SPLITS is the same set; neither module imports the other.
CONFIRMATION_SPLITS = frozenset({"heldout", "test"})
# Files whose bytes identify the Astra scoring code in Weave metadata. The generation-time
# files take their hash from the run's frozen source_hashes, the judge from the current tree.
GENERATION_CODE_FILES = frozenset({"agent.py", "svg.py", "render.py"})
ASTRA_CODE_FILES = ("agent.py", "scorers/astra_judge.py", "svg.py", "render.py")
# Stable metadata labels remain readable for runs from either source layout.
ASTRA_CODE_PATHS = {
    "agent.py": "agents/agent.py",
    "scorers/astra_judge.py": "evaluation/scorers/astra_judge.py",
    "svg.py": "rendering/svg.py",
    "render.py": "rendering/renderer.py",
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _validate(prompts):
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Benchmark must contain a nonempty list of prompts")
    ids, families = set(), {}
    for row in prompts:
        if not isinstance(row, dict):
            raise ValueError("Each prompt must be an object")
        for key in ("id", "family", "split", "prompt"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Invalid prompt {key}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate prompt id: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in {"train", "development", "validation", "heldout", "test"}:
            raise ValueError("Unknown prompt split")
        if families.setdefault(row["family"], row["split"]) != row["split"]:
            raise ValueError(f"Prompt family crosses splits: {row['family']}")
        validate_requirements(row.get("requirements"))
        if not isinstance(row.get("turns", []), list) or any(
            not isinstance(t, str) or not t.strip() for t in row.get("turns", [])
        ):
            raise ValueError("turns must be prior user requests as nonempty strings")
    return prompts


AUGMENT_PROVENANCE_PREFIX = "aug-train-"
AUGMENT_PROVENANCE = "stage-b-augment-2026-10"
_AUGMENT_ID = re.compile(r"aug-train-(?P<family>.+)-\d\d")


def _validate_augmented(prompts):
    """Augmented prompts must be train prompts traced to a train base prompt of their family.

    A row is augmented if it carries provenance or augmented_from OR has an aug-train- id; the
    id, family and provenance tag must then all agree, so no marker can be omitted to bypass the
    checks. Family disjointness across splits is enforced by _validate.
    """
    by_id = {p["id"]: p for p in prompts}
    for row in prompts:
        marked = "provenance" in row or "augmented_from" in row
        if not (marked or row["id"].startswith(AUGMENT_PROVENANCE_PREFIX)):
            continue
        match = _AUGMENT_ID.fullmatch(row["id"])
        if match is None or match["family"] != row["family"]:
            raise ValueError(
                f"augmented prompt id must be aug-train-<family>-NN with its own family: {row['id']}"
            )
        if row["split"] != "train":
            raise ValueError(f"augmented prompt must be split train: {row['id']}")
        for key in ("provenance", "augmented_from"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"augmented prompt {row['id']} needs {key}")
        if row["provenance"] != AUGMENT_PROVENANCE:
            raise ValueError(f"augmented prompt {row['id']} has unknown provenance")
        base = by_id.get(row["augmented_from"])
        if (
            base is None
            or base["split"] != "train"
            or base["id"].startswith(AUGMENT_PROVENANCE_PREFIX)
            or "provenance" in base
            or "augmented_from" in base
            or base["family"] != row["family"]
        ):
            raise ValueError(
                f"augmented_from of {row['id']} must be an existing train prompt of its family"
            )
    return prompts


def load_benchmark(path: Path) -> list[dict]:
    return _validate_augmented(_validate(yaml.safe_load(Path(path).read_text())))


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def _append(path, value):
    with path.open("a") as file:
        file.write(json.dumps(value, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _read_jsonl(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        # Only an incomplete trailing record may be discarded after a crash.
        data = data[: data.rfind(b"\n") + 1]
        with path.open("wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
    return [json.loads(line) for line in data.splitlines() if line.strip()]


@contextmanager
def _locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".benchmark.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Benchmark directory is locked by another running process") from exc
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def _source_hashes():
    paths = [
        ROOT / "src/bufogen" / name
        for name in (
            "agents/agent.py",
            "models.py",
            "runtime.py",
            "agents/prompting.py",
            "rendering/renderer.py",
            "rendering/svg.py",
            "rendering/composition.py",
            "agents/planning.py",
            "integrations/sandbox.py",
            "agents/web_context.py",
            "agents/reply_model.py",
            "curation/trajectories.py",
            "evaluation/benchmark.py",
            "evaluation/scorers/scene_quality.py",
            "evaluation/scorers/vlm_judge.py",
        )
    ]
    paths += [
        ROOT / "config/brand.yaml",
        ROOT / "scripts/render_worker.py",
        *sorted((ROOT / "assets/bufo_style").glob("*")),
        ROOT / "assets/fonts/DejaVuSans.ttf",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file()
    }


def _code_hash(function):
    # Hash stable public bytecode fields: marshal output can change its internal
    # reference/interning flags after execution even when code has not changed.
    def normalize(value):
        if isinstance(value, CodeType):
            return {
                "bytecode": value.co_code.hex(),
                "names": value.co_names,
                "variables": value.co_varnames,
                "constants": [normalize(c) for c in value.co_consts],
                "argcount": value.co_argcount,
                "kwonly": value.co_kwonlyargcount,
                "freevars": value.co_freevars,
                "flags": value.co_flags,
            }
        if isinstance(value, (tuple, frozenset)):
            return [
                normalize(c)
                for c in (sorted(value, key=repr) if isinstance(value, frozenset) else value)
            ]
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    return digest(normalize(function.__code__))


def _configuration(
    prompts, backends, *, temperature, max_tokens, max_turns, concurrency, disable_thinking=False
):
    arms = []
    for backend in backends:
        backend.require_configured()
        if not backend.supports_tools:
            raise ValueError("All benchmark arms require render tool support")
        runtime_policy = resolve_inference_policy(
            backend,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
            use_environment=False,
        )
        arms.append(
            asdict(backend)
            | {
                "sampling": {
                    "temperature": None if backend.provider == "openai" else temperature,
                    "temperature_reason": "provider default; parameter omitted"
                    if backend.provider == "openai"
                    else "explicit",
                    "max_tokens": runtime_policy.max_tokens,
                    "tool_choice": "required",
                },
                "runtime_policy": runtime_policy.to_dict(),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "prompts": prompts,
        "prompt_hash": digest(prompts),
        "prompt_version": prompt_version(),
        "arms": arms,
        "source_hashes": _source_hashes(),
        "runtime": {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("openai-agents", "weave", "cwsandbox")},
            "instructions_hash": digest(instructions()),
            "loaded_code_hashes": {
                name: _code_hash(function)
                for name, function in {
                    "instructions": instructions,
                    "agent_generate": BufoAgent.generate,
                    "renderer_close": Renderer.close,
                }.items()
            },
        },
        "policy": {
            "web_search_enabled": False,
            "precomputed_context": None,
            "repetitions": 1,
            "max_turns_per_attempt": max_turns,
            "disable_thinking": disable_thinking,
            "attempts": 2,
            "max_tokens": max_tokens,
            "requested_temperature": temperature,
            "renderer_image": os.getenv("BUFO_RENDER_IMAGE")
            or "python:3.12-slim + resvg-py==0.5.0",
            "concurrency": concurrency,
        },
        "scorer": SceneQuality().configuration(),
    }


def _key(row):
    return (row.prompt_id, row.backend)


def comparison_fingerprint(configuration: dict) -> str:
    """Match paired runs across model arms while rejecting changed experimental conditions."""
    common = {key: value for key, value in configuration.items() if key != "arms"}
    common["policy"] = {
        key: value for key, value in configuration["policy"].items() if key != "concurrency"
    }
    # Execution plumbing cannot change scene inputs; generation/scorer source stays frozen.
    common["source_hashes"] = {
        key: value
        for key, value in configuration["source_hashes"].items()
        if key not in {"src/bufogen/benchmark.py", "src/bufogen/evaluation/benchmark.py"}
    }
    return digest(common)


def _validate_rows(rows, settings):
    config = settings["configuration"]
    prompts = {p["id"]: p for p in config["prompts"]}
    arms = {b["name"]: b for b in config["arms"]}
    for row in rows:
        metadata = row.benchmark_metadata or {}
        prompt = prompts.get(row.prompt_id)
        arm = arms.get(row.backend)
        if (
            not prompt
            or not arm
            or row.prompt != prompt["prompt"]
            or row.model != arm["model"]
            or metadata.get("configuration_hash") != settings["configuration_hash"]
            or metadata.get("run_id") != settings["run_id"]
            or any(metadata.get(key) != prompt[key] for key in ("requirements", "family", "split"))
            or metadata.get("turns", []) != prompt.get("turns", [])
        ):
            raise ValueError("Manifest row does not match frozen configuration")
        if row.png_path and metadata.get("png_sha256"):
            path = Path(row.png_path)
            if (
                not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["png_sha256"]
            ):
                raise ValueError("Manifest image changed since generation")


def _metadata(prompt, run_id, config):
    return {
        "run_id": run_id,
        "configuration_hash": digest(config),
        "family": prompt["family"],
        "split": prompt["split"],
        "requirements": prompt["requirements"],
        "turns": prompt.get("turns", []),
        "status": "completed",
    }


def _failure(prompt, backend, metadata, error, *, partial=None, calls=()):
    row = partial or BufoGeneration(
        prompt_id=prompt["id"],
        prompt=prompt["prompt"],
        backend=backend.name,
        model=backend.model,
        group_id=f"{metadata['run_id']}:{backend.name}:{prompt['id']}",
        svg="",
        valid=False,
        latency_s=0,
    )
    # A valid last render remains valid, but does not make the generation successful.
    row.benchmark_metadata = metadata | {"status": "generation_error"}
    row.error = error
    if calls:
        row.model_calls = [c.to_dict() for c in calls]
        row.truncated_calls = sum(c.finish_reason == "length" for c in calls)
    return row


async def run_benchmark(
    prompts,
    backends,
    output_dir: Path,
    *,
    limit=None,
    concurrency=2,
    temperature=0.7,
    max_tokens=None,
    max_turns=100,
    disable_thinking: bool = False,
    prompt_ids: list[str] | None = None,
) -> list[BufoGeneration]:
    """Run each selected prompt against every arm; resume only an identical frozen run.

    Failure rows are terminal. In-flight requests left by a crash become explicit interrupted
    failures on resume, never silently repeated. Use a new directory for a deliberate rerun.
    """
    import math

    _validate(prompts)
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError("concurrency must be an integer in [1, 32]")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if (
        (max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1))
        or type(max_turns) is not int
        or max_turns < 1
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("Invalid generation settings")
    backends = list(backends.values()) if isinstance(backends, dict) else list(backends)
    if not backends or len({b.name for b in backends}) != len(backends):
        raise ValueError("Provide uniquely named backend arms")
    if prompt_ids is not None:
        known = {p["id"] for p in prompts}
        if (
            not prompt_ids
            or len(set(prompt_ids)) != len(prompt_ids)
            or not set(prompt_ids) <= known
        ):
            raise ValueError("prompt_ids must be distinct known prompt ids")
        prompts = [p for p in prompts if p["id"] in set(prompt_ids)]
    selected = json.loads(json.dumps(prompts[:limit]))  # Freeze caller-owned mutable rows.
    directory = Path(output_dir).resolve()
    with _locked(directory):
        config = _configuration(
            selected,
            backends,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
            concurrency=concurrency,
            disable_thinking=disable_thinking,
        )
        settings_path = directory / "benchmark.json"
        if settings_path.exists():
            settings = json.loads(settings_path.read_text())
            if settings["configuration"] != config:
                raise ValueError(
                    "Benchmark resume configuration mismatch; use a new output directory"
                )
        else:
            if (directory / "manifest.jsonl").exists():
                raise ValueError("Cannot adopt an existing manifest without frozen configuration")
            settings = {
                "run_id": uuid4().hex,
                "configuration_hash": digest(config),
                "configuration": config,
            }
            settings["comparison_fingerprint"] = comparison_fingerprint(config)
            _write_json(settings_path, settings)
            # Retain the exact source and reference bytes described by this run.
            for relative in config["source_hashes"]:
                destination = directory / "frozen" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / relative).read_bytes())
        results_path = directory / "manifest.jsonl"
        rows = [BufoGeneration(**r) for r in _read_jsonl(results_path)]
        expected = {(p["id"], b.name) for p in selected for b in backends}
        if len({_key(r) for r in rows}) != len(rows) or any(_key(r) not in expected for r in rows):
            raise ValueError("Manifest contains duplicate or unexpected paired rows")
        if any(
            (r.benchmark_metadata or {}).get("configuration_hash") != digest(config) for r in rows
        ):
            raise ValueError("Manifest configuration mismatch")
        _validate_rows(rows, settings)
        done = {_key(r) for r in rows}
        starts_path = directory / "started.jsonl"
        starts = {(r["prompt_id"], r["backend"]): r for r in _read_jsonl(starts_path)}
        renderer = Renderer()
        semaphore = asyncio.Semaphore(concurrency)

        async def generate(prompt, backend):
            key = (prompt["id"], backend.name)
            if key in done:
                return
            async with semaphore:
                metadata = _metadata(prompt, settings["run_id"], config)
                item_dir = directory / "images" / digest(key)[:24]
                if key in starts:
                    partial_path = item_dir / "partial.json"
                    partial = None
                    if partial_path.exists():
                        try:
                            partial = BufoGeneration(**json.loads(partial_path.read_text()))
                        except (ValueError, TypeError):
                            # Preserve the damaged file as evidence, then record a terminal failure.
                            partial_path.replace(item_dir / "partial.damaged.json")
                    row = _failure(
                        prompt,
                        backend,
                        metadata,
                        "Interrupted generation; not automatically repeated",
                        partial=partial,
                    )
                else:
                    _append(starts_path, {"prompt_id": prompt["id"], "backend": backend.name})
                    runtime_policy = next(
                        a["runtime_policy"] for a in config["arms"] if a["name"] == backend.name
                    )
                    agent = BufoAgent(
                        backend,
                        renderer,
                        item_dir,
                        temperature=temperature,
                        max_tokens=runtime_policy["max_tokens"],
                        max_turns=max_turns,
                        disable_thinking=runtime_policy["disable_thinking"],
                        inference_timeout_s=runtime_policy["timeout_s"],
                        web_search_enabled=False,
                        benchmark_metadata=metadata,
                    )
                    try:
                        history = []
                        for index, prior in enumerate(prompt.get("turns", [])):
                            previous = await agent.generate(
                                prior,
                                prompt_id=f"{prompt['id']}:turn{index}",
                                group_id=f"{settings['run_id']}:{backend.name}:{prompt['id']}",
                                theme=prompt["family"],
                                history=history,
                            )
                            history.append(previous.to_dict())
                            if not previous.valid or previous.error:
                                raise RuntimeError("Prior conversational turn failed")
                        # A failure before the final render must not masquerade as its prior image.
                        agent.partial_generation = None
                        partial_checkpoint = item_dir / "partial.json"
                        if partial_checkpoint.exists():
                            partial_checkpoint.replace(item_dir / "prior-turn.json")
                        row = await agent.generate(
                            prompt["prompt"],
                            prompt_id=prompt["id"],
                            group_id=f"{settings['run_id']}:{backend.name}:{prompt['id']}",
                            theme=prompt["family"],
                            history=history or None,
                        )
                        row.benchmark_metadata = metadata
                    except Exception as exc:
                        row = _failure(
                            prompt,
                            backend,
                            metadata,
                            f"{type(exc).__name__}: generation failed",
                            partial=getattr(agent, "partial_generation", None),
                            calls=getattr(agent, "last_calls", ()),
                        )
                        for field, value in getattr(agent, "native_root", {}).items():
                            setattr(row, field, value)
                        row.execution_metadata = getattr(agent, "execution_metadata", {})
                # Failed prior turns are evidence, not final-prompt completions.
                if row.prompt_id != prompt["id"]:
                    metadata = row.benchmark_metadata | {"failed_turn_id": row.prompt_id}
                    row.prompt_id, row.prompt = prompt["id"], prompt["prompt"]
                    row.benchmark_metadata = metadata
                if row.png_path:
                    row.benchmark_metadata["png_sha256"] = hashlib.sha256(
                        Path(row.png_path).read_bytes()
                    ).hexdigest()
                _append(results_path, row.to_dict())
                rows.append(row)

        try:
            outcomes = await asyncio.gather(
                *(generate(p, b) for p in selected for b in backends), return_exceptions=True
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
        finally:
            await renderer.close()
        ordered = {_key(row): row for row in rows}
        return [ordered[p["id"], b.name] for p in selected for b in backends]


def _summary(rows, settings):
    arms = {}
    for arm in settings["configuration"]["arms"]:
        subset = [r for r in rows if r.backend == arm["name"]]
        results = [r.scores["scene_quality"] for r in subset]
        scored = [r for r in results if r["status"] == "ok"]
        successful = sum(
            r.benchmark_metadata["status"] == "completed"
            and r.valid
            and r.scores["scene_quality"].get("quality_failure") is False
            for r in subset
        )
        arms[arm["name"]] = {
            "total": len(subset),
            "scored": len(scored),
            "judge_errors": sum(r["status"] == "judge_error" for r in results),
            "missing_images": sum(r["status"] == "missing_image" for r in results),
            "prior_turn_failures": sum(r["status"] == "prior_turn_failure" for r in results),
            "generation_errors": sum(r.benchmark_metadata["status"] != "completed" for r in subset),
            "invalid_renders": sum(not r.valid for r in subset),
            "provisional_passes": successful,
            "provisional_pass_rate_all_attempts": successful / len(subset) if subset else None,
            "metric_means_scored_only": {
                key: sum(r[key] for r in scored) / len(scored) if scored else None
                for key in (
                    "similarity",
                    "anatomy",
                    "composition",
                    "readability",
                    "constraint_coverage",
                )
            },
        }
    return {
        "run_id": settings["run_id"],
        "configuration_hash": settings["configuration_hash"],
        "calibration_status": "uncalibrated",
        "comparison_fingerprint": settings["comparison_fingerprint"],
        "selected_prompts": len(settings["configuration"]["prompts"]),
        "splits": sorted({p["split"] for p in settings["configuration"]["prompts"]}),
        "arms": arms,
    }


def _comparison(rows, directory, target=None):
    target = target or directory
    cells = []
    settings_path = directory / "benchmark.json"
    configs = (
        frozen_model_presentations(json.loads(settings_path.read_text())["configuration"]["arms"])
        if settings_path.exists()
        else {}
    )
    for row in sorted(rows, key=_key):
        presentation = generation_model_presentation(row, config=configs.get(row.backend))
        path = Path(row.png_path) if row.png_path else None
        link = html.escape(os.path.relpath(path, target), quote=True) if path else ""
        cells.append(
            f"<article><h2>{html.escape(row.prompt_id)} / {html.escape(presentation['model_label'])}</h2>"
            f"<p>{html.escape(presentation['model_version'] or '')}</p>"
            f"<p>{html.escape(row.prompt)}</p>"
            + (
                f'<a href="{link}"><img src="{link}" width="256"></a>'
                if path
                else "<p>No image</p>"
            )
            + "<pre>"
            + html.escape(
                json.dumps(
                    {
                        "benchmark": row.benchmark_metadata,
                        "generation_error": row.error,
                        "scores": row.scores,
                    },
                    indent=2,
                )
            )
            + "</pre></article>"
        )
    (target / "comparison.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        "<title>Bufo paired benchmark</title><style>body{font:16px sans-serif}main{display:flex;flex-wrap:wrap}"
        "article{width:460px;padding:16px;border:1px solid #aaa}pre{white-space:pre-wrap}</style>"
        '<h1>Paired image review — uncalibrated scores</h1><a href="summary.json">Raw summary</a><main>'
        + "".join(cells)
        + "</main>"
    )


def _check_frozen_scoring_sources(config):
    """Compare exactly the frozen scorer/asset paths; unlisted new files are irrelevant."""
    for path, frozen in config["source_hashes"].items():
        if not (
            path.startswith("assets/")
            or path.startswith(("src/bufogen/scorers/", "src/bufogen/evaluation/scorers/"))
        ):
            continue
        file = ROOT / path
        current = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
        if current != frozen:
            raise ValueError(
                "Scoring configuration mismatch: reference assets or scorer source changed"
            )


def _manifest_rows(directory, settings):
    config = settings["configuration"]
    rows = [BufoGeneration(**r) for r in _read_jsonl(directory / "manifest.jsonl")]
    expected = {(p["id"], b["name"]) for p in config["prompts"] for b in config["arms"]}
    if len(rows) != len(expected) or {_key(r) for r in rows} != expected:
        raise ValueError("Generation manifest incomplete or duplicated; resume generation first")
    if any(
        (r.benchmark_metadata or {}).get("configuration_hash") != settings["configuration_hash"]
        for r in rows
    ):
        raise ValueError("Generation manifest configuration mismatch")
    _validate_rows(rows, settings)
    return rows


async def score_benchmark(
    output_dir: Path,
    *,
    concurrency=2,
    publish=True,
    judge: str = "scene",
    references: Path | None = None,
    self_reference: bool = False,
    baseline_arm: str | None = None,
    limit_prompt_ids: list[str] | None = None,
    confirmation: bool = False,
    rubric_version: str = RUBRIC_VERSION,
    limit_generation_ids: list[str] | None = None,
    judge_samples: int = 1,
) -> list[BufoGeneration]:
    """Checkpoint scores, retaining judge failures; optionally publish native Weave feedback/evals.

    No CLIP dependency. Publish=False runs local scoring only (still a paid VLM call for real images).
    judge="scene": existing scores, including errors, are terminal and replayed without new
    judge requests.
    judge="astra" scores against reference PNGs into <run>/astra/ (or <run>/astra-self/);
    a rubric other than the default v1 writes to <run>/astra-<suffix>/ (e.g. astra-v2/) so
    frozen v1 evidence is never rewritten. limit_generation_ids restricts scoring to those rows.
    Saved verdicts and non-transient failures are replayed; cached ``judge_error`` rows are
    dropped and retried on every run. Heldout/test rows are refused unless confirmation=True,
    and nothing is published when no row has an ``ok`` verdict.
    """
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError("concurrency must be an integer in [1, 32]")
    if judge not in ("scene", "astra"):
        raise ValueError("judge must be 'scene' or 'astra'")
    if judge == "scene" and (
        references is not None
        or self_reference
        or baseline_arm is not None
        or limit_prompt_ids is not None
        or confirmation
        or rubric_version != RUBRIC_VERSION
        or limit_generation_ids is not None
        or judge_samples != 1
    ):
        raise ValueError(
            "references, self_reference, baseline_arm, limit_prompt_ids, limit_generation_ids, "
            "rubric_version, judge_samples and confirmation require judge='astra'"
        )
    if rubric_version not in RUBRICS:
        raise ValueError(f"unknown rubric version {rubric_version!r}")
    directory = Path(output_dir).resolve()
    with _locked(directory):
        settings = json.loads((directory / "benchmark.json").read_text())
        config = settings["configuration"]
        _check_frozen_scoring_sources(config)
        if judge == "astra":
            return await _score_astra(
                directory,
                settings,
                concurrency=concurrency,
                publish=publish,
                references=references,
                self_reference=self_reference,
                baseline_arm=baseline_arm,
                limit_prompt_ids=limit_prompt_ids,
                confirmation=confirmation,
                rubric_version=rubric_version,
                limit_generation_ids=limit_generation_ids,
                judge_samples=judge_samples,
            )
        scorer = SceneQuality()
        if config["scorer"] != scorer.configuration():
            raise ValueError("Scoring configuration mismatch: judge settings changed")
        rows = _manifest_rows(directory, settings)
        scored_path = directory / "scored.jsonl"
        saved = _read_jsonl(scored_path)
        if len({r["id"] for r in saved}) != len(saved) or any(
            r["id"] not in {r.id for r in rows} for r in saved
        ):
            raise ValueError("Unexpected or duplicate scored rows")
        original = {r.id: r.to_dict() for r in rows}
        for saved_row in saved:
            expected_row = original[saved_row["id"]]
            if any(
                saved_row.get(key) != value
                for key, value in expected_row.items()
                if key != "scores"
            ):
                raise ValueError("Saved scored row differs from generation manifest")
            saved_result = saved_row.get("scores", {}).get("scene_quality", {})
            if (
                saved_result.get("scorer_configuration") != scorer.configuration()
                or saved_result.get("requirements")
                != saved_row["benchmark_metadata"]["requirements"]
                or saved_result.get("generation_id") != saved_row["id"]
            ):
                raise ValueError("Saved scored result configuration mismatch")
        existing = {r["id"]: BufoGeneration(**r) for r in saved}
        semaphore = asyncio.Semaphore(concurrency)

        async def score(row):
            if row.id in existing:
                return existing[row.id]
            async with semaphore:
                try:
                    result = await scorer.score(row.to_dict(), rescore=True)
                except Exception as exc:
                    result = failure_result(
                        "judge_error", error=f"{type(exc).__name__}: scorer failed"
                    )
                    result |= {
                        "scorer_configuration": scorer.configuration(),
                        "generation_id": row.id,
                        "requirements": row.benchmark_metadata["requirements"],
                    }
                row.scores["scene_quality"] = result
                _append(scored_path, row.to_dict())
                return row

        rows = await asyncio.gather(*(score(r) for r in rows))
        _write_json(directory / "summary.json", _summary(rows, settings))
        _comparison(rows, directory)
        if publish:
            await _publish(rows, directory, settings, scorer)
        return rows


def _validate_eval_coverage(table_rows, expected_rows, score_keys=("scene_quality",)):
    """Require one nonempty score per scorer, bound to the row and equal to the saved score.

    ``score_keys`` is either a one-key tuple (the single scene scorer, matched by value) or a
    mapping {Weave scorer name: saved score key}, matched under each scorer's own name.
    """
    if len(table_rows) != len(expected_rows):
        raise RuntimeError("Weave evaluation omitted benchmark rows")
    if isinstance(score_keys, dict):
        for result, row in zip(table_rows, expected_rows, strict=True):
            scores = result.get("scores", {})
            if set(scores) != set(score_keys) or any(
                not scores[name]
                or scores[name].get("generation_id") != row.id
                or scores[name] != row.scores[key]
                for name, key in score_keys.items()
            ):
                raise RuntimeError("Weave evaluation omitted or changed a saved score")
        return
    if len(score_keys) != 1:
        raise ValueError("Pass a {scorer name: score key} mapping for multiple scorers")
    (key,) = score_keys
    for result, row in zip(table_rows, expected_rows, strict=True):
        scores = list(result.get("scores", {}).values())
        if (
            len(scores) != 1
            or not scores[0]
            or scores[0].get("generation_id") != row.id
            or scores[0] != row.scores[key]
        ):
            raise RuntimeError("Weave evaluation omitted or changed a saved scene score")


def _scorer_name(scorer) -> str:
    # Weave keys evaluation scores by Scorer.name, falling back to the class name.
    return getattr(scorer, "name", None) or type(scorer).__name__


def _arm_presentation(arms, arm):
    config = frozen_model_presentations(arms)[arm["name"]]
    presentation = generation_model_presentation(
        {"backend": arm["name"], "model": arm["model"]}, config=config
    )
    return config, presentation


async def _publish_source_feedback(client, rows, scorers, state, path, *, include_classic=True):
    """Attach existing evidence to its original source, with retryable per-target receipts.

    Native Conversation spans have a supported custom-feedback API, not Call.apply_scorer.
    Custom feedback preserves the exact saved verdict but does not populate Agent monitor
    score columns. Do not invent the monitor trigger/call refs required for that API.
    """
    for row in rows:
        for key, scorer in scorers.items():
            score = row.scores.get(key)
            if not isinstance(score, dict) or score.get("generation_id") != row.id:
                raise ValueError(f"Saved {key} score does not match generation {row.id}")
            score_digest = digest(score)
            receipt = (
                state.setdefault("source_feedback", {}).setdefault(row.id, {}).setdefault(key, {})
            )
            targets = ["native", "classic"] if include_classic else ["native"]
            for target in targets:
                source_id = row.native_span_id if target == "native" else row.weave_call_id
                previous = receipt.get(target, {})
                if (
                    previous.get("status") == "published"
                    and previous.get("source_id") == source_id
                    and previous.get("scores_sha256") == score_digest
                ):
                    continue
                result = {"source_id": source_id, "scores_sha256": score_digest}
                try:
                    if not source_id:
                        result["status"] = (
                            "missing_source_root" if target == "native" else "missing_source_call"
                        )
                    elif target == "native":
                        get_feedback = getattr(client, "get_agent_span_feedback", None)
                        if not callable(get_feedback):
                            result["status"] = "unsupported_sdk"
                        else:
                            feedback = get_feedback(source_id)
                            feedback_type = f"bufo.saved_{key}"
                            payload = {
                                "generation_id": row.id,
                                "native_trace_id": row.native_trace_id,
                                "native_span_id": row.native_span_id,
                                "evidence_sha256": score_digest,
                                "score": score,
                            }
                            found = next(
                                (
                                    f
                                    for f in feedback
                                    if f.feedback_type == feedback_type and f.payload == payload
                                ),
                                None,
                            )
                            feedback_id = (
                                found.id
                                if found
                                else feedback.add(
                                    feedback_type,
                                    payload=payload,
                                    creator="Bufo saved scorer evidence",
                                )
                            )
                            client.flush()
                            result.update(
                                status="published",
                                feedback_id=feedback_id,
                                score_columns="unsupported_custom_feedback",
                            )
                    else:
                        get_call = getattr(client, "get_call", None)
                        if not callable(get_call):
                            result["status"] = "unsupported_sdk"
                        else:
                            call = get_call(source_id)
                            found = next(
                                (
                                    f
                                    for f in call.feedback
                                    if f.feedback_type.startswith("wandb.runnable.")
                                    and f.payload.get("output") == score
                                ),
                                None,
                            )
                            if not found:
                                from bufogen.integrations.observability import SavedGenerationScore

                                # The source output predates Astra reference attachment. A
                                # clearly named deterministic replay validates row identity
                                # without silently invoking a judge on a cache miss.
                                replay = SavedGenerationScore(name="Saved" + _scorer_name(scorer))
                                await call.apply_scorer(replay, {"saved_score": score})
                            client.flush()
                            result.update(status="published")
                except Exception as exc:
                    result.update(status="error", error_type=type(exc).__name__)
                receipt[target] = result
                _write_json(path, state)


async def _publish_arm_evaluation(
    state,
    path,
    name,
    *,
    class_name,
    dataset_name,
    evaluation_name,
    dataset_rows,
    subset,
    scorers,
    metadata,
    score_keys,
    record=None,
    model=None,
):
    """Publish and run one arm's coverage-checked Weave evaluation, checkpointing its state.

    ``model`` defaults to the passthrough ``visual_generation`` op over rows that carry their
    generation; the astra path passes a ``ReplayArm`` so every arm shares one prompt dataset.
    """
    import weave

    from bufogen.integrations.observability import visual_generation

    dataset = weave.Dataset(name=dataset_name, rows=dataset_rows)
    base = weave.Evaluation

    async def get_eval_results(self, model):
        table = await base.get_eval_results(self, model)
        _validate_eval_coverage(list(table.rows), subset, score_keys)
        return table

    evaluation_type = type(
        class_name, (base,), {"__module__": __name__, "get_eval_results": get_eval_results}
    )
    evaluation = evaluation_type(
        name=evaluation_name,
        evaluation_name=evaluation_name,
        dataset=dataset,
        scorers=list(scorers),
        metadata=metadata,
    )
    try:
        ref = weave.publish(evaluation).uri()
        result = await evaluation.evaluate(model if model is not None else visual_generation)
        state["evaluations"][name] = {
            "ref": ref,
            "summary": result,
            "scored_rows": len(subset),
        } | (record or {})
        state.setdefault("errors", {}).pop("evaluation:" + name, None)
        _write_json(path, state)
    except Exception as exc:
        state.setdefault("errors", {})["evaluation:" + name] = type(exc).__name__
        _write_json(path, state)
        raise


async def _publish(rows, directory, settings, scorer):
    from bufogen.integrations.observability import record_generation
    from bufogen.integrations.telemetry import init_weave

    client = init_weave()
    path = directory / "weave-publication.json"
    state = json.loads(path.read_text()) if path.exists() else {"feedback": {}, "evaluations": {}}
    for row in rows:
        if row.id in state["feedback"]:
            continue
        try:
            if not row.weave_call_id:
                record_generation(row)
            client.flush()
            if not row.weave_call_id:
                raise RuntimeError("No visual generation call created")
            call = client.get_call(row.weave_call_id)
            await call.apply_scorer(scorer, {"saved_score": row.scores["scene_quality"]})
            client.flush()
            state["feedback"][row.id] = {"call_id": row.weave_call_id}
            _write_json(path, state)
        except Exception as exc:
            state.setdefault("errors", {})[row.id] = type(exc).__name__
            _write_json(path, state)
            raise
    await _publish_source_feedback(
        client, rows, {"scene_quality": scorer}, state, path, include_classic=False
    )
    for arm in settings["configuration"]["arms"]:
        name = arm["name"]
        if name in state["evaluations"]:
            continue
        subset = [r for r in rows if r.backend == name]
        _, presentation = _arm_presentation(settings["configuration"]["arms"], arm)
        display_name = presentation["model_label"]
        await _publish_arm_evaluation(
            state,
            path,
            name,
            class_name="CompleteSceneEvaluation",
            dataset_name=f"bufo-scene-{settings['run_id']}-{display_name}",
            evaluation_name=f"bufo-scene-{display_name}",
            dataset_rows=[{"prompt": r.prompt, "generation": r.to_dict()} for r in subset],
            subset=subset,
            scorers=[scorer],
            metadata={"configuration_hash": settings["configuration_hash"], **presentation},
            score_keys=("scene_quality",),
        )
    client.flush()


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _quantile(values, q):
    values = list(values)
    return float(np.quantile(values, q)) if values else None


def _astra_summary(rows, settings, *, baseline_arm, calibration_status="uncalibrated"):
    """Every row stays in its arm's denominator; non-ok judge statuses are counted, not dropped."""
    arms = {}
    for arm in settings["configuration"]["arms"]:
        subset = [r for r in rows if r.backend == arm["name"]]
        results = [r.scores["astra_judge"] for r in subset]
        efficiency = [r.scores["efficiency"] for r in subset]
        scored = [r for r in results if r["status"] == "ok"]
        total = len(subset)
        passes = sum(astra_pass(r) for r in subset)
        arms[arm["name"]] = {
            "total": total,
            "scored": len(scored),
            "judge_errors": sum(r["status"] == "judge_error" for r in results),
            "missing_references": sum(r["status"] == "missing_reference" for r in results),
            "reference_mismatches": sum(r["status"] == "reference_mismatch" for r in results),
            # Any other non-ok status (e.g. missing_image) stays accounted for here.
            "other_failures": sum(r["status"] not in ACCOUNTED_STATUSES for r in results),
            "strict_passes": passes,
            "strict_pass_rate": passes / total if total else None,
            "requirement_coverage_mean": _mean(r["requirement_coverage"] for r in scored),
            "vs_reference": {
                key: sum(r["vs_reference"] == key for r in scored) for key in VS_REFERENCE
            },
            "low_confidence": sum(r["low_confidence"] is True for r in scored),
            "craft_means": {key: _mean(r[key] for r in scored) for key in CRAFT},
            "latency_p50_s": _quantile((e["latency_s"] for e in efficiency), 0.5),
            "latency_p90_s": _quantile((e["latency_s"] for e in efficiency), 0.9),
            "output_tokens_mean": _mean(e["output_tokens"] for e in efficiency),
            "reasoning_tokens_mean": _mean(e["reasoning_tokens"] for e in efficiency),
            "truncated_calls": sum(e["truncated_calls"] for e in efficiency),
            # From the frozen arm, matching the Weave evaluation metadata.
            "serving_path": SERVING_PATHS.get(arm["provider"], "unknown"),
        }
    paired = {}
    if baseline_arm is not None:
        for name in arms:
            if name != baseline_arm:
                paired[name] = paired_family_interval(rows, name, baseline_arm, pass_fn=astra_pass)
    return {
        "run_id": settings["run_id"],
        "configuration_hash": settings["configuration_hash"],
        "judge": RUBRIC_VERSION,
        "calibration_status": calibration_status,
        "splits": sorted({r.benchmark_metadata["split"] for r in rows}),
        "arms": arms,
        "paired": paired,
    }


async def _score_astra(
    directory,
    settings,
    *,
    concurrency,
    publish,
    references,
    self_reference,
    baseline_arm,
    limit_prompt_ids,
    confirmation,
    rubric_version=RUBRIC_VERSION,
    limit_generation_ids=None,
    judge_samples=1,
):
    config = settings["configuration"]
    if self_reference and references is not None:
        raise ValueError("Pass either references or self_reference, not both")
    if not self_reference and references is None:
        raise ValueError("judge='astra' requires references unless self_reference")
    if baseline_arm is not None and baseline_arm not in {a["name"] for a in config["arms"]}:
        raise ValueError(f"Unknown baseline arm: {baseline_arm}")
    selected = None
    if limit_prompt_ids is not None:
        selected = list(limit_prompt_ids)
        known = {p["id"] for p in config["prompts"]}
        if not selected or len(set(selected)) != len(selected) or set(selected) - known:
            raise ValueError("limit_prompt_ids must be distinct known prompt ids")
    rows = _manifest_rows(directory, settings)
    to_score = rows if selected is None else [r for r in rows if r.prompt_id in set(selected)]
    if limit_generation_ids is not None:
        wanted = list(limit_generation_ids)
        known_ids = {r.id for r in to_score}
        if not wanted or len(set(wanted)) != len(wanted) or set(wanted) - known_ids:
            raise ValueError("limit_generation_ids must be distinct generation ids of this run")
        to_score = [r for r in to_score if r.id in set(wanted)]
    reserved = sorted(
        {
            r.prompt_id
            for r in to_score
            if (r.benchmark_metadata or {}).get("split") in CONFIRMATION_SPLITS
        }
    )
    if reserved and not confirmation:
        raise ValueError(
            f"{len(reserved)} prompts are in a heldout/test split; judging them is reserved "
            "for the single confirmation run (confirmation=True)"
        )
    manifest, reference_record = None, None
    if not self_reference:
        path = Path(references).resolve()
        if path.is_dir():
            path = path / "references.json"
        digest_hex = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = load_reference_manifest(path)
        reference_record = {
            "path": _relative_to_root(path),
            "version": manifest["version"],
            "source_run_id": manifest["source_run_id"],
            "sha256": digest_hex,
        }
    attach_references(rows, manifest, self_reference=self_reference)
    judge = AstraReferenceJudge(rubric_version=rubric_version, samples=judge_samples)
    efficiency = EfficiencyScorer()
    judge_config = judge.configuration()
    record = {
        "judge": judge_config,
        "efficiency": EFFICIENCY_CONFIGURATION,
        "references": reference_record,
        "self_reference": self_reference,
        "confirmation": confirmation,
    }
    # Self-equal checks never share files with real-reference scoring of the same run, and a
    # rubric other than v1 gets its own directory (astra-v2/, astra-self-v2/).
    target = directory / _astra_dir_name(self_reference, rubric_version, judge_samples)
    target.mkdir(exist_ok=True)
    scorer_path = target / "scorer.json"
    if scorer_path.exists():
        saved_record = json.loads(scorer_path.read_text())
        # Records written before the confirmation guard existed were all non-confirmation.
        saved_record.setdefault("confirmation", False)
        if _record_key(saved_record) != _record_key(record):
            raise ValueError("Astra scorer configuration changed; use a new output directory")
    else:
        _write_json(scorer_path, record)

    scored_path = target / "scored.jsonl"
    saved = _read_jsonl(scored_path)
    original = {r.id: r.to_dict() for r in rows}
    if len({r["id"] for r in saved}) != len(saved) or any(r["id"] not in original for r in saved):
        raise ValueError("Unexpected or duplicate scored rows")
    for saved_row in saved:
        expected_row = original[saved_row["id"]]
        if any(saved_row.get(k) != v for k, v in expected_row.items() if k != "scores"):
            raise ValueError("Saved scored row differs from generation manifest")
        scores = saved_row.get("scores", {})
        result, cost = scores.get("astra_judge", {}), scores.get("efficiency", {})
        reference = saved_row["benchmark_metadata"].get("reference") or {}
        if (
            config_key(result.get("scorer_configuration")) != config_key(judge_config)
            or result.get("requirements") != saved_row["benchmark_metadata"]["requirements"]
            or result.get("generation_id") != saved_row["id"]
            or (result.get("reference") or {}).get("png_sha256") != reference.get("png_sha256")
            or cost.get("generation_id") != saved_row["id"]
        ):
            raise ValueError("Saved scored result configuration mismatch")
    # Judge errors are transient by definition; drop them from the cache so a re-run
    # retries them (the file is rewritten so retried rows are not duplicated).
    kept = [r for r in saved if r["scores"]["astra_judge"].get("status") != "judge_error"]
    if len(kept) != len(saved):
        scored_path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    existing = {r["id"]: BufoGeneration(**r) for r in kept}
    rows = to_score
    semaphore = asyncio.Semaphore(concurrency)

    async def score(row):
        if row.id in existing:
            return existing[row.id]
        async with semaphore:
            try:
                result = await judge.score(row.to_dict(), rescore=True)
            except Exception as exc:
                reference = row.benchmark_metadata.get("reference") or {}
                result = judge_failure(
                    "judge_error",
                    error=f"{type(exc).__name__}: scorer failed",
                    rubric_version=rubric_version,
                    calibration_status=judge_config["calibration_status"],
                )
                result |= {
                    "scorer_configuration": judge_config,
                    "requirements": row.benchmark_metadata["requirements"],
                    "judge_model": judge.judge_model,
                    "generation_id": row.id,
                    "reference": {
                        "png_sha256": reference.get("png_sha256"),
                        "generation_id": reference.get("generation_id"),
                        "is_self": reference.get("is_self"),
                    },
                    "raw_judgment": None,
                }
            row.scores["astra_judge"] = result
            row.scores["efficiency"] = efficiency.score(row.to_dict())
            _append(scored_path, row.to_dict())
            return row

    rows = await asyncio.gather(*(score(r) for r in rows))
    _write_json(
        target / "summary.json",
        _astra_summary(
            rows,
            settings,
            baseline_arm=baseline_arm,
            calibration_status=judge_config["calibration_status"],
        ),
    )
    _comparison(rows, directory, target=target)
    statuses = [r.scores["astra_judge"]["status"] for r in rows]
    # No ok verdict at all (all judge errors, or e.g. a wrong --references making every row
    # missing_reference) would publish a plausible-looking 0% evaluation: refuse.
    if "ok" not in statuses:
        counts = ", ".join(f"{n} {s}" for s, n in sorted(Counter(statuses).items()))
        raise RuntimeError(
            f"Astra judge returned no verdicts: {counts}; "
            f"see {target / 'scored.jsonl'} and fix the judge or references before publishing"
        )
    if publish:
        await _publish_astra(rows, target, settings, [judge, efficiency])
    return rows


def _rubric_suffix(rubric_version: str, samples: int = 1) -> str:
    suffix = "" if rubric_version == RUBRIC_VERSION else "-" + rubric_version.rsplit("-", 1)[-1]
    return suffix + (f"-k{samples}" if samples != 1 else "")


def _astra_dir_name(self_reference: bool, rubric_version: str, samples: int = 1) -> str:
    return ("astra-self" if self_reference else "astra") + _rubric_suffix(rubric_version, samples)


def _record_key(record: dict) -> dict:
    return {**record, "judge": config_key(record.get("judge"))}


def _astra_dataset_row(row):
    metadata = row.benchmark_metadata
    generation = row.to_dict()
    for key in ("png_path", "svg_path", "input_image_path"):
        if generation.get(key):
            generation[key] = _relative_to_root(generation[key])
    reference = metadata.get("reference") or {}
    if reference.get("png_path"):
        generation["benchmark_metadata"] = {
            **generation["benchmark_metadata"],
            "reference": {**reference, "png_path": _relative_to_root(reference["png_path"])},
        }
    return {
        "prompt": row.prompt,
        "prompt_id": row.prompt_id,
        "family": metadata["family"],
        "split": metadata["split"],
        "requirements": metadata["requirements"],
        "reference_png_sha256": reference.get("png_sha256"),
        "reference_generation_id": reference.get("generation_id"),
        "reference_is_self": reference.get("is_self"),
        "generation": generation,
    }


# Written by the Stage B sweep: one entry per checkpoint, newest last. Schema:
# {arm_name, artifact, forge_job_id, forge_dataset_id,
#  sft_config: {epochs, batch_size, learning_rate, lora_rank, assistant_turns},
#  adapter_rank_observed}
CHECKPOINTS_SIDECAR = ROOT / "runs/distill/stage-b/checkpoints.json"


SFT_CONFIG_FIELDS = {"epochs", "batch_size", "learning_rate", "lora_rank", "assistant_turns"}


def _checkpoint_metadata(arm):
    """Forge lineage of a `wandb-artifact:` arm from its config or the sidecar; refuse if absent."""
    empty = dict.fromkeys(("forge_job_id", "forge_dataset_id", "sft_config", "checkpoint_version"))
    if not arm["model"].startswith("wandb-artifact:"):
        return empty
    found = {k: arm.get(k) for k in empty}
    if not (found["forge_job_id"] and found["forge_dataset_id"] and found["sft_config"]):
        try:
            entries = json.loads(CHECKPOINTS_SIDECAR.read_text())
        except (OSError, ValueError):
            entries = []
        matches = [
            e
            for e in entries
            if isinstance(e, dict)
            and e.get("arm_name") == arm["name"]
            and e.get("artifact") == arm["model"]
        ]
        if matches:
            found.update({k: matches[-1].get(k) for k in found if matches[-1].get(k)})
    tag = arm["model"].rpartition(":")[2] if arm["model"].count(":") > 1 else None
    version = tag or found["checkpoint_version"]
    found["checkpoint_version"] = version if isinstance(version, str) else None
    problems = [k for k, v in found.items() if not v]
    if found["checkpoint_version"] and not re.fullmatch(r"v\d+", found["checkpoint_version"]):
        problems.append("checkpoint_version (must be an immutable vN tag)")
    if isinstance(found["sft_config"], dict):
        missing = SFT_CONFIG_FIELDS - set(found["sft_config"])
        if missing:
            problems.append(f"sft_config fields {sorted(missing)}")
    elif found["sft_config"]:
        problems.append("sft_config (must be an object)")
    if problems:
        raise ValueError(
            f"Arm {arm['name']} serves a wandb-artifact checkpoint but its checkpoint metadata "
            f"is missing or invalid: {', '.join(problems)}; add a complete entry to "
            f"{CHECKPOINTS_SIDECAR}"
        )
    return found


def checkpoint_artifact_url(model: str) -> str | None:
    """Link pinned Forge LoRA checkpoints using their actual artifact collection type."""
    match = re.fullmatch(r"wandb-artifact:///([^/]+)/([^/]+)/([^/:]+):(v\d+)", model)
    if match is None:
        return None
    entity, project, name, version = (quote(part, safe="") for part in match.groups())
    return f"https://forge.coreweave.com/wandb/{entity}/{project}/artifacts/lora/{name}/{version}"


def _require_checkpoint_metadata(arms):
    for arm in arms:
        _checkpoint_metadata(arm)


def _astra_metadata(arm, subset, settings, record, presentation_config, code):
    """Evaluation metadata from the frozen run and scorer record only, never the environment."""
    config, judge = settings["configuration"], record["judge"]
    policy = config["policy"]
    references = record["references"]
    return {
        "rubric_version": judge["rubric_version"],
        "judge_samples": judge.get("samples", 1),
        "judge_model": judge["judge_model"],
        "judge_prompt_sha256": judge["rubric_hash"],
        "schema_hash": judge["schema_hash"],
        "prompt_version": config["prompt_version"],
        "configuration_hash": settings["configuration_hash"],
        "prompt_set_sha256": config["prompt_hash"],
        "backend": arm["name"],
        "model": arm["model"],
        "base_model": presentation_config.get("base_model") or None,
        "serving_path": SERVING_PATHS.get(arm["provider"], "unknown"),
        "checkpoint_artifact": arm["model"] if arm["model"].startswith("wandb-artifact:") else None,
        "checkpoint_artifact_url": checkpoint_artifact_url(arm["model"]),
        "max_turns": policy["max_turns_per_attempt"],
        "max_tokens": arm.get("runtime_policy", {}).get("max_tokens", policy["max_tokens"]),
        "disable_thinking": arm.get("runtime_policy", {}).get(
            "disable_thinking", policy.get("disable_thinking", False)
        ),
        "retry_loop_enabled": True,
        "web_search_enabled": False,
        "reference_run_id": references["source_run_id"] if references else settings["run_id"],
        "self_reference": record["self_reference"],
        "split": ",".join(sorted({r.benchmark_metadata["split"] for r in subset})),
        "code_sha256": code,
        "confirmation": record.get("confirmation", False),
        # Reported at publish time from the registry, not from the saved scorer.json.
        "calibration_status": calibration.status(judge),
        "calibration_record": calibration.record_path(judge),
        **_checkpoint_metadata(arm),
    }


def _astra_scores_digest(subset) -> str:
    """Digest of what an arm's evaluation shows: sorted (row id, judge status, strict_pass)."""
    rows = sorted(
        [
            r.id,
            r.scores["astra_judge"].get("status"),
            bool(r.scores["astra_judge"].get("strict_pass")),
        ]
        for r in subset
    )
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


async def _publish_astra(rows, directory, settings, scorers):
    from bufogen.integrations.telemetry import init_weave

    client = init_weave()
    path = directory / "weave-publication.json"
    state = json.loads(path.read_text()) if path.exists() else {"evaluations": {}}
    record = json.loads((directory / "scorer.json").read_text())
    arms = settings["configuration"]["arms"]
    # Refuse before any publication if a checkpoint arm lacks its Forge lineage.
    _require_checkpoint_metadata([a for a in arms if any(r.backend == a["name"] for r in rows)])
    judge, efficiency = scorers
    score_keys = {_scorer_name(judge): "astra_judge", _scorer_name(efficiency): "efficiency"}
    await _publish_source_feedback(client, rows, {"astra_judge": judge}, state, path)
    # Generation code comes from the run's frozen hashes (what drew the rows); the judge,
    # which is not part of generation, is hashed from the tree that scored them.
    frozen = settings["configuration"]["source_hashes"]
    code = {
        name: (
            frozen.get(
                f"src/bufogen/{ASTRA_CODE_PATHS[name]}", frozen.get(f"src/bufogen/{name}")
            )
            if name in GENERATION_CODE_FILES
            else hashlib.sha256((ROOT / "src/bufogen" / ASTRA_CODE_PATHS[name]).read_bytes()).hexdigest()
        )
        for name in ASTRA_CODE_FILES
    }
    for arm in arms:
        name = arm["name"]
        subset = [r for r in rows if r.backend == name]
        if not subset:
            continue
        prompt_ids = sorted(r.prompt_id for r in subset)
        scores_digest = _astra_scores_digest(subset)
        publication = {"prompt_ids": prompt_ids, "scores_sha256": scores_digest}
        published = state["evaluations"].get(name)
        if published is not None:
            if (
                published.get("scored_rows") != len(subset)
                or published.get("prompt_ids") != prompt_ids
            ):
                raise ValueError(
                    f"Weave evaluation for arm {name} was published from different rows; "
                    "use a new output directory"
                )
            # Records from before the digest existed carry none; they are left as published.
            if published.get("scores_sha256") in (None, scores_digest):
                continue
            # Same rows, different scores (e.g. retried judge errors): publish a superseding
            # evaluation and keep the old ref on record.
            publication["superseded"] = [*published.get("superseded", []), published["ref"]]
        config, presentation = _arm_presentation(arms, arm)
        display_name = presentation["model_label"] + _rubric_suffix(
            record["judge"]["rubric_version"], record["judge"].get("samples", 1)
        )
        # One prompt-only dataset shared by every arm (same rows, same Weave digest); the arm's
        # saved outputs are replayed by prompt_id so the UI can compare arms row by row.
        ordered = sorted(subset, key=lambda r: r.prompt_id)
        replay = ReplayArm(
            arm=name,
            model_label=presentation["model_label"],
            checkpoint=arm["model"] if arm["model"].startswith("wandb-artifact:///") else None,
            generations={r.prompt_id: _astra_dataset_row(r)["generation"] for r in ordered},
        )
        await _publish_arm_evaluation(
            state,
            path,
            name,
            class_name="CompleteAstraEvaluation",
            dataset_name=shared_dataset_name(split_of(ordered), len(ordered)),
            evaluation_name=f"bufo-astra-{display_name}",
            dataset_rows=shared_dataset_rows(ordered),
            subset=ordered,
            scorers=scorers,
            metadata=_astra_metadata(arm, ordered, settings, record, config, code),
            score_keys=score_keys,
            record=publication,
            model=replay,
        )
    client.flush()
