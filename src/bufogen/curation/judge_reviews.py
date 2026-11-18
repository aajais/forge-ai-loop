"""Astra judge verdicts as training reviews, accepted only from a calibrated judge setting."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from bufogen.evaluation import calibration
from bufogen.evaluation.scorers.astra_judge import AstraReferenceJudge, config_key
from bufogen.generation import BufoGeneration
from bufogen.settings import ROOT

CONFIDENCE_FLOOR = 0.6


def _reviewer_id(registry_key: tuple[str, int]) -> str:
    rubric, samples = registry_key
    return f"gpt-6-astra/{rubric}/k{samples}"


def _accepts(verdict: dict, metadata: dict) -> bool:
    confidence = verdict.get("judge_confidence")
    return (
        verdict.get("status") == "ok"
        and verdict.get("strict_pass") is True
        and isinstance(confidence, int | float)
        and confidence >= CONFIDENCE_FLOOR
        and metadata.get("status") == "completed"
    )


def is_low_confidence(row) -> bool:
    """Judge status ok with a numeric confidence below the acceptance floor (row dict or object)."""
    scores = row.get("scores") if isinstance(row, dict) else getattr(row, "scores", None)
    verdict = (scores or {}).get("astra_judge") or {}
    confidence = verdict.get("judge_confidence")
    return (
        verdict.get("status") == "ok"
        and isinstance(confidence, int | float)
        and confidence < CONFIDENCE_FLOOR
    )


def decision_from_row(
    row: BufoGeneration, *, registry_key: tuple[str, int], source_ref: str, reviewed_at: str
) -> dict:
    """Review dict for one scored row: accept only a confident strict pass of a completed run."""
    verdict = (row.scores or {}).get("astra_judge") or {}
    metadata = row.benchmark_metadata or {}
    configuration = verdict.get("scorer_configuration") or {}
    confidence = verdict.get("judge_confidence")
    accepted = _accepts(verdict, metadata)
    reason = str(verdict.get("reason") or "").strip()[:1000]
    return {
        "schema_version": 1,
        "source_generation_id": row.id,
        "prompt_id": row.prompt_id,
        "decision": "accept" if accepted else "reject",
        "reviewer_kind": "astra_judge",
        "reviewer_id": _reviewer_id(registry_key),
        "reviewed_at": reviewed_at,
        "source_ref": source_ref,
        "evidence": reason or f"Astra judge status {verdict.get('status')!r}, no reason recorded.",
        "partition": metadata.get("split"),
        "prompt_family": metadata.get("family"),
        "judge": {
            "strict_pass": verdict.get("strict_pass"),
            "judge_confidence": confidence,
            "rubric_version": verdict.get("rubric_version", configuration.get("rubric_version")),
            "samples": configuration.get("samples", 1),
            "rubric_hash": configuration.get("rubric_hash"),
            "schema_hash": configuration.get("schema_hash"),
            "vs_reference": verdict.get("vs_reference"),
        },
    }


def write_decisions(scored_dir: Path, out: Path, *, registry_key: tuple[str, int]) -> dict:
    """Write one review per scored row; refuse uncalibrated or differently configured judges."""
    rubric, samples = registry_key
    live = AstraReferenceJudge(rubric_version=rubric, samples=samples).configuration()
    if calibration.status(live) != "calibrated":
        raise ValueError(f"Judge setting {registry_key} is not calibrated; refusing its verdicts")
    expected = config_key(live)
    path = Path(scored_dir).resolve() / "scored.jsonl"
    try:
        source_ref = str(path.relative_to(ROOT))
    except ValueError:
        source_ref = str(path)
    reviewed_at = datetime.now(UTC).isoformat()
    counts = {"accepted": 0, "rejected": 0, "skipped": 0}
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = BufoGeneration(**json.loads(line))
        verdict = (row.scores or {}).get("astra_judge")
        if not verdict:
            counts["skipped"] += 1
            continue
        if config_key(verdict.get("scorer_configuration")) != expected:
            raise ValueError(
                f"Row {row.id} was not scored by judge setting {registry_key}; refusing mixed verdicts"
            )
        review = decision_from_row(
            row, registry_key=registry_key, source_ref=source_ref, reviewed_at=reviewed_at
        )
        if review["partition"] is None or review["prompt_family"] is None:
            counts["skipped"] += 1
            continue
        counts["accepted" if review["decision"] == "accept" else "rejected"] += 1
        lines.append(json.dumps(review))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{line}\n" for line in lines))
    return counts


def _review_key(review: dict) -> tuple[str, int]:
    judge = review.get("judge")
    if isinstance(judge, dict) and judge.get("rubric_version") and judge.get("samples"):
        return judge["rubric_version"], int(judge["samples"])
    match = re.fullmatch(r"gpt-6-astra/(.+)/k(\d+)", str(review.get("reviewer_id")))
    if not match:
        raise ValueError(
            f"astra_judge review {review.get('source_generation_id')} names no judge setting"
        )
    return match.group(1), int(match.group(2))


def verify_decisions(reviews: list[dict], *, root: Path) -> None:
    """Refuse astra_judge reviews that are uncalibrated, mis-configured or disagree with their scored row."""
    cache: dict[Path, dict[str, BufoGeneration]] = {}
    expected_configs: dict[tuple[str, int], dict] = {}
    for review in reviews:
        if review.get("reviewer_kind") != "astra_judge":
            continue
        gid = review.get("source_generation_id")
        key = _review_key(review)
        if key not in expected_configs:
            live = AstraReferenceJudge(rubric_version=key[0], samples=key[1]).configuration()
            if calibration.status(live) != "calibrated":
                raise ValueError(f"Judge setting {key} is not calibrated; refusing review of {gid}")
            expected_configs[key] = config_key(live)
        path = Path(root) / str(review.get("source_ref"))
        if path not in cache:
            try:
                lines = path.read_text().splitlines()
            except OSError as exc:
                raise ValueError(f"Cannot read scored rows {path} for review of {gid}") from exc
            rows = (BufoGeneration(**json.loads(line)) for line in lines if line.strip())
            cache[path] = {row.id: row for row in rows}
        row = cache[path].get(gid)
        verdict = ((row.scores or {}).get("astra_judge") if row else None) or {}
        if not verdict:
            raise ValueError(f"No judge verdict for {gid} in {path}")
        if config_key(verdict.get("scorer_configuration")) != expected_configs[key]:
            raise ValueError(f"Row {gid} was not scored by judge setting {key}")
        expected = "accept" if _accepts(verdict, row.benchmark_metadata or {}) else "reject"
        if review.get("decision") != expected:
            raise ValueError(f"Review of {gid} disagrees with its scored row ({expected})")
