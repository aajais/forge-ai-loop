"""Reviewed scene calibration and train-only environment tasks for serverless RL."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from bufogen.evaluation.scorers.scene_quality import METRICS, scene_result, validate_requirements


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment_tasks(rows: list[dict]) -> list[dict]:
    """Convert benchmark training scenarios, never teacher response messages.

    Conversational edits are deliberately excluded from this single-turn RL v1.
    They stay in evaluation; do not silently discard their prior requests.
    """
    result, seen, splits = [], set(), {}
    for row in rows:
        family, split = row.get("family"), row.get("split")
        if not family or split not in {"train", "validation", "heldout", "test"}:
            raise ValueError("Environment tasks require family and explicit split")
        if family in splits and splits[family] != split:
            raise ValueError("Prompt family crosses dataset splits")
        splits[family] = split
        if split != "train" or row.get("turns"):
            continue
        prompt_id = row.get("prompt_id") or row.get("id")
        if not prompt_id or prompt_id in seen or not isinstance(row.get("prompt"), str):
            raise ValueError("Environment tasks require unique prompt IDs and prompt text")
        if not row["prompt"].strip():
            raise ValueError("Environment prompt must be nonempty")
        validate_requirements(row.get("requirements"))
        if "messages" in row:
            raise ValueError("RL data must be environment tasks, not teacher messages")
        seen.add(prompt_id)
        result.append(
            {
                "prompt_id": prompt_id,
                "prompt": row["prompt"],
                "family": family,
                "split": "train",
                "requirements": row["requirements"],
                **({"reference": row["reference"]} if row.get("reference") else {}),
            }
        )
    if not result:
        raise ValueError("No single-turn training environment tasks with exact requirements")
    return result


def load_environment_tasks(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # A training artifact cannot sneak in heldout examples for the helper to filter away.
    if any(row.get("split") != "train" or row.get("turns") for row in rows):
        raise ValueError("RL artifact must contain only single-turn train tasks")
    return environment_tasks(rows)


def write_environment_tasks(rows: list[dict], output: Path) -> dict:
    tasks = environment_tasks(rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks))
    metadata = {
        "schema_version": 1,
        "kind": "rl_environment_tasks",
        "count": len(tasks),
        "tasks_sha256": digest(tasks),
        "source_sha256": digest(rows),
        "excluded_nontrain": sum(row.get("split") != "train" for row in rows),
        "excluded_conversational_train": sum(
            row.get("split") == "train" and bool(row.get("turns")) for row in rows
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def build_calibration_report(
    scored_rows: list[dict], reviews: list[dict], *, scorer_configuration: dict
) -> dict:
    """Measure the existing rubric, without tuning it or claiming population calibration.

    Visual acceptance is independent of SFT eligibility: an excluded conversation
    can still be a valid positive image judgment. All disagreements stay visible.
    """
    rows = {row["id"]: row for row in scored_rows}
    if len(rows) != len(scored_rows):
        raise ValueError("Duplicate generation IDs in calibration rows")
    observations, excluded, seen = [], [], set()
    for review in reviews:
        generation_id = review.get("source_generation_id")
        if generation_id in seen:
            raise ValueError("Duplicate image reviews require explicit adjudication")
        seen.add(generation_id)
        row = rows.get(generation_id)
        if not row:
            excluded.append({"generation_id": generation_id, "reason": "missing_scored_generation"})
            continue
        if (
            review.get("reviewer_kind")
            not in {"human", "assistant", "aria_reviewed", "astra_judge"}
            or not all(
                review.get(key) for key in ("reviewer_id", "reviewed_at", "evidence", "source_ref")
            )
            or type(review.get("visual_quality_pass")) is not bool
        ):
            excluded.append({"generation_id": generation_id, "reason": "incomplete_visual_review"})
            continue
        try:
            timestamp = datetime.fromisoformat(review["reviewed_at"])
            if timestamp.tzinfo is None:
                raise ValueError("timezone missing")
        except (ValueError, TypeError):
            raise ValueError("Review must have a timezone-aware reviewed_at timestamp") from None
        metadata = row.get("benchmark_metadata") or {}
        if metadata.get("split") not in {"train", "validation"}:
            raise ValueError("Heldout images must not be used to qualify the RL reward")
        if review.get("partition") != metadata["split"]:
            raise ValueError("Review partition does not match generation")
        requirements = metadata.get("requirements")
        validate_requirements(requirements)
        checks = review.get("requirements")
        if not isinstance(checks, list) or sorted(
            c.get("requirement", "") for c in checks
        ) != sorted(requirements):
            raise ValueError("Review requirements do not match generation")
        if any(type(c.get("satisfied")) is not bool or not c.get("evidence") for c in checks):
            raise ValueError("Review checklist requires boolean decisions and evidence")
        if review.get("prompt_id") != row.get("prompt_id"):
            raise ValueError("Review prompt does not match generation")
        if not review.get("png_sha256") or review["png_sha256"] != metadata.get("png_sha256"):
            raise ValueError("Review image hash does not match scored generation")
        score = row.get("scores", {}).get("scene_quality") or {}
        if (
            score.get("generation_id") != generation_id
            or score.get("scorer_configuration") != scorer_configuration
            or score.get("requirements") != requirements
        ):
            raise ValueError("Calibration score is not bound to current generation/rubric")
        checked, error = None, None
        try:
            if score.get("status") != "ok":
                raise ValueError(str(score.get("status")))
            checked = scene_result(score, requirements, threshold=scorer_configuration["threshold"])
        except (ValueError, TypeError, KeyError):
            error = "unusable_judgment"
        predicted = (
            None
            if checked is None
            else bool(not checked["quality_failure"] and row.get("valid") and not row.get("error"))
        )
        observation = {
            "generation_id": generation_id,
            "review": review,
            "score": score,
            "generation": {
                key: row.get(key)
                for key in ("id", "prompt_id", "valid", "error", "benchmark_metadata")
            },
            "reviewed_pass": review["visual_quality_pass"],
            "predicted_pass": predicted,
            "judge_error": error,
            "checklist_disagreements": [],
            "absolute_metric_errors": {},
        }
        if checked:
            judged = {c["requirement"]: c["satisfied"] for c in checked["prompt_checks"]}
            observation["checklist_disagreements"] = [
                c["requirement"] for c in checks if c["satisfied"] != judged[c["requirement"]]
            ]
            for metric in METRICS:
                label = review.get("identity" if metric == "similarity" else metric)
                if type(label) is not int or not 0 <= label <= 4:
                    raise ValueError("Visual review metrics must be integers 0..4")
                observation["absolute_metric_errors"][metric] = abs(checked[metric] - label / 4)
        observations.append(observation)
    positives = sum(item["reviewed_pass"] for item in observations)
    negatives = len(observations) - positives
    errors = sum(item["judge_error"] is not None for item in observations)
    false_positives = sum(
        item["predicted_pass"] is True and not item["reviewed_pass"] for item in observations
    )
    agreement = (
        sum(item["predicted_pass"] == item["reviewed_pass"] for item in observations)
        / len(observations)
        if observations
        else None
    )
    ready = bool(
        positives and negatives and not errors and not false_positives and agreement >= 0.8
    )
    return {
        "schema_version": 1,
        "status": "limited_pilot_agreement" if ready else "blocked_calibration",
        "rl_ready": ready,
        "population_calibrated": False,
        "scorer_configuration": scorer_configuration,
        "reviewed_count": len(observations),
        "positive_count": positives,
        "negative_count": negatives,
        "judge_errors": errors,
        "false_positives": false_positives,
        "agreement": agreement,
        "readiness_policy": {
            "minimum_positive": 1,
            "minimum_negative": 1,
            "maximum_false_positives": 0,
            "maximum_judge_errors": 0,
            "minimum_agreement": 0.8,
        },
        "observations": observations,
        "excluded_reviews": excluded,
        "evidence_sha256": digest(
            {"observations": observations, "configuration": scorer_configuration}
        ),
    }


def require_calibration(report: dict, *, scorer_configuration: dict) -> dict:
    """Recompute from embedded evidence; a hand-edited rl_ready flag grants nothing."""
    if report.get("scorer_configuration") != scorer_configuration:
        raise ValueError(
            "Calibration uses a different scorer configuration; review the current rubric"
        )
    observations = report.get("observations") or []
    rows = [
        item["generation"] | {"scores": {"scene_quality": item["score"]}} for item in observations
    ]
    rebuilt = build_calibration_report(
        rows, [item["review"] for item in observations], scorer_configuration=scorer_configuration
    )
    if rebuilt["evidence_sha256"] != report.get("evidence_sha256") or not rebuilt["rl_ready"]:
        raise ValueError(
            "RL blocked: reviewed positive/negative calibration disagrees or lacks evidence; inspect the calibration report"
        )
    return rebuilt


ASTRA_MIN_AGREEMENT = 0.8
ASTRA_CALIBRATION_SPLITS = frozenset({"train", "validation"})


def _astra_predicted_pass(score: dict, metadata: dict) -> bool:
    from bufogen.curation.judge_reviews import _accepts

    return _accepts(score, metadata)


def build_astra_calibration_report(
    scored_rows: list[dict], decisions: list[dict], *, scorer_configuration: dict
) -> dict:
    """Agreement of merged Astra rows with the human packet decisions; nothing is tuned.

    Each decision carries `generation_id` (joined from the unblinding key), `pass` and
    `missing` (0-based requirement indices the reviewer found unmet).
    """
    from bufogen.evaluation.scorers.astra_judge import config_key

    rows = {row["id"]: row for row in scored_rows}
    if len(rows) != len(scored_rows):
        raise ValueError("Duplicate generation IDs in calibration rows")
    observations, seen = [], set()
    for decision in decisions:
        gid = decision.get("generation_id")
        if gid in seen:
            raise ValueError("Duplicate decisions for one generation")
        seen.add(gid)
        row = rows.get(gid)
        if row is None:
            raise ValueError(f"Decision for {gid} has no scored generation")
        score = (row.get("scores") or {}).get("astra_judge") or {}
        metadata = row.get("benchmark_metadata") or {}
        requirements = metadata.get("requirements")
        if config_key(score.get("scorer_configuration")) != config_key(scorer_configuration):
            raise ValueError(f"Scored row {gid} is not bound to the calibrated judge configuration")
        if score.get("generation_id") != gid or not isinstance(requirements, list):
            raise ValueError(f"Scored row {gid} is not bound to its generation")
        missing = decision.get("missing")
        if (
            type(decision.get("pass")) is not bool
            or not isinstance(missing, list)
            or any(type(m) is not int or not 0 <= m < len(requirements) for m in missing)
        ):
            raise ValueError(f"Decision for {gid} is malformed")
        split = metadata.get("split")
        if split not in ASTRA_CALIBRATION_SPLITS:
            raise ValueError(
                f"Scored row {gid} is from split {split!r}; only train and validation "
                "evidence may calibrate the RL judge"
            )
        usable = score.get("status") == "ok"
        predicted = bool(
            usable
            and _astra_predicted_pass(score, metadata)
            and row.get("valid")
            and not row.get("error")
        )
        trimmed = {
            key: score.get(key)
            for key in (
                "status",
                "generation_id",
                "strict_pass",
                "judge_confidence",
                "vs_reference",
                "scorer_configuration",
            )
        }
        trimmed["prompt_checks"] = [
            {"requirement": c.get("requirement"), "satisfied": c.get("satisfied")}
            for c in score.get("prompt_checks") or []
        ]
        judged = {c["requirement"]: c["satisfied"] for c in trimmed["prompt_checks"]}
        observations.append(
            {
                "generation_id": gid,
                "decision": {
                    "index": decision.get("index"),
                    "pass": decision["pass"],
                    "vs_reference": decision.get("vs_reference"),
                    "missing": sorted(missing),
                },
                "score": trimmed,
                "generation": {
                    "id": row["id"],
                    "valid": row.get("valid"),
                    "error": row.get("error"),
                    "benchmark_metadata": {
                        "status": metadata.get("status"),
                        "split": split,
                        "requirements": requirements,
                    },
                },
                "reviewed_pass": decision["pass"],
                "predicted_pass": predicted,
                "judge_error": None if usable else str(score.get("status")),
                "judge_satisfied_but_reviewer_missing": [
                    requirements[m] for m in missing if judged.get(requirements[m]) is True
                ],
            }
        )
    confusion = {
        "judge_pass_reviewer_pass": 0,
        "judge_pass_reviewer_fail": 0,
        "judge_fail_reviewer_pass": 0,
        "judge_fail_reviewer_fail": 0,
    }
    # A judge error is a failure of measurement, not a judged negative: it is counted as an
    # error and kept out of the confusion matrix, the class counts and agreement.
    judged = [item for item in observations if item["judge_error"] is None]
    judge_errors = len(observations) - len(judged)
    for item in judged:
        confusion[
            f"judge_{'pass' if item['predicted_pass'] else 'fail'}"
            f"_reviewer_{'pass' if item['reviewed_pass'] else 'fail'}"
        ] += 1
    positives = sum(item["reviewed_pass"] for item in judged)
    negatives = len(judged) - positives
    pass_missing = sum(
        item["predicted_pass"] and bool(item["decision"]["missing"]) for item in judged
    )
    agreement = (
        sum(item["predicted_pass"] == item["reviewed_pass"] for item in judged) / len(judged)
        if judged
        else None
    )
    ready = bool(
        not judge_errors
        and positives
        and negatives
        and not pass_missing
        and agreement is not None
        and agreement >= ASTRA_MIN_AGREEMENT
    )
    return {
        "schema_version": 1,
        "kind": "astra_calibration",
        "status": "limited_pilot_agreement" if ready else "blocked_calibration",
        "rl_ready": ready,
        "population_calibrated": False,
        "scorer_configuration": scorer_configuration,
        "reviewed_count": len(observations),
        "positive_count": positives,
        "negative_count": negatives,
        "judge_errors": judge_errors,
        "confusion": confusion,
        "judge_pass_reviewer_missing": pass_missing,
        "agreement": agreement,
        "readiness_policy": {
            "minimum_positive": 1,
            "minimum_negative": 1,
            "maximum_judge_pass_reviewer_missing": 0,
            "maximum_judge_errors": 0,
            "calibration_splits": sorted(ASTRA_CALIBRATION_SPLITS),
            "minimum_agreement": ASTRA_MIN_AGREEMENT,
        },
        "observations": observations,
        "evidence_sha256": digest(
            {"observations": observations, "configuration": config_key(scorer_configuration)}
        ),
    }


def require_astra_calibration(report: dict, *, scorer_configuration: dict) -> dict:
    """Recompute from embedded evidence; a hand-edited rl_ready flag grants nothing."""
    from bufogen.evaluation.scorers.astra_judge import config_key

    if config_key(report.get("scorer_configuration")) != config_key(scorer_configuration):
        raise ValueError(
            "Calibration uses a different judge configuration; review the current rubric"
        )
    observations = report.get("observations") or []
    rows = [
        item["generation"] | {"scores": {"astra_judge": item["score"]}} for item in observations
    ]
    decisions = [
        item["decision"] | {"generation_id": item["generation_id"]} for item in observations
    ]
    rebuilt = build_astra_calibration_report(
        rows, decisions, scorer_configuration=scorer_configuration
    )
    if rebuilt["evidence_sha256"] != report.get("evidence_sha256") or not rebuilt["rl_ready"]:
        raise ValueError(
            "RL blocked: Astra calibration disagrees with the reviewer or lacks evidence; "
            "inspect the calibration report"
        )
    return rebuilt
