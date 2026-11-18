"""Read-only showcase assembled from saved evidence, never from demo constants."""

import base64
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from bufogen.evaluation.benchmark import _summary, _validate_rows
from bufogen.generation import Manifest
from bufogen.models import frozen_model_presentations, generation_model_presentation


def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _weave_url(ref):
    if not isinstance(ref, str) or not ref.startswith("weave:///"):
        return None
    parts = ref.removeprefix("weave:///").split("/")
    if len(parts) != 4 or parts[2] != "object" or ":" not in parts[3]:
        return None
    name, version = parts[3].rsplit(":", 1)
    return f"https://wandb.ai/{quote(parts[0])}/{quote(parts[1])}/weave/objects/{quote(name)}/versions/{quote(version)}"


def _benchmark(directory):
    settings = _json(directory / "benchmark.json")
    if settings is None:
        raise ValueError(f"Missing frozen benchmark.json in {directory}")
    summary = _json(directory / "summary.json")
    if summary and any(
        summary.get(k) != settings.get(k)
        for k in ("run_id", "configuration_hash", "comparison_fingerprint")
    ):
        raise ValueError(f"Summary does not match frozen benchmark: {directory}")
    rows = Manifest(directory / "manifest.jsonl").read()
    config = settings["configuration"]
    prompts = config["prompts"]
    arms = config["arms"]
    model_configs = frozen_model_presentations(arms)
    model_presentations = {
        name: generation_model_presentation({"backend": name, "model": arm["model"]}, config=arm)
        for name, arm in model_configs.items()
    }
    expected = len(prompts) * len(arms)
    keys = [(r.prompt_id, r.backend) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate generation rows cannot be counted in a comparison")
    _validate_rows(rows, settings)
    splits = sorted({p["split"] for p in prompts})
    status = "not_started" if not rows else "running" if len(rows) < expected else "generated"
    if summary:
        expected_keys = {(p["id"], arm["name"]) for p in prompts for arm in arms}
        counts = Counter(row.backend for row in rows)
        if (
            set(keys) != expected_keys
            or set(summary.get("arms", {})) != {a["name"] for a in arms}
            or any(summary["arms"][a["name"]].get("total") != counts[a["name"]] for a in arms)
        ):
            raise ValueError("Summary coverage differs from the frozen generation manifest")
        scored = Manifest(directory / "scored.jsonl").read()
        original = {row.id: row.to_dict() for row in rows}
        if len(scored) != len(rows) or {row.id for row in scored} != set(original):
            raise ValueError("Summary requires complete, unique saved scored-row coverage")
        for row in scored:
            if any(
                value != row.to_dict().get(key)
                for key, value in original[row.id].items()
                if key != "scores"
            ):
                raise ValueError("Scored row differs from its source generation manifest")
            result = row.scores.get("scene_quality", {})
            if (
                result.get("generation_id") != row.id
                or result.get("scorer_configuration") != config.get("scorer")
                or result.get("requirements") != row.benchmark_metadata["requirements"]
            ):
                raise ValueError("Scored row is not bound to frozen judge/source requirements")
        if _summary(scored, settings) != summary:
            raise ValueError("Summary metrics differ from the saved scored rows")
        status = "scored"
    publication = _json(directory / "weave-publication.json") or {}
    evaluations = [
        {
            "backend": name,
            "ref": entry.get("ref"),
            "url": _weave_url(entry.get("ref")),
            "scored_rows": entry.get("scored_rows"),
        }
        for name, entry in publication.get("evaluations", {}).items()
    ]
    return {
        "directory": str(directory.resolve()),
        "run_id": settings["run_id"],
        "comparison_fingerprint": settings.get("comparison_fingerprint"),
        "split_kind": "heldout" if set(splits) <= {"heldout", "test"} else "pilot/development",
        "splits": splits,
        "expected_generations": expected,
        "generated": len(rows),
        "selected_prompts": len(prompts),
        "status": status,
        "scorer_configuration": config.get("scorer"),
        "arms": (summary or {}).get("arms", {}),
        "models": arms,
        "model_presentations": model_presentations,
        "evaluations": evaluations,
        "publication_errors": publication.get("errors", {}),
        "queue": _json(directory / "review-queue.json"),
        "rows": [
            {
                **{k: v for k, v in r.to_dict().items() if k not in {"svg", "trajectory"}},
                **generation_model_presentation(r, config=model_configs.get(r.backend)),
            }
            for r in rows
        ],
    }


def build_loop_report(
    root,
    *,
    benchmark_dirs=None,
    forge_state=None,
    review_paths=None,
    calibration_path=None,
    rl_state=None,
):
    root = Path(root)
    directories = (
        benchmark_dirs
        if benchmark_dirs is not None
        else sorted(p.parent for p in root.glob("*/benchmark.json"))
    )
    runs = [_benchmark(Path(p)) for p in directories]
    # Keep incompatible configurations separate. Never sum repeated arms/reruns.
    groups = defaultdict(list)
    for run in runs:
        fingerprint = run["comparison_fingerprint"]
        groups[fingerprint or f"unqualified:{run['run_id']}"].append(run)
    comparisons = []
    for fingerprint, group in groups.items():
        seen = set()
        duplicate_arms = set()
        for run in group:
            duplicate_arms.update(seen.intersection(run["arms"]))
            seen.update(run["arms"])
        comparisons.append(
            {
                "fingerprint": fingerprint,
                "run_ids": [r["run_id"] for r in group],
                "status": "ambiguous_repeated_arms"
                if duplicate_arms
                else "compatible"
                if not fingerprint.startswith("unqualified:")
                else "unqualified",
                "duplicate_arms": sorted(duplicate_arms),
                "note": "Per-run denominators retained. Repeated arms are not pooled; no parity inference.",
            }
        )
    state = _json(Path(forge_state) if forge_state else root / "forge-state.json") or {}
    calibration = (
        _json(
            Path(calibration_path) if calibration_path else root / "calibration/rl-calibration.json"
        )
        or {}
    )
    limitations = _json(root / "calibration/limitations.json") or {}
    rl = _json(Path(rl_state)) if rl_state else None
    if rl and isinstance(rl.get("calibration"), dict):
        rl["calibration"] = {k: v for k, v in rl["calibration"].items() if k != "observations"}
    row_by_id = {r["id"]: r for run in runs for r in run["rows"]}
    review_paths = (
        review_paths
        if review_paths is not None
        else sorted((root / "pilot").glob("*reviews*.jsonl"))
    )
    reviews = {}
    for path in review_paths:
        for review in _jsonl(Path(path)):
            key = review["source_generation_id"]
            if key in reviews and reviews[key] != review:
                raise ValueError("Conflicting reviews in report inputs; resolve explicitly")
            reviews[key] = review
    counts = defaultdict(Counter)
    for generation_id, review in reviews.items():
        row = row_by_id.get(generation_id)
        arm = row["backend"] if row else "outside_selected_benchmarks"
        counts[arm]["reviewed"] += 1
        counts[arm]["visual_pass"] += review.get("visual_quality_pass") is True
        counts[arm]["accepted_target"] += review.get("decision") == "accept"
        counts[arm][f"reviewer:{review.get('reviewer_kind', 'unknown')}"] += 1
    calibration_summary = {k: v for k, v in calibration.items() if k != "observations"}
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "benchmarks": runs,
        "comparison_groups": comparisons,
        "phases": {
            "Run": {
                "status": "recorded" if any(r["generated"] for r in runs) else "not_started",
                "proxy_receipt": _json(root / "proxy-preflight.json"),
            },
            "Observe": {
                "status": "scored" if any(r["arms"] for r in runs) else "not_started",
                "calibration": calibration_summary,
                "pilot_visualization": _json(root / "pilot-visualization.json"),
                "agent_lens_status": _json(root / "agent-lens-status.json"),
                "limitations": limitations,
            },
            "Curate": {
                "status": "reviewed" if reviews else "not_started",
                "review_counts": dict(counts),
                "review_sources": [str(Path(p).resolve()) for p in review_paths],
                "dataset_id": state.get("dataset_id"),
                "entry_counts": state.get("entry_counts"),
                "dataset_artifact_ref": state.get("dataset_artifact_ref"),
            },
            "Improve": {
                "status": state.get("status", "not_started"),
                "pending_operation": state.get("pending_operation"),
                "last_error": state.get("last_error"),
                "finetune_id": state.get("finetune_id"),
                "model_artifact_ref": state.get("model_artifact_ref"),
                "forge_url": "https://forge.coreweave.com/post-train/distillation"
                if state
                else None,
                "rl": rl or {"status": "not_started"},
                "rl_recovery_failure": _json(root / "rl-recovery-failure.json"),
                "rl_environment": _json(root / "rl-environment-artifact.json"),
            },
            "Evaluate": {
                "heldout_baseline_visualization": _json(
                    root / "heldout-baseline-visualization.json"
                ),
                "heldout_visualization": _json(root / "heldout-visualization.json"),
                "status": "recorded"
                if any(r["split_kind"] == "heldout" and r["arms"] for r in runs)
                else "not_started",
                "note": "Pilot visual results are development evidence. Heldout results, failed attempts, judge abstentions and sample counts remain separate; no gpt-6.1-sol parity claim.",
            },
        },
        "limitations": [
            "Visual scores are provisional VLM judgments, not embedding similarity or population-calibrated probabilities.",
            "Assistant image reviews are not human annotation-queue approvals.",
            "The teacher and visual judge use gpt-6.1-sol; correlated preference remains possible.",
            "Training completion is not evidence of improved generation. Compare immutable checkpoints on matching heldout runs.",
        ],
    }


def write_loop_report(root, output, **kwargs):
    report = build_loop_report(root, **kwargs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "loop-report.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    def esc(value):
        return html.escape(str(value), quote=True)

    def evidence_links(value):
        found = []
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and item.startswith("https://"):
                    found.append(f'<a href="{esc(item)}">{esc(key.replace("_", " "))}</a>')
                elif isinstance(item, dict):
                    found.extend(evidence_links(item))
        return found

    sections = []
    for title, phase in report["phases"].items():
        sections.append(
            f"<article><h2>{esc(title)}</h2><strong>{esc(phase['status'])}</strong><p>{' · '.join(evidence_links(phase))}</p><details><summary>Evidence</summary><pre>{esc(json.dumps(phase, indent=2))}</pre></details></article>"
        )
    tables = []
    galleries = []
    for run in report["benchmarks"]:
        title = f"{run['split_kind']} · {run['selected_prompts']} prompts · {run['status']}"
        rows = []
        for arm, score in run["arms"].items():
            presentation = run["model_presentations"][arm]
            label = esc(presentation["model_label"])
            if presentation["model_version"]:
                label += f"<br><small>{esc(presentation['model_version'])}</small>"
            count = score["total"]
            passed = score["provisional_passes"]
            rows.append(
                f"<tr><td>{label}</td><td>{passed}/{count}</td><td>{score['judge_errors']}</td><td>{score['generation_errors']}</td><td>{esc(json.dumps(score['metric_means_scored_only']))}</td></tr>"
            )
        links = []
        if run["queue"] and run["queue"].get("url"):
            links.append(f'<a href="{esc(run["queue"]["url"])}">Annotation queue</a>')
        for evaluation in run["evaluations"]:
            if evaluation["url"]:
                links.append(
                    f'<a href="{esc(evaluation["url"])}">{esc(run["model_presentations"][evaluation["backend"]]["model_label"])} evaluation object</a>'
                )
        tables.append(
            f"<section><h2>{esc(title)}</h2><p>Run {esc(run['run_id'])} · {run['generated']}/{run['expected_generations']} generations</p><p>Frozen scorer: {esc(json.dumps(run['scorer_configuration']))}</p><p>{' · '.join(links)}</p><table><thead><tr><th>Arm</th><th>Provisional passes / all attempts</th><th>Judge errors</th><th>Generation errors</th><th>Means (scored only)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
        )
        by_prompt = defaultdict(list)
        for row in run["rows"]:
            by_prompt[row["prompt_id"]].append(row)
        for prompt_id, examples in list(by_prompt.items())[:12]:
            cards = []
            for row in examples:
                image = "<p>No rendered image retained</p>"
                if row.get("png_path") and Path(row["png_path"]).is_file():
                    data = base64.b64encode(Path(row["png_path"]).read_bytes()).decode()
                    image = f'<img alt="{esc(row["model_label"])} rendering of {esc(row["prompt"])}" src="data:image/png;base64,{data}">'
                trace = (
                    f'<a href="{esc(row["conversation_url"])}">Conversation trace</a>'
                    if row.get("conversation_url")
                    else "Trace unavailable"
                )
                version = (
                    f"<p><small>{esc(row['model_version'])}</small></p>"
                    if row["model_version"]
                    else ""
                )
                cards.append(
                    f"<figure><figcaption>{esc(row['model_label'])}</figcaption>{version}{image}<p>{esc(row.get('response_text', ''))}</p>{trace}</figure>"
                )
            galleries.append(
                f'<section><h3>{esc(prompt_id)}</h3><p>{esc(examples[0]["prompt"])}</p><div class="gallery">{"".join(cards)}</div></section>'
            )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Bufo · Forge AI Loop</title><style>
    body{font:16px/1.5 system-ui,sans-serif;background:#f4f6ef;color:#223126;margin:0}main{max-width:1400px;margin:auto;padding:40px}h1{font-size:44px;letter-spacing:-2px}h2{font-size:24px}.phases,.gallery{display:flex;gap:16px;flex-wrap:wrap}article{flex:1;min-width:170px;padding:20px;background:#dce8d5;border-radius:12px}section{margin:32px 0;padding:24px;background:white;border-radius:12px}strong{color:#375c39}a{color:#2458a0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce1da;text-align:left;padding:12px;vertical-align:top}figure{flex:1;min-width:220px;max-width:400px;margin:0}img{width:100%;background:#eef0eb;border-radius:8px}figcaption{font-weight:700;margin-bottom:8px}p{overflow-wrap:anywhere}details{margin-top:12px}.note{max-width:950px;color:#526455}</style><main><p>BUFO / FORGE</p><h1>From production drawings to a better agent.</h1>"""
    page += f'<p class="note">Saved evidence as of {esc(report["generated_at"])}. This report makes no network requests and never starts training.</p><div class="phases">{"".join(sections)}</div>'
    page += (
        "<section><h2>Reading the evidence</h2><ul>"
        + "".join(f"<li>{esc(item)}</li>" for item in report["limitations"])
        + "</ul><p>Comparisons are grouped only by matching frozen fingerprints. Separate runs and repeated arms are never pooled.</p><pre>"
        + esc(json.dumps(report["comparison_groups"], indent=2))
        + "</pre></section>"
    )
    review_rows = []
    for arm, counts in report["phases"]["Curate"]["review_counts"].items():
        reviewers = {
            k.removeprefix("reviewer:"): v for k, v in counts.items() if k.startswith("reviewer:")
        }
        review_rows.append(
            f"<tr><td>{esc(arm)}</td><td>{counts['visual_pass']}/{counts['reviewed']}</td>"
            f"<td>{counts['accepted_target']}</td><td>{esc(json.dumps(reviewers))}</td></tr>"
        )
    if review_rows:
        page += (
            "<section><h2>Actual image reviews</h2><p>Visual passes differ from accepted SFT targets: "
            "trajectory provenance and partition rules can exclude a visually good image.</p>"
            "<table><thead><tr><th>Arm</th><th>Visual passes / reviewed</th>"
            "<th>Accepted targets</th><th>Reviewer provenance</th></tr></thead><tbody>"
            + "".join(review_rows)
            + "</tbody></table></section>"
        )
    page += (
        "".join(tables)
        + "<h2>Paired images</h2><p>First 12 prompts per saved benchmark. All attempts remain in the tables and JSON report.</p>"
        + "".join(galleries)
        + "</main></html>"
    )
    destination = output / "index.html"
    destination.write_text(page)
    return destination
