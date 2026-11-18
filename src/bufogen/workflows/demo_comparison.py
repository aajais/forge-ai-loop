"""Export saved validation evidence without inference; publication is explicitly opt-in.

Matched arms require identical generation policy/runtime/source and per-prompt references.
Historical reference arms are displayed with disclosed differences, never paired statistically.
Only selected validation rows are exported, not the other splits in a source run.
"""

import argparse
import base64
import fcntl
import hashlib
import html
import itertools
import json
import math
import os
import re
import statistics
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from bufogen.evaluation.scorers.astra_judge import config_key
from bufogen.evaluation.statistics import paired_family_interval

METRICS = (
    "similarity",
    "anatomy",
    "composition",
    "readability",
    "requirement_coverage",
    "judge_confidence",
)
ROOT = Path(__file__).resolve().parents[3]
EXPORT_FILES = frozenset({"comparison.json", "manifest.json", "index.html"})


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _locked(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".comparison.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _path(value, root):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _runtime(config, arm):
    # Full fingerprints are stricter than paired_across_runs, which only guards the judge,
    # reference drawings and prompt text. A runtime loaded-code hash omits client timeouts.
    return {
        key: config[key] for key in ("runtime", "source_hashes", "policy", "prompt_version")
    } | {
        "sampling": arm["sampling"],
        "runtime_policy": arm.get("runtime_policy"),
        "transport": {key: arm.get(key) for key in ("provider", "base_url", "supports_tools")},
    }


def _training_metadata(spec, root, arm):
    """Resolve exact checkpoint lineage; never infer epochs from an arm name."""
    if not spec.get("lineage_path"):
        return None
    entries = _read(_path(spec["lineage_path"], root))
    matches = [
        entry
        for entry in entries
        if entry.get("artifact") == arm["model"] and entry.get("arm_name") == spec["backend"]
    ]
    if len(matches) != 1:
        raise ValueError("Require unique exact-checkpoint training lineage")
    entry = matches[0]
    epochs = entry.get("sft_config", {}).get("epochs")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Training lineage needs a positive integer epoch count")
    return {
        key: entry[key]
        for key in ("artifact", "arm_name", "sft_config", "forge_job_id", "forge_dataset_id")
        if key in entry
    }


def _eval_url(ref):
    if not ref or not ref.startswith("weave:///"):
        return None
    parts = ref.removeprefix("weave:///").split("/")
    if len(parts) != 4 or parts[2] != "object" or ":" not in parts[3]:
        return None
    name, version = parts[3].rsplit(":", 1)
    return f"https://wandb.ai/{parts[0]}/{parts[1]}/weave/objects/{quote(name, safe='')}/versions/{quote(version, safe='')}"


def _strict(row):
    judge = row.get("scores", {}).get("astra_judge", {})
    return bool(
        row["valid"]
        and not row.get("error")
        and row["benchmark_metadata"]["status"] == "completed"
        and judge.get("status") == "ok"
        and judge.get("strict_pass") is True
    )


def _load_arm(spec, root, prompts=None):
    source = _path(spec["source"], root)
    scored = source / spec.get("scored_dir", "astra-v5-k3")
    paths = {
        "benchmark": source / "benchmark.json",
        "scorer": scored / "scorer.json",
        "scores": scored / "scored.jsonl",
    }
    for name, expected in spec.get("sha256", {}).items():
        if name not in paths or _sha(paths[name]) != expected:
            raise ValueError(f"Frozen evidence digest changed: {spec['id']}/{name}")
    settings, scorer = _read(paths["benchmark"]), _read(paths["scorer"])
    config = settings["configuration"]
    selected = {p["id"]: p for p in config["prompts"] if p["split"] == "validation"}
    if prompts is None:
        prompts = selected
    if not prompts or any(selected.get(pid) != prompt for pid, prompt in prompts.items()):
        raise ValueError("Validation prompt text/requirements/family differ between arms")
    arms = [a for a in config["arms"] if a["name"] == spec["backend"]]
    if len(arms) != 1 or arms[0]["model"] != spec["model_id"]:
        raise ValueError("Exact model ID differs from selected frozen arm")
    arm = arms[0]
    tuned = bool(arm.get("tuning_method")) or arm["model"].startswith("wandb-artifact:")
    if tuned and not re.fullmatch(r"wandb-artifact:///[^/\s]+/[^/\s]+/[^/:\s]+:v\d+", arm["model"]):
        raise ValueError("Fine-tuned checkpoint must use an immutable vN artifact version")
    if scorer.get("self_reference") or scorer.get("confirmation"):
        raise ValueError("Self-reference/confirmation scoring is not validation evidence")
    refs_path = _path(scorer["references"]["path"], root)
    if _sha(refs_path) != scorer["references"]["sha256"]:
        raise ValueError("Reference manifest digest changed")
    refs = _read(refs_path)["rows"]
    rows = []
    with paths["scores"].open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["backend"] != spec["backend"] or row["prompt_id"] not in prompts:
                continue
            meta, prompt = row.get("benchmark_metadata") or {}, prompts[row["prompt_id"]]
            if (
                row["model"] != arm["model"]
                or row["prompt"] != prompt["prompt"]
                or (
                    row.get("prompt_version") != config["prompt_version"]
                    and not (
                        meta.get("status") == "generation_error"
                        and row.get("prompt_version") is None
                    )
                )
                or meta.get("split") != "validation"
                or meta.get("family") != prompt["family"]
                or meta.get("requirements") != prompt["requirements"]
                or meta.get("run_id") != settings["run_id"]
                or meta.get("configuration_hash") != settings["configuration_hash"]
            ):
                raise ValueError(f"Frozen row identity or prompt mismatch: {row['id']}")
            judge = row.get("scores", {}).get("astra_judge", {})
            if (
                config_key(judge.get("scorer_configuration")) != config_key(scorer["judge"])
                or judge.get("generation_id") != row["id"]
            ):
                raise ValueError(f"Saved judge configuration/identity mismatch: {row['id']}")
            reference = judge.get("reference") or {}
            if (
                reference.get("is_self")
                or reference.get("png_sha256") != refs[row["prompt_id"]]["png_sha256"]
            ):
                raise ValueError(f"Saved reference digest mismatch: {row['id']}")
            if meta.get("status") not in ("completed", "generation_error", "error", "failed"):
                raise ValueError(f"Nonterminal generation: {row['id']}")
            png = _path(row["png_path"], root) if row.get("png_path") else None
            image_bytes = png.read_bytes() if png and png.is_file() else None
            if image_bytes and (
                not image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                or _sha(png) != meta.get("png_sha256")
            ):
                raise ValueError(f"PNG digest mismatch: {row['id']}")
            # Saved judgments remain historical facts if local PNGs were lost afterwards.
            # Missing file counts are separate and always visible.
            rows.append(
                {
                    "id": row["id"],
                    "prompt_id": row["prompt_id"],
                    "model_id": row["model"],
                    "prompt": row["prompt"],
                    "family": meta["family"],
                    "requirements": meta["requirements"],
                    "strict_pass": _strict(row),
                    "saved_judge": judge,
                    "valid": row["valid"],
                    "generation_status": meta["status"],
                    "error": row.get("error"),
                    "latency_s": row["latency_s"],
                    "input_tokens": row.get("input_tokens"),
                    "output_tokens": row.get("output_tokens"),
                    "tool_calls": row.get("tool_calls"),
                    "trace_url": row.get("conversation_url"),
                    "weave_call_id": row.get("weave_call_id"),
                    "png_sha256": _sha(png) if image_bytes else None,
                    "image_data_url": "data:image/png;base64,"
                    + base64.b64encode(image_bytes).decode()
                    if image_bytes
                    else None,
                    "image_status": "available" if image_bytes else "missing_png",
                }
            )
    if len(rows) != len(prompts) or {r["prompt_id"] for r in rows} != set(prompts):
        raise ValueError("Incomplete or duplicate paired prompt coverage")
    publication_path = scored / "weave-publication.json"
    evaluation = (
        _read(publication_path).get("evaluations", {}).get(spec["backend"], {})
        if publication_path.exists()
        else {}
    )
    label = (
        "ft-" + arm["base_model"]
        if tuned and arm.get("base_model")
        else ("ft-model (base ID unrecorded)" if tuned else arm["model"])
    )
    training = _training_metadata(spec, root, arm)
    arm_label = spec.get("label", spec["id"])
    if arm.get("tuning_method") == "sft":
        if training:
            epochs = training["sft_config"]["epochs"]
            arm_label = f"SFT · {epochs} {'epoch' if epochs == 1 else 'epochs'} · {arm['model'].rsplit(':', 1)[1]}"
        else:
            # Missing lineage must not turn arbitrary caller-provided epoch text into evidence.
            arm_label = f"SFT · {arm['model'].rsplit(':', 1)[1]} · training epochs unverified"
    return prompts, {
        "id": spec["id"],
        "arm_label": arm_label,
        "model_label": label,
        "model_id": arm["model"],
        "checkpoint": arm["model"] if tuned else None,
        "training_metadata": training,
        "tuning_method": arm.get("tuning_method") or None,
        "role": spec.get("role", "matched"),
        "caveat": spec.get("caveat"),
        "source": str(source),
        "run_id": settings["run_id"],
        "configuration_hash": settings["configuration_hash"],
        "source_sha256": {k: _sha(p) for k, p in paths.items()},
        "runtime": _runtime(config, arm),
        "scorer": scorer,
        "reference_digests": {pid: refs[pid]["png_sha256"] for pid in prompts},
        "evaluation_ref": evaluation.get("ref"),
        "evaluation_url": _eval_url(evaluation.get("ref")),
        "rows": sorted(rows, key=lambda r: r["prompt_id"]),
    }


def _summary(rows):
    scored = [r for r in rows if r["saved_judge"].get("status") == "ok"]
    times = [
        r["latency_s"]
        for r in rows
        if isinstance(r["latency_s"], (float, int)) and math.isfinite(r["latency_s"])
    ]
    passes = sum(r["strict_pass"] for r in rows)
    result = {
        "attempts": len(rows),
        "strict_passes": passes,
        "strict_pass_rate_all_attempts": passes / len(rows),
        "scored_images": len(scored),
        "generation_errors": sum(
            not r["valid"] or bool(r["error"]) or r["generation_status"] != "completed"
            for r in rows
        ),
        "judge_errors": sum(
            r["saved_judge"].get("status") not in ("ok", "missing_image") for r in rows
        ),
        "missing_png_files": sum(r["image_status"] != "available" for r in rows),
        "latency_mean_s_all_attempts": statistics.mean(times) if times else None,
        "latency_median_s_all_attempts": statistics.median(times) if times else None,
        "latency_observations": len(times),
    }
    for metric in METRICS:
        values = [
            r["saved_judge"][metric]
            for r in scored
            if isinstance(r["saved_judge"].get(metric), (float, int))
        ]
        result[metric + "_mean_scored_only"] = statistics.mean(values) if values else None
    return result


def _render(report):
    def esc(value):
        return html.escape(str(value))

    def link(url, label):
        return (
            f'<a href="{esc(url)}">{esc(label)}</a>'
            if url and url.startswith(("https://", "http://"))
            else f"<span>{esc(label)} unavailable</span>"
        )

    page = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Bufo Forge · frozen validation comparison</title><style>body{font:16px system-ui,sans-serif;margin:0;background:#f5f3ea;color:#19382b}main{max-width:1600px;margin:auto;padding:32px}h1{font-size:36px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #d9dfd4;text-align:left;vertical-align:top}th{background:#e5eddf}code,small{overflow-wrap:anywhere}img{width:100%;max-width:330px;background:white}article{background:white;border:1px solid #d9dfd4;padding:18px}.grid{display:grid;grid-template-columns:repeat('
        + str(len(report["arms"]))
        + ",minmax(180px,1fr));gap:12px;overflow:auto}.warning{background:#fff0c9;padding:16px}.error{color:#9b2e21}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#166c59}section{margin:28px 0}.scroll{overflow:auto}meter{width:100%}</style><main><p>BUFO FORGE / SAVED EVIDENCE</p><h1>Frozen validation comparison</h1>",
        "<p>All attempted validation prompts remain in the denominator. Strict pass uses saved Astra judgments and completed valid generations. Metrics marked “scored only” exclude missing judgments. No new inference was run.</p>",
        '<p class="warning">Exploratory validation evidence, not held-out confirmation or a parity claim. Generation latency includes retries and failures. Missing local PNG files do not rewrite historical saved judgments.</p>',
    ]
    if report["manifest"].get("notes"):
        page.append(f'<p class="warning">{esc(report["manifest"]["notes"])}</p>')
    page.append(
        '<div class="scroll"><table><tr><th>Exact model / checkpoint</th><th>Arm</th><th>Strict pass / all attempts</th><th>Coverage, scored only</th><th>Latency mean / median</th><th>Generation / judge errors / missing PNG</th><th>Evidence</th></tr>'
    )
    for arm in report["arms"]:
        s = arm["summary"]
        coverage = s["requirement_coverage_mean_scored_only"]
        page.append(
            f'<tr><td><b>{esc(arm["model_label"])}</b><br><small>{esc(arm["model_id"])}</small></td><td>{esc(arm["arm_label"])}<br>{esc(arm["role"])}</td><td>{s["strict_passes"]}/{s["attempts"]} ({s["strict_pass_rate_all_attempts"]:.1%})<meter min="0" max="1" value="{s["strict_pass_rate_all_attempts"]}"></meter></td><td>{esc(round(coverage, 3) if coverage is not None else "unavailable")} ({s["scored_images"]} scored)</td><td>{s["latency_mean_s_all_attempts"]:.1f}s / {s["latency_median_s_all_attempts"]:.1f}s</td><td>{s["generation_errors"]} / {s["judge_errors"]} / {s["missing_png_files"]}</td><td>{link(arm["evaluation_url"], "Original evaluation")}</td></tr>'
        )
    page.append("</table></div>")
    for arm in report["arms"]:
        if arm["caveat"]:
            page.append(
                f'<p class="warning"><b>{esc(arm["model_label"])} — {esc(arm["role"])}</b>: {esc(arm["caveat"])}</p>'
            )
    page.append(
        "<h2>Paired uncertainty · matched runtime only</h2><p>95% family-cluster bootstrap intervals; families are resampled together. Differences are left minus right. Validation selection and small family counts limit generalization.</p><ul>"
    )
    for pair in report["paired_intervals"]:
        interval = pair["interval"]
        page.append(
            f"<li>{esc(pair['left'])} minus {esc(pair['right'])}: {interval['difference']:+.1%}, 95% CI [{interval['ci95'][0]:+.1%}, {interval['ci95'][1]:+.1%}]; {interval['families']} families.</li>"
        )
    page.append(
        "</ul><details><summary>Frozen runtime, judge and provenance</summary><pre>"
        + esc(
            json.dumps(
                [
                    {k: v for k, v in a.items() if k not in ("rows", "summary")}
                    for a in report["arms"]
                ],
                indent=2,
            )
        )
        + "</pre></details>"
    )
    for prompt in report["prompts"]:
        page.append(
            f'<section><h2>{esc(prompt["id"])}</h2><p>{esc(prompt["prompt"])}</p><div class="grid">'
        )
        for arm in report["arms"]:
            row = next(r for r in arm["rows"] if r["prompt_id"] == prompt["id"])
            image = (
                f'<img loading="lazy" src="{row["image_data_url"]}" alt="{esc(arm["model_label"])} drawing">'
                if row["image_data_url"]
                else '<p class="error">Missing PNG</p>'
            )
            page.append(
                f'<article><h3>{esc(arm["model_label"])}</h3><small>{esc(arm["arm_label"])}</small>{image}<p><b>{"PASS" if row["strict_pass"] else "FAIL / UNSCORED"}</b> · {row["latency_s"]:.1f}s</p><p class="error">{esc(row["error"] or "")}</p><p>{link(row["trace_url"], "Original trace")}</p><details><summary>Saved judge evidence</summary><pre>{esc(json.dumps(row["saved_judge"], indent=2))}</pre></details></article>'
            )
        page.append("</div></section>")
    return "".join(page) + "</main></html>"


def build_comparison(manifest_path, output, *, root=ROOT):
    manifest_path, output, root = Path(manifest_path), Path(output), Path(root)
    manifest = _read(manifest_path)
    if manifest.get("version") != 1 or not manifest.get("arms"):
        raise ValueError("Require comparison manifest version 1 with selected arms")
    if len({a["id"] for a in manifest["arms"]}) != len(manifest["arms"]):
        raise ValueError("Duplicate comparison arm IDs")
    prompts, arms, baseline = None, [], None
    for spec in manifest["arms"]:
        prompts, arm = _load_arm(spec, root, prompts)
        if len(prompts) != manifest["expected_prompts"]:
            raise ValueError("Unexpected validation prompt count")
        if baseline is None:
            if arm["role"] != "matched":
                raise ValueError("First arm must be the matched baseline")
            baseline = arm
        if (
            config_key(arm["scorer"]["judge"]) != config_key(baseline["scorer"]["judge"])
            or arm["reference_digests"] != baseline["reference_digests"]
        ):
            raise ValueError("Judge/rubric or per-prompt reference differs between arms")
        differences = {
            k: {"baseline": baseline["runtime"][k], "arm": v}
            for k, v in arm["runtime"].items()
            if v != baseline["runtime"][k]
        }
        if arm["role"] == "matched" and differences:
            raise ValueError(
                f"Runtime/policy/source differs for matched arm {arm['id']}: {list(differences)}"
            )
        if arm["role"] not in ("matched", "historical_reference") or (
            arm["role"] == "historical_reference" and not arm["caveat"]
        ):
            raise ValueError("Historical reference requires an explicit comparability caveat")
        arm["runtime_differences"] = differences
        arm["summary"] = _summary(arm["rows"])
        arms.append(arm)
    intervals = []
    for left, right in itertools.combinations([a for a in arms if a["role"] == "matched"], 2):
        rows = [
            SimpleNamespace(
                prompt_id=r["prompt_id"],
                backend=a["id"],
                benchmark_metadata={"family": r["family"]},
                strict_pass=r["strict_pass"],
            )
            for a in (left, right)
            for r in a["rows"]
        ]
        intervals.append(
            {
                "left": left["id"],
                "right": right["id"],
                "interval": paired_family_interval(
                    rows, left["id"], right["id"], pass_fn=lambda r: int(r.strict_pass)
                ),
            }
        )
    report = {
        "schema_version": 1,
        "split": "validation",
        "manifest": manifest,
        "prompts": sorted(prompts.values(), key=lambda p: p["id"]),
        "arms": arms,
        "paired_intervals": intervals,
    }
    report["evidence_id"] = _digest(report)
    with _locked(output):
        if (output / "comparison.json").exists():
            prior = _read(output / "comparison.json")
            if prior.get("evidence_id") != report["evidence_id"]:
                raise ValueError("Output contains different evidence; use a new output directory")
        _write(output / "comparison.json", report)
        _write(output / "manifest.json", manifest)
        (output / "index.html").write_text(_render(report))
        _write(
            output / "files.json",
            {
                name: _sha(output / name)
                for name in ("comparison.json", "manifest.json", "index.html")
            },
        )
    return report


def publish_comparison(output, *, entity, project, wandb_module=None):
    """Retry with a stable W&B run ID; preserve successful receipts and record failures.

    A crash between remote artifact upload and receipt update can create another artifact
    version on retry. The evidence digest and run ID remain stable; no inference is repeated.
    """
    output = Path(output)
    with _locked(output):
        files = _read(output / "files.json")
        if not isinstance(files, dict) or set(files) != EXPORT_FILES:
            raise ValueError(
                "Invalid export inventory: require exactly the generated evidence files"
            )
        if any(
            (output / name).resolve().parent != output.resolve() for name in (*files, "files.json")
        ):
            raise ValueError("Invalid export inventory: evidence files must stay within output")
        if any(_sha(output / name) != expected for name, expected in files.items()):
            raise ValueError("Local evidence changed after export")
        report = _read(output / "comparison.json")
        if (
            _digest({k: v for k, v in report.items() if k != "evidence_id"})
            != report["evidence_id"]
        ):
            raise ValueError("Evidence digest mismatch")
        destination = f"{entity}/{project}"
        receipt_path = output / "publication.json"
        receipt = (
            _read(receipt_path)
            if receipt_path.exists()
            else {
                "evidence_id": report["evidence_id"],
                "destination": destination,
                "run_id": "bufo-demo-" + _digest([destination, report["evidence_id"]])[:16],
                "attempts": 0,
            }
        )
        if receipt["destination"] != destination or receipt["evidence_id"] != report["evidence_id"]:
            raise ValueError("Publication destination/evidence differs from receipt")
        if receipt.get("status") == "published":
            return receipt
        receipt.update(status="publishing", attempts=receipt["attempts"] + 1)
        _write(receipt_path, receipt)
        run = None
        try:
            if wandb_module is None:
                import wandb as wandb_module
            wb = wandb_module
            run = wb.init(
                entity=entity,
                project=project,
                id=receipt["run_id"],
                resume="allow",
                name="Bufo Forge frozen validation comparison",
                job_type="matched-model-comparison",
                mode="online",
                config={"evidence_id": report["evidence_id"], "validation_only": True},
            )
            if getattr(getattr(run, "settings", None), "mode", None) not in (None, "online"):
                raise RuntimeError("Offline/disabled W&B mode cannot confirm publication")
            metrics = list(report["arms"][0]["summary"])
            columns = [
                "arm_id",
                "model_id",
                "model_label",
                "checkpoint",
                "role",
                "caveat",
                "evaluation_url",
            ] + metrics
            data = [
                [
                    a["id"],
                    a["model_id"],
                    a["model_label"],
                    a["checkpoint"],
                    a["role"],
                    a["caveat"],
                    a["evaluation_url"],
                ]
                + [a["summary"][k] for k in metrics]
                for a in report["arms"]
            ]
            summary = wb.Table(columns=columns, data=data)
            matched = [a for a in report["arms"] if a["role"] == "matched"]
            chart = wb.Table(
                columns=["arm_id", "strict_pass_rate_all_attempts"],
                data=[[a["id"], a["summary"]["strict_pass_rate_all_attempts"]] for a in matched],
            )
            image_columns = ["prompt_id", "prompt"] + [
                a["id"] + suffix
                for a in report["arms"]
                for suffix in ("_image", "_strict_pass", "_error", "_trace", "_judge_status")
            ]
            image_table = wb.Table(columns=image_columns)
            with tempfile.TemporaryDirectory(prefix="bufo-demo-images-") as temp:
                for prompt in report["prompts"]:
                    values = [prompt["id"], prompt["prompt"]]
                    for arm in report["arms"]:
                        row = next(r for r in arm["rows"] if r["prompt_id"] == prompt["id"])
                        png = None
                        if row["image_data_url"]:
                            path = Path(temp) / (
                                hashlib.sha256(row["id"].encode()).hexdigest() + ".png"
                            )
                            path.write_bytes(
                                base64.b64decode(row["image_data_url"].split(",", 1)[1])
                            )
                            png = wb.Image(str(path))
                        values += [
                            png,
                            row["strict_pass"],
                            row["error"] or (None if png else "Missing PNG"),
                            row["trace_url"],
                            row["saved_judge"].get("status"),
                        ]
                    image_table.add_data(*values)
                pairs = wb.Table(
                    columns=["left", "right", "difference", "ci95_low", "ci95_high", "families"],
                    data=[
                        [
                            p["left"],
                            p["right"],
                            p["interval"]["difference"],
                            *p["interval"]["ci95"],
                            p["interval"]["families"],
                        ]
                        for p in report["paired_intervals"]
                    ],
                )
                run.log(
                    {
                        "comparison_summary": summary,
                        "matched_images": image_table,
                        "paired_uncertainty": pairs,
                        "matched_strict_pass_rate": wb.plot.bar(
                            chart,
                            "arm_id",
                            "strict_pass_rate_all_attempts",
                            title="Strict passes / all attempts · matched runtime only",
                        ),
                    }
                )
                for arm in report["arms"]:
                    for key, value in arm["summary"].items():
                        run.summary[f"arms/{arm['id']}/{key}"] = value
                artifact = wb.Artifact(
                    "bufo-demo-validation-" + report["evidence_id"][:12], type="evaluation"
                )
                for name in files:
                    artifact.add_file(str(output / name), name=name)
                artifact.add_file(str(output / "files.json"), name="files.json")
                logged = run.log_artifact(artifact)
                logged.wait()
                receipt.update(run_url=run.url, artifact_ref=f"{destination}/{logged.name}")
                run.finish()
                run = None
            receipt.update(status="published", error=None)
            _write(receipt_path, receipt)
            return receipt
        except Exception as exc:
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _write(receipt_path, receipt)
            if run is not None:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/forge-demo-comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/forge-demo/comparison")
    parser.add_argument(
        "--publish", action="store_true", help="Publish saved evidence; never run inference"
    )
    parser.add_argument("--entity")
    parser.add_argument("--project")
    args = parser.parse_args()
    if args.publish and (not args.entity or not args.project):
        parser.error("--publish requires explicit --entity and --project")
    report = build_comparison(args.manifest, args.output)
    print(
        json.dumps(
            {
                "evidence_id": report["evidence_id"],
                "html": str(args.output / "index.html"),
                "arms": {a["id"]: a["summary"] for a in report["arms"]},
            },
            indent=2,
        )
    )
    if args.publish:
        print(
            json.dumps(
                publish_comparison(args.output, entity=args.entity, project=args.project), indent=2
            )
        )


if __name__ == "__main__":
    main()
