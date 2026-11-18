"""Publish the completed four-way interface pilot; default is offline validation."""

import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import statistics
import uuid

from bufogen.evaluation import benchmark
from bufogen.generation import BufoGeneration
from bufogen.settings import ROOT
from bufogen.settings import project_path

HERE = Path(__file__).resolve().parent
ARMS = ("open_base", "closed_quality_bar")


def evaluator():
    spec = importlib.util.spec_from_file_location("contact_pilot_publication_evaluator", HERE / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_reviews(directory, evidence, *, sha, read):
    """Join blinded decisions by exact index; do not infer reviewer verdicts from scores."""
    folder = directory / "blind-review"
    key_path = folder / "unblinding-key.json"
    input_paths = [folder / f"reviewer-{i}/inputs.json" for i in range(3)]
    decision_paths = [folder / f"reviewer-{i}/decisions.jsonl" for i in range(3)]
    missing = [str(path.relative_to(directory)) for path in [key_path, *input_paths] if not path.exists()]
    if missing:
        return {}, {}, {"status": "waiting", "reviewed_attempts": 0, "required_attempts": 64, "missing_files": missing}
    actual = {(interface, row["id"]): row for interface, rows in evidence.items() for row in rows}
    key = read(key_path)
    if (
        len(actual) != 64 or len({row["id"] for row in actual.values()}) != 64 or len(key) != 64
        or any(type(item.get("index")) is not int for item in key)
        or {item["index"] for item in key} != set(range(64))
        or {(item["interface"], item["id"]) for item in key} != set(actual)
    ):
        raise ValueError("Blind-review key must uniquely map all 64 attempts")
    indexed = {}
    for item in key:
        row = actual[item["interface"], item["id"]]
        if any(item.get(field) != row[field] for field in ("id", "prompt_id", "backend", "model")):
            raise ValueError("Unblinding key differs from actual generation identity")
        indexed[item["index"]] = row
    reviews, hashes = {}, {str(key_path.relative_to(directory)): sha(key_path)}
    outstanding = []
    for reviewer, (input_path, decision_path) in enumerate(zip(input_paths, decision_paths, strict=True)):
        inputs = read(input_path)
        expected_indices = {i for i in range(64) if i % 3 == reviewer}
        if (
            len(inputs) != len(expected_indices)
            or any(type(item.get("index")) is not int for item in inputs)
            or {item["index"] for item in inputs} != expected_indices
        ):
            raise ValueError("Reviewer packet has missing/duplicate/unassigned image indices")
        by_index = {item["index"]: item for item in inputs}
        for index, item in by_index.items():
            row = indexed[index]
            expected_path = row.get("png_path")
            expected_hash = sha(expected_path) if expected_path else None
            if (
                item.get("label") != f"Image {index + 1:02d}"
                or item.get("prompt") != row["prompt"]
                or item.get("requirements") != row["benchmark_metadata"]["requirements"]
                or item.get("png_path") != expected_path
                or item.get("png_sha256") != expected_hash
            ):
                raise ValueError("Blind packet label/prompt/checklist/PNG differs from generation")
        hashes[str(input_path.relative_to(directory))] = sha(input_path)
        if not decision_path.exists():
            outstanding.append(str(decision_path.relative_to(directory)))
            continue
        decisions = jsonl(decision_path)
        indices = [decision.get("index") for decision in decisions]
        if (
            any(type(index) is not int for index in indices)
            or len(set(indices)) != len(indices)
            or not set(indices) <= expected_indices
        ):
            raise ValueError("Duplicate or unassigned blind review decision")
        for decision in decisions:
            index = decision["index"]
            item, row = by_index[index], indexed[index]
            failed = decision.get("failed_requirements")
            if (
                decision.get("label") != item["label"]
                or decision.get("png_sha256") != item["png_sha256"]
                or decision.get("reviewer_kind") != "assistant"
                or type(decision.get("passed")) is not bool
                or not isinstance(decision.get("evidence"), str)
                or not decision["evidence"].strip()
                or not isinstance(failed, list)
                or any(not isinstance(value, str) for value in failed)
                or len(set(failed)) != len(failed)
                or not set(failed) <= set(item["requirements"])
                or (decision["passed"] and (failed or not item["png_path"]))
            ):
                raise ValueError("Invalid blind review verdict, PNG hash, evidence, or requirements")
            reviews[row["id"]] = decision | {"reviewer_packet": f"reviewer-{reviewer}"}
        hashes[str(decision_path.relative_to(directory))] = sha(decision_path)
        if set(indices) != expected_indices:
            outstanding.append(f"{decision_path.relative_to(directory)}: {len(expected_indices - set(indices))} decisions missing")
        # Retain the actual blinded contact sheets as inspection evidence, never as model outputs.
        for sheet in sorted(input_path.parent.glob("sheet-*.png")):
            hashes[str(sheet.relative_to(directory))] = sha(sheet)
    return reviews, hashes, {
        "status": "complete" if len(reviews) == 64 and not outstanding else "waiting",
        "reviewed_attempts": len(reviews), "required_attempts": 64, "missing_files": outstanding,
        "blinding": "model_blinded_interface_paths_visible",
        "limitation": "Original PNG paths disclosed the interface; this was not a fully interface-blinded review.",
    }


def validate(root):
    e = evaluator()
    prompts, _, expected_protocol = e.prepare(root)
    directory = root / "comparison"
    protocol = e.read(directory / "pilot-protocol.json")
    if protocol != expected_protocol:
        raise ValueError("Pilot protocol, fixture acceptance, or frozen code/configuration changed")
    evidence, summaries, hashes = {}, {}, {}
    for interface in e.INTERFACES:
        folder = directory / interface
        settings = e.read(folder / "benchmark.json")
        config = settings["configuration"]
        if (
            config != protocol["configurations"][interface]
            or settings["configuration_hash"] != benchmark.digest(config)
            or settings["comparison_fingerprint"] != benchmark.comparison_fingerprint(config)
            or {arm["name"] for arm in config["arms"]} != set(ARMS)
        ):
            raise ValueError(f"{interface}: frozen configuration/hash differs")
        expected_pairs = {(p["id"], arm) for p in prompts for arm in ARMS}
        manifest, scored = jsonl(folder / "manifest.jsonl"), jsonl(folder / "scored.jsonl")
        for rows in (manifest, scored):
            if (
                len(rows) != 32
                or len({r["id"] for r in rows}) != 32
                or {(r["prompt_id"], r["backend"]) for r in rows} != expected_pairs
            ):
                raise ValueError(f"{interface}: require all 32 unique attempts, including failures")
            benchmark._validate_rows([BufoGeneration(**r) for r in rows], settings)
        originals = {r["id"]: r for r in manifest}
        for row in scored:
            original = originals.get(row["id"])
            if original is None or any(
                row.get(key) != value for key, value in original.items() if key != "scores"
            ):
                raise ValueError(f"{interface}: scoring changed the generation evidence")
            if row.get("png_path") and (
                not row["benchmark_metadata"].get("png_sha256")
                or e.sha(row["png_path"]) != row["benchmark_metadata"]["png_sha256"]
            ):
                raise ValueError(f"{interface}: candidate PNG hash missing or changed")
            if row.get("svg_path") and Path(row["svg_path"]).read_text() != row["svg"]:
                raise ValueError(f"{interface}: persisted SVG differs from the actual generation")
            result = row.get("scores", {}).get("scene_quality", {})
            if (
                result.get("scorer_configuration") != config["scorer"]
                or result.get("generation_id") != row["id"]
                or result.get("requirements") != row["benchmark_metadata"]["requirements"]
                or result.get("status") not in {"ok", "judge_error", "missing_image", "prior_turn_failure"}
            ):
                raise ValueError(f"{interface}: missing/mismatched terminal score")
            if result["status"] == "ok":
                from bufogen.evaluation.scorers.scene_quality import scene_result
                normalized = scene_result(result, result["requirements"], threshold=result["threshold"])
                if any(result.get(key) != value for key, value in normalized.items()):
                    raise ValueError(f"{interface}: inconsistent score/checklist arithmetic")
            if row.get("trajectory"):
                trajectory = row["trajectory"]
                tool_names = {tool["function"]["name"] for tool in trajectory["tools"]}
                required = "render_scene" if interface == "contact" else "render_svg"
                if (
                    required not in tool_names
                    or trajectory["provenance"]["prompt_version"] != config["prompt_version"]
                    or row["prompt_version"] != config["prompt_version"]
                    or trajectory["provenance"]["model"] != row["model"]
                ):
                    raise ValueError(f"{interface}: captured tool/harness/model identity differs")
                provenance = trajectory["provenance"]
                if interface == "contact" and provenance.get("scene_path") and (
                    e.sha(provenance["scene_path"]) != provenance.get("scene_sha256")
                    or e.read(provenance["scene_path"]) != provenance.get("scene")
                ):
                    raise ValueError("Contact scene JSON changed since the captured tool call")
        values = [BufoGeneration(**r) for r in scored]
        summary = benchmark._summary(values, settings)
        if summary != e.read(folder / "summary.json"):
            raise ValueError(f"{interface}: summary excludes or changes attempted generations")
        publication = e.read(folder / "weave-publication.json")
        if (
            publication.get("errors")
            or set(publication.get("feedback", {})) != {r["id"] for r in scored}
            or any(not item.get("call_id") for item in publication["feedback"].values())
            or set(publication.get("evaluations", {})) != set(ARMS)
            or any(item.get("scored_rows") != 16 or not item.get("ref")
                   for item in publication["evaluations"].values())
        ):
            raise ValueError(f"{interface}: native Weave feedback/evaluation coverage incomplete")
        evidence[interface], summaries[interface] = scored, summary
        for name in ("benchmark.json", "manifest.jsonl", "scored.jsonl", "summary.json", "weave-publication.json"):
            hashes[f"{interface}/{name}"] = e.sha(folder / name)
    expected_comparison = {
        "purpose": protocol["purpose"], "protocol_sha256": e.sha(directory / "pilot-protocol.json"),
        "summaries": summaries,
        "note": "All attempts counted. Same image scorer; assistant visual review also required.",
    }
    if e.read(directory / "comparison.json") != expected_comparison:
        raise ValueError("Combined comparison differs from the complete interface evidence")
    for name in ("pilot-protocol.json", "comparison.json"):
        hashes[name] = e.sha(directory / name)
    hashes["publisher_source"] = e.sha(__file__)
    hashes["review_packet_source"] = e.sha(HERE / "review_packets.py")
    reviews, review_hashes, review_status = validate_reviews(directory, evidence, sha=e.sha, read=e.read)
    hashes.update(review_hashes)
    return e, prompts, evidence, summaries, protocol, hashes, reviews, review_status


def passed(row):
    return bool(row["valid"] and row["benchmark_metadata"]["status"] == "completed"
                and row["scores"]["scene_quality"].get("quality_failure") is False)


def assistant_passed(row, review):
    return bool(row["valid"] and row["benchmark_metadata"]["status"] == "completed" and review["passed"])


def publish(root, execute=False):
    directory = root / "comparison"
    # A publisher must not race a still-flushing native evaluation.
    with ExitStack() as locks:
        for name in ("svg", "contact"):
            if not (directory / name).is_dir():
                raise ValueError(f"Missing completed {name} interface")
            locks.enter_context(benchmark._locked(directory / name))
        e, prompts, evidence, summaries, protocol, hashes, reviews, review_status = validate(root)
    if not execute:
        return {
            "offline_validation": "passed" if review_status["status"] == "complete" else "waiting_for_blind_review",
            "network_calls": 0, "attempts": 64, "blind_review": review_status,
            "paired_rows": 16, "models": [e.BASE, e.TEACHER],
            "purpose": protocol["purpose"],
        }
    if review_status["status"] != "complete":
        raise ValueError("Publication requires all 64 valid blinded assistant reviews; no upload attempted")
    receipt_path = directory / "visualization.json"
    with benchmark._locked(directory):
        if receipt_path.exists():
            receipt = e.read(receipt_path)
            if receipt.get("evidence_sha256") != hashes:
                raise ValueError("Prior publication evidence differs")
            if receipt.get("status") == "published":
                return receipt
            raise RuntimeError("Publication already started; reconcile the saved run ID before retrying")
        receipt = {
            "schema_version": 1, "status": "publishing", "run_id": uuid.uuid4().hex[:8],
            "purpose": protocol["purpose"], "attempts": 64, "paired_rows": 16,
            "fine_tuning_gain_claim": False, "promotion": False, "evidence_sha256": hashes,
            "blind_review": review_status,
            "accounting_note": "Pass rates include every attempt. Tokens are recorded usage; failed requests may omit usage. Latency aggregates below cover completed attempts.",
        }
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    import wandb

    def png(row):
        return wandb.Image(row["png_path"]) if row.get("png_path") else None

    lookups = {name: {(r["prompt_id"], r["backend"]): r for r in rows}
               for name, rows in evidence.items()}
    pairs = wandb.Table(columns=[
        "prompt_id", "prompt", "requirements", "base_model_id", "teacher_model_id",
        "svg_base_png", "svg_teacher_png", "contact_base_png", "contact_teacher_png",
        "svg_base_checks", "svg_teacher_checks", "contact_base_checks", "contact_teacher_checks",
        "svg_base_status", "svg_teacher_status", "contact_base_status", "contact_teacher_status",
        "svg_base_provisional_pass", "svg_teacher_provisional_pass", "contact_base_provisional_pass", "contact_teacher_provisional_pass",
        "svg_base_assistant_pass", "svg_teacher_assistant_pass", "contact_base_assistant_pass", "contact_teacher_assistant_pass",
        "svg_base_assistant_evidence", "svg_teacher_assistant_evidence", "contact_base_assistant_evidence", "contact_teacher_assistant_evidence",
        "svg_base_trace", "svg_teacher_trace", "contact_base_trace", "contact_teacher_trace",
    ])
    order = [(interface, arm) for interface in e.INTERFACES for arm in ARMS]
    for prompt in prompts:
        rows = [lookups[interface][prompt["id"], arm] for interface, arm in order]
        pairs.add_data(
            prompt["id"], prompt["prompt"], prompt["requirements"], e.BASE, e.TEACHER,
            *[png(row) for row in rows],
            *[row["scores"]["scene_quality"]["prompt_checks"] for row in rows],
            *[row["benchmark_metadata"]["status"] for row in rows],
            *[passed(row) for row in rows],
            *[assistant_passed(row, reviews[row["id"]]) for row in rows],
            *[reviews[row["id"]]["evidence"] for row in rows],
            *[row.get("conversation_url") for row in rows],
        )
    attempts = wandb.Table(columns=[
        "interface", "model_id", "prompt_id", "generation_id", "png", "generation_status",
        "generation_error", "score_status", "provisional_pass", "constraint_coverage",
        "assistant_visual_pass", "assistant_pass", "assistant_evidence", "assistant_failed_requirements", "blind_label", "reviewer_packet",
        "requirement_checks", "recorded_input_tokens", "recorded_output_tokens",
        "recorded_latency_s", "tool_calls", "prompt_version", "trace",
    ])
    metrics = wandb.Table(columns=[
        "interface", "model_id", "attempts", "provisional_passes", "pass_rate_all_attempts",
        "assistant_visual_passes", "assistant_passes", "assistant_pass_rate_all_attempts",
        "requirement_coverage_scored_only", "generation_errors", "judge_errors",
        "completed_attempts", "completed_mean_latency_s", "completed_median_latency_s",
        "total_recorded_input_tokens", "total_recorded_output_tokens",
    ])
    for interface, arm in order:
        subset = [r for r in evidence[interface] if r["backend"] == arm]
        model = e.BASE if arm == "open_base" else e.TEACHER
        for row in subset:
            result = row["scores"]["scene_quality"]
            review = reviews[row["id"]]
            attempts.add_data(
                interface, row["model"], row["prompt_id"], row["id"], png(row),
                row["benchmark_metadata"]["status"], row.get("error"), result["status"],
                passed(row), result.get("constraint_coverage"),
                review["passed"], assistant_passed(row, review), review["evidence"],
                review["failed_requirements"], review["label"], review["reviewer_packet"],
                result["prompt_checks"],
                row["input_tokens"], row["output_tokens"], row["latency_s"],
                row["tool_calls"], row["prompt_version"], row.get("conversation_url"),
            )
        summary = summaries[interface]["arms"][arm]
        latencies = [r["latency_s"] for r in subset if r["benchmark_metadata"]["status"] == "completed"]
        metrics.add_data(
            interface, model, len(subset), summary["provisional_passes"],
            summary["provisional_pass_rate_all_attempts"],
            sum(reviews[row["id"]]["passed"] for row in subset),
            sum(assistant_passed(row, reviews[row["id"]]) for row in subset),
            sum(assistant_passed(row, reviews[row["id"]]) for row in subset) / len(subset),
            summary["metric_means_scored_only"]["constraint_coverage"],
            summary["generation_errors"], summary["judge_errors"], len(latencies),
            statistics.mean(latencies) if latencies else None,
            statistics.median(latencies) if latencies else None,
            sum(r["input_tokens"] for r in subset), sum(r["output_tokens"] for r in subset),
        )
    entity, project = project_path().split("/")
    with wandb.init(
        entity=entity, project=project, id=receipt["run_id"], resume="never",
        name=f"SVG vs contact interface: {e.BASE} / {e.TEACHER}",
        job_type="interface-comparison", config=receipt | {"models": [e.BASE, e.TEACHER]},
    ) as run:
        run.log({"four_way_paired_images": pairs, "all_64_attempts": attempts, "interface_summary": metrics})
        for interface, arm in order:
            summary = summaries[interface]["arms"][arm]
            run.summary[f"{interface}/{arm}/attempts"] = summary["total"]
            run.summary[f"{interface}/{arm}/pass_rate_all_attempts"] = summary["provisional_pass_rate_all_attempts"]
            subset = [row for row in evidence[interface] if row["backend"] == arm]
            run.summary[f"{interface}/{arm}/assistant_pass_rate_all_attempts"] = (
                sum(assistant_passed(row, reviews[row["id"]]) for row in subset) / len(subset)
            )
        artifact = wandb.Artifact("bufo-contact-interface-comparison", type="evaluation")
        for name in hashes:
            if name not in {"publisher_source", "review_packet_source"}:
                artifact.add_file(str(directory / name), name=name)
        artifact.add_file(__file__, name="code/publish.py")
        for name in ("evaluate.py", "contact_agent.py", "scene_renderer.py", "prompts.json"):
            artifact.add_file(str(HERE / name), name=f"code/{name}")
        artifact.add_file(str(HERE / "review_packets.py"), name="code/review_packets.py")
        artifact.add_file(str(root / "fixtures/manifest.json"), name="fixtures/manifest.json")
        artifact.add_file(str(root / "fixtures/assistant-review.json"), name="fixtures/assistant-review.json")
        saved = run.log_artifact(artifact)
        saved.wait()
        run_url, artifact_ref = run.url, f"{entity}/{project}/{saved.name}"
    receipt.update(status="published", run_url=run_url, artifact_ref=artifact_ref)
    benchmark._write_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/forge-contact-pilot")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args.root.resolve(), execute=args.publish), indent=2))


if __name__ == "__main__":
    main()
