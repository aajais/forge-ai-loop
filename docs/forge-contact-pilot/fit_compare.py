"""Local, fixed-checkpoint 24-prompt fit comparison. No model or service calls.

Default validates completed generation evidence. --packets writes neutral mixed
review packets. --summary requires all 48 scores/reviews and freezes the fit gate.
"""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont
from bufogen.evaluation import benchmark
from bufogen.generation import BufoGeneration
from bufogen.settings import ROOT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import require_checkpoint

BASE = "google/gemma-4-26B-A4B-it"
MODEL_NAME = "bufo-gemma-contact-v2-fit-20261020"
RECIPE = {"epochs": 6, "batch_size": 2, "learning_rate": 1e-4, "assistant_turns": "all"}
ARMS = {"baseline": ("fit-baseline", "open_base"), "candidate": ("fit-candidate", "open_tuned")}
ROOT_DEFAULT = ROOT / "runs/forge-contact-pilot/learnability"
GATE = {"minimum_combined_passes": 20, "denominator": 24,
        "rule": "completed valid generation AND automated pass AND assistant pass on the same attempt"}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or invalid completed evidence: {path}") from exc


def jsonl(path):
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or invalid completed JSONL: {path}") from exc


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze_all(files):
    for path, data in files.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Frozen comparison evidence changed: {path}")
    for path, data in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)


def candidate_identity(root):
    """Bind the single predeclared candidate, not a caller-selected epoch/alias."""
    training = root / "training"
    spec_path, state_path = training / "candidate/candidate.json", training / "candidate/forge-state.json"
    spec, state = read(spec_path), read(state_path)
    receipt_path = training / "source-receipt.json"
    receipt = read(receipt_path)
    source, upload = training / "reviewed-examples.jsonl", training / "upload-entries.json"
    if (
        spec.get("candidate") != "contact-v2-fit" or spec.get("base_model") != BASE
        or spec.get("model_name") != MODEL_NAME or spec.get("display_model_id") != f"ft-{BASE}"
        or spec.get("initialization") != "independent_base_model" or spec.get("training_config") != RECIPE
        or spec.get("source_receipt_sha256") != sha(receipt_path)
        or spec.get("source_path") != str(source)
        or spec.get("source_sha256") != sha(source) or spec.get("upload_sha256") != sha(upload)
        or receipt.get("source_sha256") != spec["source_sha256"]
        or receipt.get("upload_sha256") != spec["upload_sha256"]
        or receipt.get("contract_sha256") != spec.get("contract_sha256")
        or receipt.get("entry_counts") != {"train": 24, "val": 0}
        or receipt.get("training_request_policy") != RECIPE
    ):
        raise ValueError("Candidate differs from the predeclared 24-target independent six-epoch recipe")
    expected = {key: spec[key] for key in (
        "base_model", "model_name", "dataset_name", "source_path", "source_sha256", "upload_sha256",
    )} | spec["project"] | {
        "training_config": RECIPE, "heldout_ids": receipt["heldout_ids"],
        "heldout_families": receipt["heldout_families"],
    }
    if (
        any(state.get(key) != value for key, value in expected.items())
        or state.get("pending_operation") or state.get("status") != "deployed"
        or not state.get("finetune_id") or not state.get("entries_imported")
        or state.get("entry_counts") != {"train": 24, "val": 0}
    ):
        raise ValueError("Candidate workflow is incomplete, uncertain, or differs from its frozen recipe")
    checkpoint = require_checkpoint(state.get("model_artifact_ref"))
    files = {str(path): sha(path) for path in (spec_path, receipt_path, source, upload)}
    for item in receipt["source_files"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Predeclared training source changed")
        files[item["path"]] = item["sha256"]
    # Poll timestamps/progress may change; the observed immutable model identity must not.
    identity = {"checkpoint": checkpoint, "finetune_id": state["finetune_id"],
                "model_name": MODEL_NAME, "recipe": RECIPE, "source_sha256": spec["source_sha256"],
                "upload_sha256": spec["upload_sha256"], "contract_sha256": spec["contract_sha256"]}
    return identity, receipt, files


def current_configurations(root, checkpoint):
    import experiment
    result = {}
    fixtures = None
    for name, (_, arm) in ARMS.items():
        _, _, result[name], accepted = experiment.prepare(
            root / "train-prompts.json", [arm], checkpoint if name == "candidate" else None,
        )
        if fixtures is not None and fixtures != accepted:
            raise ValueError("Fixture acceptance changed while preparing comparison")
        fixtures = accepted
    return result, fixtures


def verify_arm_conditions(configurations):
    left, right = (configurations[name] for name in ARMS)
    if {k: v for k, v in left.items() if k != "arms"} != {k: v for k, v in right.items() if k != "arms"}:
        raise ValueError("Fit arms have different prompts, source hashes, instructions, budgets or scorer")
    changing = {"name", "label", "model", "model_env", "base_model", "tuning_method"}
    if any(len(config["arms"]) != 1 for config in configurations.values()):
        raise ValueError("Each fit source must contain exactly one model arm")
    arms = [config["arms"][0] for config in (left, right)]
    if {k: v for k, v in arms[0].items() if k not in changing} != {
        k: v for k, v in arms[1].items() if k not in changing
    }:
        raise ValueError("Fit arms differ in serving/sampling settings")
    if (arms[0]["model"] != BASE or arms[0].get("base_model") != BASE or arms[0].get("tuning_method")
            or any(arm.get("provider") != "coreweave" for arm in arms)
            or arms[1].get("base_model") != BASE or arms[1].get("tuning_method") != "sft"):
        raise ValueError("Expected unchanged Gemma base versus its single SFT checkpoint")
    require_checkpoint(arms[1]["model"])


def validate(root):
    root = Path(root).resolve()
    identity, receipt, hashes = candidate_identity(root)
    prompts = benchmark.load_benchmark(root / "train-prompts.json")
    if (
        len(prompts) != 24 or len({p["id"] for p in prompts}) != 24
        or any(p["split"] != "train" for p in prompts)
        or len(set(receipt["heldout_ids"])) != 56 or len(set(receipt["heldout_families"])) != 20
        or {p["id"] for p in prompts} & set(receipt["heldout_ids"])
        or {p["family"] for p in prompts} & set(receipt["heldout_families"])
    ):
        raise ValueError("Require exactly 24 training prompts excluding 8 diagnostics and 48 confirmation IDs")
    configurations, fixtures = current_configurations(root, identity["checkpoint"])
    verify_arm_conditions(configurations)
    rows, settings_by_arm = [], {}
    for name, (source_name, arm) in ARMS.items():
        source = root / source_name
        settings, protocol = read(source / "benchmark.json"), read(source / "experiment-protocol.json")
        config = configurations[name]
        expected_hashes = {
            "fixture-manifest.json": fixtures["manifest_sha256"],
            "fixture-assistant-review.json": fixtures["review_sha256"],
            "fixture-verifier.py": config["source_hashes"][str((HERE / "evaluate.py").relative_to(ROOT))],
        }
        if (
            settings["configuration"] != config or settings["configuration_hash"] != benchmark.digest(config)
            or settings["comparison_fingerprint"] != benchmark.comparison_fingerprint(config)
            or config["prompts"] != prompts or config["arms"][0]["name"] != arm
            or not config["prompt_version"].startswith("bufo-contact-v2-")
            or protocol != {"schema_version": 1, "configuration": config,
                "configuration_hash": benchmark.digest(config), "fixture_evidence": fixtures,
                "evidence_sha256": expected_hashes}
            or (name == "candidate" and config["arms"][0]["model"] != identity["checkpoint"])
        ):
            raise ValueError("Fit benchmark/protocol differs from the fixed current v2 harness")
        for relative, digest in config["source_hashes"].items():
            for path in (ROOT / relative, source / "frozen" / relative):
                if sha(path) != digest:
                    raise ValueError("Fit source bytes changed")
                hashes[str(path)] = digest
        for filename, digest in expected_hashes.items():
            path = source / "protocol-evidence" / filename
            if sha(path) != digest:
                raise ValueError("Fit fixture acceptance bytes changed")
            hashes[str(path)] = digest
        values = jsonl(source / "manifest.jsonl")
        if (len(values) != 24 or len({row["id"] for row in values}) != 24
                or {(row["prompt_id"], row["backend"]) for row in values} != {(p["id"], arm) for p in prompts}):
            raise ValueError("Require every original fit attempt, including failures; no omission or duplicates")
        benchmark._validate_rows([BufoGeneration(**row) for row in values], settings)
        for row in values:
            if row["benchmark_metadata"].get("status") not in {"completed", "generation_error"}:
                raise ValueError("Every fit attempt must be terminal")
            if row.get("png_path"):
                if sha(row["png_path"]) != row["benchmark_metadata"].get("png_sha256"):
                    raise ValueError("PNG differs from its original captured hash")
                with Image.open(row["png_path"]) as image:
                    if image.format != "PNG" or image.size != (512, 512) or image.info:
                        raise ValueError("Review needs original 512px PNGs without identifying metadata")
                    image.verify()
                hashes[row["png_path"]] = sha(row["png_path"])
            rows.append({"arm": name, "generation": row})
        settings_by_arm[name] = settings
        for filename in ("benchmark.json", "experiment-protocol.json", "manifest.jsonl"):
            path = source / filename
            hashes[str(path)] = sha(path)
    if len({item["generation"]["id"] for item in rows}) != 48:
        raise ValueError("Mixed comparison requires globally unique generation IDs")
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    hashes[str(root / "train-prompts.json")] = sha(root / "train-prompts.json")
    return {"root": root, "rows": rows, "prompts": prompts, "settings": settings_by_arm,
            "candidate_identity": identity, "generation_evidence_sha256": hashes}


def indexed(data):
    return sorted(data["rows"], key=lambda item: hashlib.sha256(
        ("contact-fit-review-v1:" + item["generation"]["id"]).encode()
    ).hexdigest())


def packet_values(data):
    folder = data["root"] / "fit-comparison/visual-review"
    items, key, images = [], [], {}
    for index, entry in enumerate(indexed(data)):
        row = entry["generation"]
        copied = folder / "images" / f"image-{index + 1:02d}.png" if row.get("png_path") else None
        image_hash = sha(row["png_path"]) if copied else None
        if copied:
            images[copied] = Path(row["png_path"]).read_bytes()
        items.append({"index": index, "label": f"Image {index + 1:02d}", "prompt": row["prompt"],
                      "requirements": row["benchmark_metadata"]["requirements"],
                      "png_path": str(copied) if copied else None, "png_sha256": image_hash})
        key.append({"index": index, "arm": entry["arm"],
                    **{field: row[field] for field in ("id", "prompt_id", "backend", "model")}})
    protocol = {"version": "contact-fit-neutral-review-v1", "attempts": 48, "reviewers": 3,
                "candidate_identity": data["candidate_identity"], "gate": GATE,
                "generation_evidence_sha256": data["generation_evidence_sha256"],
                "blinding": "Reviewer inputs contain neutral image paths, prompts and requirements only."}
    return items, key, images, protocol


def ensure_generation_hashes(data):
    if any(sha(path) != digest for path, digest in data["generation_evidence_sha256"].items()):
        raise ValueError("Validated generation or training evidence changed before freezing")


def create_packets(data):
    ensure_generation_hashes(data)
    folder = data["root"] / "fit-comparison/visual-review"
    items, key, images, protocol = packet_values(data)
    outputs = images | {folder / "unblinding-key.json": encoded(key), folder / "protocol.json": encoded(protocol)}
    font = ImageFont.truetype(str(Path(__file__).resolve().parents[2] / "assets/fonts/DejaVuSans.ttf"), 16)
    for reviewer in range(3):
        packet = folder / f"reviewer-{reviewer}"
        batch = items[reviewer::3]
        outputs[packet / "inputs.json"] = encoded(batch)
        for offset in range(0, 16, 4):
            sheet = Image.new("RGB", (1080, 1800), "#eef0ed")
            draw = ImageDraw.Draw(sheet)
            for n, item in enumerate(batch[offset:offset + 4]):
                x, y = n % 2 * 540, n // 2 * 900
                draw.text((x + 10, y + 5), item["label"], font=font, fill="black")
                if item["png_path"]:
                    with Image.open(io.BytesIO(images[Path(item["png_path"])])) as source:
                        canvas = Image.new("RGBA", (512, 512), "white")
                        canvas.alpha_composite(source.convert("RGBA"))
                        sheet.paste(canvas.convert("RGB"), (x + 10, y + 30))
                else:
                    draw.text((x + 10, y + 180), "No image generated", font=font, fill="black")
                lines = textwrap.wrap(item["prompt"], width=62)
                for requirement in item["requirements"]:
                    lines.extend(textwrap.wrap("- " + requirement, width=62))
                if len(lines) > 20:
                    raise ValueError("Review text does not fit sheet; no truncated requirements permitted")
                draw.multiline_text((x + 10, y + 550), "\n".join(lines), font=font, fill="black", spacing=2)
            image_bytes = io.BytesIO()
            sheet.save(image_bytes, format="PNG")
            outputs[packet / f"sheet-{offset // 4 + 1:02d}.png"] = image_bytes.getvalue()
    hashes = {str(path.relative_to(folder)): hashlib.sha256(content).hexdigest()
              for path, content in outputs.items()}
    outputs[folder / "packet-manifest.json"] = encoded(hashes)
    ensure_generation_hashes(data)
    freeze_all(outputs)
    return {"attempts": 48, "reviewers": 3, "per_reviewer": 16, "sheets_per_reviewer": 4,
            "directory": str(folder), "network_calls": 0}


def completed_reviews(data):
    folder = data["root"] / "fit-comparison/visual-review"
    items, key, images, protocol = packet_values(data)
    if read(folder / "protocol.json") != protocol or read(folder / "unblinding-key.json") != key:
        raise ValueError("Review protocol/key differ from the fixed mixed generation evidence")
    hashes = read(folder / "packet-manifest.json")
    expected_names = {"protocol.json", "unblinding-key.json"} | {
        str(path.relative_to(folder)) for path in images
    } | {f"reviewer-{r}/inputs.json" for r in range(3)} | {
        f"reviewer-{r}/sheet-{s:02d}.png" for r in range(3) for s in range(1, 5)
    }
    if set(hashes) != expected_names or any(sha(folder / name) != digest for name, digest in hashes.items()):
        raise ValueError("Frozen packet image/input/sheet evidence changed or is incomplete")
    for path, content in images.items():
        if path.read_bytes() != content:
            raise ValueError("Neutral PNG differs from original captured bytes")
    reviews, evidence = {}, {str(folder / name): digest for name, digest in hashes.items()}
    evidence[str(folder / "packet-manifest.json")] = sha(folder / "packet-manifest.json")
    ordered = indexed(data)
    for reviewer in range(3):
        packet = folder / f"reviewer-{reviewer}"
        expected = items[reviewer::3]
        if read(packet / "inputs.json") != expected:
            raise ValueError("Reviewer inputs expose or alter neutral image bindings")
        decisions = jsonl(packet / "decisions.jsonl")
        by_index = {item["index"]: item for item in expected}
        if (len(decisions) != 16 or any(type(d.get("index")) is not int for d in decisions)
                or {d["index"] for d in decisions} != set(by_index)):
            raise ValueError("Require all 16 distinct decisions per reviewer, including failures")
        for decision in decisions:
            item = by_index[decision["index"]]
            failed = decision.get("failed_requirements")
            if (
                decision.get("label") != item["label"] or decision.get("png_sha256") != item["png_sha256"]
                or decision.get("reviewer_kind") != "assistant" or type(decision.get("passed")) is not bool
                or not isinstance(decision.get("evidence"), str) or not decision["evidence"].strip()
                or not isinstance(failed, list) or any(not isinstance(v, str) for v in failed)
                or len(set(failed)) != len(failed) or not set(failed) <= set(item["requirements"])
                or (decision["passed"] and (failed or not item["png_path"]))
                or (not item["png_path"] and failed != item["requirements"])
            ):
                raise ValueError("Invalid assistant decision, PNG binding or atomic failed requirements")
            reviews[ordered[decision["index"]]["generation"]["id"]] = decision
        notes = packet / "review-notes.md"
        if not notes.is_file() or not notes.read_text().strip():
            raise ValueError("Each reviewer must record review procedure and limitations")
        for path in (packet / "decisions.jsonl", notes):
            evidence[str(path)] = sha(path)
    return reviews, evidence


def completed_scores(data):
    from bufogen.evaluation.scorers.scene_quality import scene_result
    scores, evidence = {}, {}
    for arm, (directory, _) in ARMS.items():
        folder = data["root"] / directory
        settings = data["settings"][arm]
        originals = {item["generation"]["id"]: item["generation"] for item in data["rows"] if item["arm"] == arm}
        rows = jsonl(folder / "scored.jsonl")
        if len(rows) != 24 or {row["id"] for row in rows} != set(originals):
            raise ValueError("Require exactly 24 terminal scored attempts per arm")
        for row in rows:
            if {k: v for k, v in row.items() if k != "scores"} != {
                k: v for k, v in originals[row["id"]].items() if k != "scores"
            }:
                raise ValueError("Scoring changed generation evidence")
            score = row.get("scores", {}).get("scene_quality", {})
            if (
                score.get("generation_id") != row["id"]
                or score.get("scorer_configuration") != settings["configuration"]["scorer"]
                or score.get("requirements") != row["benchmark_metadata"]["requirements"]
                or score.get("status") not in {"ok", "judge_error", "missing_image", "prior_turn_failure"}
            ):
                raise ValueError("Missing or mismatched terminal score")
            if score["status"] == "ok":
                normalized = scene_result(score, score["requirements"],
                                          threshold=settings["configuration"]["scorer"]["threshold"])
                if any(score.get(k) != v for k, v in normalized.items()):
                    raise ValueError("Inconsistent score/checklist arithmetic")
            elif score.get("quality_failure") is False:
                raise ValueError("An unsuccessful score cannot count as an automated pass")
            scores[row["id"]] = score
        if benchmark._summary([BufoGeneration(**row) for row in rows], settings) != read(folder / "summary.json"):
            raise ValueError("Scored summary differs from the complete attempted denominator")
        for path in (folder / "scored.jsonl", folder / "summary.json"):
            evidence[str(path)] = sha(path)
    return scores, evidence


def summarize(data):
    ensure_generation_hashes(data)
    reviews, review_hashes = completed_reviews(data)
    scores, score_hashes = completed_scores(data)
    groups, disagreements = {}, []
    for arm in ARMS:
        attempts = [item["generation"] for item in data["rows"] if item["arm"] == arm]
        counts = CounterDict()
        for row in attempts:
            score, review = scores[row["id"]], reviews[row["id"]]
            complete = row["valid"] and row["benchmark_metadata"]["status"] == "completed"
            raw_auto, raw_review = score.get("quality_failure") is False, review["passed"]
            auto, human = bool(complete and raw_auto), bool(complete and raw_review)
            counts.add("automated_passes", auto)
            counts.add("assistant_image_passes", raw_review)
            counts.add("assistant_completed_passes", human)
            counts.add("combined_passes", auto and human)
            counts.add("generation_errors", row["benchmark_metadata"]["status"] != "completed")
            counts.add("judge_errors", score["status"] == "judge_error")
            if raw_auto != raw_review:
                disagreements.append({
                    "arm": arm, "generation_id": row["id"], "prompt_id": row["prompt_id"],
                    "automated_image_pass": raw_auto, "assistant_image_pass": raw_review,
                    "generation_completed_and_valid": bool(complete),
                    "score_status": score["status"], "automated_reason": score.get("reason", score.get("error")),
                    "assistant_evidence": review["evidence"], "failed_requirements": review["failed_requirements"],
                })
        groups[arm] = {
            "model_id": BASE if arm == "baseline" else data["candidate_identity"]["checkpoint"],
            "model_label": BASE if arm == "baseline" else f"ft-{BASE}", "attempts": 24, **counts,
            "automated_pass_rate_all_attempts": counts["automated_passes"] / 24,
            "assistant_pass_rate_all_attempts": counts["assistant_completed_passes"] / 24,
            "combined_pass_rate_all_attempts": counts["combined_passes"] / 24,
            "disagreements": sum(item["arm"] == arm for item in disagreements),
        }
    result = {
        "schema_version": 1, "purpose": "in-sample 24-target contact-v2 learnability fit",
        "attempts": 48, "candidate_identity": data["candidate_identity"], "models": groups,
        "gate": GATE | {"passed": groups["candidate"]["combined_passes"] >= 20},
        "disagreements": disagreements,
        "evidence_sha256": data["generation_evidence_sha256"] | review_hashes | score_hashes,
        "limitations": [
            "Training requests are in-sample; this is not generalization, Sol parity, or promotion evidence.",
            "One predeclared six-epoch candidate; no epoch selection or diagnostic-driven replacement.",
            "Assistant visual reviews are not human annotations; automated image scores are uncalibrated.",
            "Partial PNGs from failed agent turns retain image-only verdicts but cannot pass the fit gate.",
            "Eight diagnostic and 48 reserved confirmation prompts are excluded from this comparison.",
        ],
    }
    for path, digest in result["evidence_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Validated comparison evidence changed before freezing the summary")
    freeze_all({data["root"] / "fit-comparison/comparison.json": encoded(result)})
    return result


class CounterDict(dict):
    def add(self, key, condition):
        self[key] = self.get(key, 0) + int(bool(condition))


def run(root, mode="validate"):
    root = Path(root).resolve()
    if mode not in {"validate", "packets", "summary"}:
        raise ValueError("Unknown comparison mode")
    with ExitStack() as stack:
        for directory, _ in ARMS.values():
            if not (root / directory).is_dir():
                raise ValueError(f"Missing completed fit source: {directory}")
            stack.enter_context(benchmark._locked(root / directory))
        if mode != "validate":
            stack.enter_context(benchmark._locked(root / "fit-comparison"))
        data = validate(root)
        if mode == "packets":
            return create_packets(data)
        if mode == "summary":
            return summarize(data)
        return {"offline_validation": "passed", "network_calls": 0, "attempts": 48, "gate": GATE,
                "candidate_identity": data["candidate_identity"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--packets", action="store_true")
    modes.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.root, "packets" if args.packets else "summary" if args.summary else "validate"), indent=2))


if __name__ == "__main__":
    main()
