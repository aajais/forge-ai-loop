"""Prepare 24 reviewed contact targets locally; --train runs one managed Forge SFT."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys
import tempfile
from uuid import uuid4

import certifi
import httpx

from bufogen.evaluation import benchmark
from bufogen.curation.curate import _validate_review
from bufogen.integrations.forge import ForgeClient
from bufogen.integrations.forge import STUDENT_MODEL
from bufogen.integrations.forge import reviewed_entries
from bufogen.workflows.forge_loop import _clear_or_block
from bufogen.workflows.forge_loop import _locked
from bufogen.workflows.forge_loop import import_reviewed_dataset
from bufogen.workflows.forge_loop import initialize_workflow
from bufogen.workflows.forge_loop import resume_training
from bufogen.workflows.forge_loop import submit_training
from bufogen.generation import Manifest
from bufogen.settings import ROOT
from bufogen.settings import required

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from training_contract import ContactTrainingContract

DIRECTORY = ROOT / "runs/forge-contact-pilot/learnability"
RESERVED_RECEIPT = ROOT / "runs/forge-iteration2/training/source-receipt.json"
TEACHER = "gpt-6.1-sol"
MODEL_NAME = "bufo-gemma-contact-v2-fit-20261020"
RECIPE = {"epochs": 6, "batch_size": 2, "learning_rate": 1e-4, "assistant_turns": "all"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def jsonl(rows):
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()


def read(path):
    if not Path(path).is_file():
        raise ValueError(f"Required completed evidence is missing: {path}")
    return json.loads(Path(path).read_text())


def freeze(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen training evidence changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def transport_for_cli(tls12):
    if not tls12:
        return None, {"mode": "default"}
    context = ssl.create_default_context(cafile=certifi.where())
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_2
    return httpx.HTTPTransport(verify=context), {
        "mode": "tls12", "certificate_verification": True, "hostname_check": True,
        "minimum_tls_version": "TLSv1.2", "maximum_tls_version": "TLSv1.2",
    }


def curriculum(root):
    """Inspect only reserved identities, never reserved prompts or model outputs."""
    rows = benchmark.load_benchmark(root / "curriculum.json")
    train = [row for row in rows if row["split"] == "train"]
    diagnostic = [row for row in rows if row["split"] == "heldout"]
    reserved = read(RESERVED_RECEIPT)
    reserved_ids, reserved_families = reserved["heldout_ids"], reserved["heldout_families"]
    if (
        len(rows) != 32 or len(train) != 24 or len(diagnostic) != 8
        or len({row["id"] for row in rows}) != 32
        or len({row["family"] for row in rows}) != 32
        or len(set(reserved_ids)) != 48 or len(set(reserved_families)) != 12
        or {row["id"] for row in rows} & set(reserved_ids)
        or {row["family"] for row in rows} & set(reserved_families)
        or sorted(Counter(row["relation_class"] for row in train).values()) != [6, 6, 6, 6]
        or sorted(Counter(row["relation_class"] for row in diagnostic).values()) != [2, 2, 2, 2]
        or read(root / "train-prompts.json") != train
        or read(root / "diagnostic-prompts.json") != diagnostic
    ):
        raise ValueError("Require disjoint frozen 24 train / 8 diagnostic prompts and 48 reserved IDs")
    return train, sorted(set(reserved_ids) | {r["id"] for r in diagnostic}), sorted(
        set(reserved_families) | {r["family"] for r in diagnostic}
    )


def current_configuration(root):
    import experiment

    _, _, configuration, fixtures = experiment.prepare(root / "train-prompts.json", ["closed_quality_bar"])
    return configuration, fixtures


def prepare(root=DIRECTORY):
    root = Path(root).resolve()
    directory, teacher = root / "training", root / "teacher"
    with _locked(directory / "prepare"), benchmark._locked(teacher):
        train, heldout_ids, heldout_families = curriculum(root)
        config, fixtures = current_configuration(root)
        settings_path, protocol_path = teacher / "benchmark.json", teacher / "experiment-protocol.json"
        settings, protocol = read(settings_path), read(protocol_path)
        if (
            settings["configuration"] != config or protocol["configuration"] != config
            or settings["configuration_hash"] != benchmark.digest(config)
            or protocol["configuration_hash"] != settings["configuration_hash"]
            or settings["comparison_fingerprint"] != benchmark.comparison_fingerprint(config)
            or protocol["fixture_evidence"] != fixtures
            or config["prompts"] != train
            or len(config["arms"]) != 1
            or config["arms"][0]["name"] != "closed_quality_bar"
            or config["arms"][0]["model"] != TEACHER
        ):
            raise ValueError("Teacher benchmark/protocol differs from current frozen v2 collection")
        source_files = [
            root / "curriculum.json", root / "train-prompts.json", root / "diagnostic-prompts.json",
            RESERVED_RECEIPT, settings_path, protocol_path, Path(__file__).resolve(),
            HERE / "training_contract.py",
        ]
        # Bind both live bytes and benchmark snapshots to the actual collection configuration.
        for relative, expected in config["source_hashes"].items():
            live, snapshot = ROOT / relative, teacher / "frozen" / relative
            if sha(live.read_bytes()) != expected or sha(snapshot.read_bytes()) != expected:
                raise ValueError(f"Collected source bytes changed: {relative}")
            source_files.extend([live, snapshot])
        expected_evidence = {
            "fixture-manifest.json": fixtures["manifest_sha256"],
            "fixture-assistant-review.json": fixtures["review_sha256"],
            "fixture-verifier.py": config["source_hashes"][str((HERE / "evaluate.py").relative_to(ROOT))],
        }
        if protocol.get("evidence_sha256") != expected_evidence:
            raise ValueError("Fixture protocol evidence differs from collection")
        for name, expected in expected_evidence.items():
            path = teacher / "protocol-evidence" / name
            if sha(path.read_bytes()) != expected:
                raise ValueError("Frozen fixture acceptance evidence changed")
            source_files.append(path)
        manifest_path = teacher / "manifest.jsonl"
        if not manifest_path.is_file():
            raise ValueError("Wait for all 24 initial teacher attempts before preparation")
        rows = sorted(Manifest(manifest_path).read(), key=lambda row: row.prompt_id)
        expected = {row["id"]: row for row in train}
        if (len(rows) != 24 or len({row.id for row in rows}) != 24
                or {row.prompt_id for row in rows} != set(expected)):
            raise ValueError("Require all 24 initial teacher attempts, with no duplicates or replacements")
        benchmark._validate_rows(rows, settings)
        review_path = root / "reviews/decisions.jsonl"
        if not review_path.is_file():
            raise ValueError("Wait for all 24 explicit PNG-bound assistant reviews")
        reviews = [json.loads(line) for line in review_path.read_text().splitlines() if line.strip()]
        by_id = {row.id: row for row in rows}
        if (len(reviews) != 24 or {r.get("source_generation_id") for r in reviews} != set(by_id)):
            raise ValueError("Require one explicit review per initial teacher attempt")
        reviews.sort(key=lambda review: review["prompt_id"])
        rejected = []
        for review in reviews:
            row = by_id[review["source_generation_id"]]
            _validate_review(review, row)
            prompt = expected[row.prompt_id]
            if (
                review["reviewer_kind"] != "assistant" or review["partition"] != "train"
                or review["prompt_family"] != prompt["family"]
                or review.get("visual_quality_pass") is not (review["decision"] == "accept")
            ):
                raise ValueError("Review identity, partition, family or visual verdict is inconsistent")
            if review["decision"] != "accept":
                rejected.append(row.prompt_id)
        if rejected:
            raise ValueError(
                f"Preparation stopped: {len(rejected)} initial teacher targets rejected; "
                f"exactly 24 accepted targets required. No silent replacement: {rejected}"
            )
        first = rows[0].trajectory or {}
        specification = {
            "instructions": first.get("system_instructions"), "tools": first.get("tools"),
            "prompt_version": config["prompt_version"], "source_hashes": config["source_hashes"],
            "tool_contract": first.get("provenance", {}).get("tool_contract"),
        }
        if benchmark.digest(specification["instructions"]) != config["runtime"]["instructions_hash"]:
            raise ValueError("Actual capture instructions differ from frozen collection")
        contract = ContactTrainingContract(**specification)
        examples, artifacts = [], []
        with contract.activate():
            for review in reviews:
                row = by_id[review["source_generation_id"]]
                if row.benchmark_metadata.get("status") != "completed":
                    raise ValueError(f"Initial teacher target did not complete: {row.prompt_id}")
                try:
                    example = contract.export(row, review)
                except ValueError as exc:
                    raise ValueError(
                        f"Preparation stopped: target {row.prompt_id} is not a clean one-call capture: {exc}"
                    ) from exc
                examples.append(example)
                artifacts.append({
                    "generation_id": row.id, "prompt_id": row.prompt_id,
                    **{kind: {"path": str(Path(path).resolve()), "sha256": sha(Path(path).read_bytes())}
                       for kind, path in {
                           "png": row.png_path, "svg": row.svg_path,
                           "scene": row.trajectory["provenance"]["scene_path"],
                       }.items()},
                })
            entries = reviewed_entries(examples, heldout_ids=heldout_ids, heldout_families=heldout_families)
        if len(entries) != 24 or Counter(entry["split"] for entry in entries) != {"train": 24}:
            raise ValueError("Exactly 24 training entries and zero validation entries required")
        source_files.extend([manifest_path, review_path])
        source_files.extend(ROOT / name for name in contract.specification["training_source_hashes"])
        source_files = list(dict.fromkeys(path.resolve() for path in source_files))
        source_data = jsonl(examples)
        upload_data = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        receipt = {
            "schema_version": 1, "purpose": "contact_v2_24_target_learnability_fit",
            "source_files": [{"path": str(path), "sha256": sha(path.read_bytes())} for path in source_files],
            "artifacts": artifacts, "source_sha256": sha(source_data), "upload_sha256": sha(upload_data),
            "entry_counts": {"train": 24, "val": 0}, "initial_teacher_attempts": 24,
            "review_counts": {"accept": 24, "reject": 0},
            "teacher_model": TEACHER, "student_model": STUDENT_MODEL,
            "prompt_version": config["prompt_version"], "contract_sha256": contract.contract_sha256,
            "comparison_fingerprint": settings["comparison_fingerprint"],
            "heldout_ids": heldout_ids, "heldout_families": heldout_families,
            "training_request_policy": RECIPE,
            "nominal_optimizer_updates": 72,
            "optimizer_update_note": "24 examples / batch 2 * 6 epochs; nominal, not observed service updates",
        }
        outputs = {
            directory / "reviewed-examples.jsonl": source_data,
            directory / "upload-entries.json": upload_data,
            directory / "review-decisions.jsonl": jsonl(reviews),
            directory / "contact-contract.json": encoded(contract.specification),
            directory / "source-receipt.json": encoded(receipt),
        }
        for index, path in enumerate(source_files):
            outputs[directory / "sources" / f"{index:03d}-{path.name}"] = path.read_bytes()
        for path, data in outputs.items():
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"Refusing changed frozen training evidence: {path}")
        for path, data in outputs.items():
            freeze(path, data)
        return receipt


def load_contract(root):
    spec = read(root / "training/contact-contract.json")
    contract = ContactTrainingContract(**{key: spec[key] for key in (
        "instructions", "tools", "prompt_version", "source_hashes", "tool_contract",
    )})
    if contract.specification != spec:
        raise ValueError("Training contract or training source changed")
    return contract


def candidate_spec(root, receipt, project):
    return {
        "schema_version": 1, "candidate": "contact-v2-fit", "base_model": STUDENT_MODEL,
        "display_model_id": f"ft-{STUDENT_MODEL}", "initialization": "independent_base_model",
        "model_name": MODEL_NAME, "dataset_name": "bufo-sol-contact-v2-fit-20261020",
        "source_path": str(root / "training/reviewed-examples.jsonl"),
        "source_sha256": receipt["source_sha256"], "upload_sha256": receipt["upload_sha256"],
        "source_receipt_sha256": sha((root / "training/source-receipt.json").read_bytes()),
        "training_config": RECIPE, "contract_sha256": receipt["contract_sha256"],
        "nominal_optimizer_updates": 72, "actual_optimizer_updates": None,
        "project": {key: project[key] for key in (
            "entity", "project", "project_id", "project_alias", "base_url", "training_base_url",
        )},
    }


def train_or_status(mode, root, client, project, *, transport_metadata=None):
    if mode not in {"train", "status"}:
        raise ValueError("Expected train or status")
    root = Path(root).resolve()
    receipt = prepare(root)
    contract = load_contract(root)
    for field in ("entity", "project_alias", "base_url", "training_base_url"):
        if project[field] != getattr(client, field):
            raise ValueError("Client differs from declared project")
    spec = candidate_spec(root, receipt, project)
    folder = root / "training/candidate"
    spec_path, state_path = folder / "candidate.json", folder / "forge-state.json"
    with _locked(root / "training/operation"), contract.activate():
        if spec_path.exists() and read(spec_path) != spec:
            raise ValueError("Frozen candidate recipe changed")
        if state_path.exists() and not spec_path.exists():
            raise ValueError("Cannot adopt workflow without its frozen candidate specification")
        if mode == "status" and (not state_path.exists() or not spec_path.exists()):
            raise ValueError("Cannot poll a candidate without its saved workflow")
        if state_path.exists():
            state = read(state_path)
            _clear_or_block(state)
            expected_state = {
                key: spec[key] for key in (
                    "source_path", "source_sha256", "upload_sha256", "model_name", "dataset_name", "base_model",
                )
            } | spec["project"] | {
                "heldout_ids": receipt["heldout_ids"], "heldout_families": receipt["heldout_families"],
            }
            if any(state.get(key) != value for key, value in expected_state.items()):
                raise ValueError("Existing candidate workflow identity or source differs")
            if state.get("training_config") is not None and state["training_config"] != RECIPE:
                raise ValueError("Existing candidate training configuration changed")
        observed_at = datetime.now(timezone.utc).isoformat()
        freeze(root / "training/transport-history" / f"{uuid4().hex}.json", encoded({
            "observed_at": observed_at, "operation": mode,
            "transport": transport_metadata or {"mode": "client_supplied"},
        }))
        if mode == "train":
            freeze(spec_path, encoded(spec))
            initialize_workflow(
                state_path, client, source_path=spec["source_path"], dataset_name=spec["dataset_name"],
                model_name=spec["model_name"], project_id=project["project_id"], project=project["project"],
                heldout_ids=receipt["heldout_ids"], heldout_families=receipt["heldout_families"],
                lineage={"source_receipt_sha256": spec["source_receipt_sha256"],
                         "contract_sha256": spec["contract_sha256"], "initialization": "independent_base_model"},
            )
            # No remote preflight precedes the source/candidate/pending-operation safeguards.
            client.preflight(STUDENT_MODEL, project=project["project"], project_id=project["project_id"])
            import_reviewed_dataset(state_path, client)
            state = submit_training(state_path, client, epochs=6, batch_size=2, learning_rate=1e-4)
        else:
            state = resume_training(state_path, client)
        observed = {
            "observed_at": observed_at, "model_label": f"ft-{STUDENT_MODEL}",
            **{key: state.get(key) for key in (
                "status", "training_progress", "finetune_id", "dataset_id", "dataset_revision",
                "model_artifact_ref", "training_config", "source_sha256", "upload_sha256", "entry_counts",
            )},
        }
        if observed["model_artifact_ref"]:
            from experiment import require_checkpoint
            require_checkpoint(observed["model_artifact_ref"])
        freeze(folder / "history" / f"{uuid4().hex}.json", encoded(observed))
        return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--status", action="store_true")
    parser.add_argument("--root", type=Path, default=DIRECTORY)
    parser.add_argument("--project-state", type=Path, default=ROOT / "runs/forge-loop/forge-state.json")
    parser.add_argument("--tls12", action="store_true", help="Verified TLS 1.2 upload compatibility")
    args = parser.parse_args()
    if not args.train and not args.status:
        print(json.dumps(prepare(args.root), indent=2))
        return
    project = read(args.project_state)
    transport, metadata = transport_for_cli(args.tls12)
    with ForgeClient(
        api_key=required("WANDB_API_KEY"), entity=project["entity"], project_alias=project["project_alias"],
        base_url=project["base_url"], training_base_url=project["training_base_url"],
        timeout=120, transport=transport,
    ) as client:
        result = train_or_status(
            "train" if args.train else "status", args.root, client, project, transport_metadata=metadata,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
