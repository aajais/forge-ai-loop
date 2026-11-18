"""Resumable managed SFT with atomic local lineage and fail-closed uncertain writes.

Each public operation holds a process lock. A persisted intent precedes every remote
mutation, so a crash after the request cannot silently duplicate an import or job.
"""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from bufogen.integrations.forge import (
    FORGE_LORA_RANK_FIELD,
    STUDENT_MODEL,
    STUDENT_MODELS,
    ForgeError,
    reviewed_entries,
    validate_learning_rate,
    validate_lora_rank,
)


class AmbiguousOperation(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _save(path, state):
    state["updated_at"] = _now()
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _locked(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Forge operation holds this workflow lock") from None
        try:
            yield path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _source(path, *, heldout_ids=(), heldout_families=()):
    data = Path(path).read_bytes()
    examples = [json.loads(line) for line in data.splitlines() if line.strip()]
    entries = reviewed_entries(examples, heldout_ids=heldout_ids, heldout_families=heldout_families)
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return entries, _sha(data), _sha(encoded)


def _load(path, client):
    state = json.loads(path.read_text())
    if (
        state.get("schema_version") != 1
        or state["entity"] != client.entity
        or state["project_alias"] != client.project_alias
        or state["base_url"] != client.base_url
        or state["training_base_url"] != client.training_base_url
    ):
        raise ValueError("Workflow client identity or schema mismatch")
    if _sha(Path(state["source_path"]).read_bytes()) != state["source_sha256"]:
        raise ValueError("Workflow source dataset changed; create a separate workflow")
    return state


def _clear_or_block(state):
    if state.get("pending_operation"):
        operation = state["pending_operation"]["name"]
        raise AmbiguousOperation(
            f"Unresolved {operation}: reconcile the remote ID before retrying; "
            "do not delete pending_operation or blindly replay the write"
        )


def _mutate(path, state, name, operation, accept):
    _clear_or_block(state)
    state["pending_operation"] = {"name": name, "started_at": _now()}
    state["status"] = name + "_pending"
    _save(path, state)
    try:
        result = operation()
        accept(result)  # Validate acknowledgement and save IDs before clearing intent.
    except ForgeError as exc:
        state["last_error"] = {
            "message": str(exc),
            "status_code": exc.status_code,
            "uncertain": exc.uncertain,
            "operation": name,
        }
        if not exc.uncertain:
            state["pending_operation"] = None
            state["status"] = name + "_rejected"
        else:
            state["status"] = "reconciliation_required"
        _save(path, state)
        raise
    except (ValueError, KeyError, TypeError):
        state["status"] = "reconciliation_required"
        _save(path, state)
        raise
    state["pending_operation"] = None
    state["last_error"] = None
    _save(path, state)
    return result


def _remote_id(response):
    value = response.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("Forge write acknowledgement omitted its ID; reconcile before retrying")
    return value


def initialize_workflow(
    state_path,
    client,
    *,
    source_path,
    dataset_name,
    model_name,
    project_id,
    project=None,
    base_model=STUDENT_MODEL,
    heldout_ids=(),
    heldout_families=(),
    lineage=None,
):
    """Validate locally and freeze lineage. Performs no HTTP operations."""
    if base_model not in STUDENT_MODELS:
        raise ValueError("This workflow requires an exact configured student model")
    if not all(isinstance(v, str) and v.strip() for v in (dataset_name, model_name, project_id)):
        raise ValueError("Explicit dataset/model names and existing project ID are required")
    source_path = Path(source_path).resolve()
    entries, source_hash, upload_hash = _source(
        source_path,
        heldout_ids=heldout_ids,
        heldout_families=heldout_families,
    )
    state = {
        "schema_version": 1,
        "entity": client.entity,
        "project_alias": client.project_alias,
        "project": project or client.project_alias,
        "project_id": project_id,
        "base_url": client.base_url,
        "training_base_url": client.training_base_url,
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "upload_sha256": upload_hash,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "base_model": base_model,
        "dataset_id": None,
        "relabel_run_id": None,
        "finetune_id": None,
        "task_id": None,
        "model_artifact_ref": None,
        "dataset_artifact_ref": None,
        "training_model_identity": None,
        "entry_counts": {
            split: sum(e["split"] == split for e in entries) for split in ("train", "val")
        },
        "heldout_ids": sorted(heldout_ids),
        "heldout_families": sorted(heldout_families),
        "lineage": lineage or {},
        "entries_imported": False,
        "pending_operation": None,
        "status": "prepared",
        "created_at": _now(),
    }
    with _locked(state_path) as path:
        if path.exists():
            existing = _load(path, client)
            for field in (
                "source_path",
                "source_sha256",
                "upload_sha256",
                "dataset_name",
                "model_name",
                "project_id",
                "project",
                "base_model",
                "heldout_ids",
                "heldout_families",
                "lineage",
            ):
                if existing[field] != state[field]:
                    raise ValueError(f"Existing workflow differs at {field}")
            return existing
        _save(path, state)
    return state


def import_reviewed_dataset(state_path, client):
    """Create a dataset once, dry-run exact mapped entries, then append once."""
    with _locked(state_path) as path:
        state = _load(path, client)
        _clear_or_block(state)
        if state["entries_imported"]:
            return state
        entries, _, upload_hash = _source(
            state["source_path"],
            heldout_ids=state["heldout_ids"],
            heldout_families=state["heldout_families"],
        )
        if upload_hash != state["upload_sha256"]:
            raise ValueError("Mapped source dataset changed")
        if not state["dataset_id"]:

            def accept_dataset(result):
                state["dataset_id"] = _remote_id(result)
                state["status"] = "dataset_created"

            _mutate(
                path,
                state,
                "create_dataset",
                lambda: client.create_dataset(state["dataset_name"]),
                accept_dataset,
            )
        validation = client.upload_entries(state["dataset_id"], entries, dry_run=True)
        if validation.get("dry_run") is not True or validation.get("group_conflicts"):
            raise ValueError("Forge dry-run validation failed or returned group conflicts")
        state["dry_run_result"] = validation
        state["status"] = "dataset_validated"
        _save(path, state)

        def accept_upload(result):
            if result.get("dry_run") is not False or result.get("group_conflicts"):
                raise ValueError("Forge upload acknowledgement is inconsistent; reconcile dataset")
            counts = result.get("entry_counts", {})
            if counts.get("total") != len(entries) or any(
                counts.get(split) != state["entry_counts"][split] for split in ("train", "val")
            ):
                raise ValueError("Forge uploaded counts differ from source; reconcile dataset")
            state["upload_result"] = result
            state["dataset_revision"] = result.get("revision")
            state["entries_imported"] = True
            state["status"] = "dataset_ready"

        _mutate(
            path,
            state,
            "upload_entries",
            lambda: client.upload_entries(state["dataset_id"], entries, dry_run=False),
            accept_upload,
        )
        return state


def _record_job(state, response):
    remote_id = _remote_id(response)
    if state["finetune_id"] and remote_id != state["finetune_id"]:
        raise ValueError("Forge returned a different fine-tune ID")
    for key in ("dataset_id", "base_model", "model_name"):
        if response.get(key) and response[key] != state[key]:
            raise ValueError(f"Forge fine-tune {key} does not match workflow")
    state["finetune_id"] = remote_id
    for key in ("task_id", "dataset_artifact_ref", "training_progress", "training_failure_type"):
        if key in response:
            state[key] = response[key]
    # Never equate task_id with the Serverless Training model ID.
    identity = {
        key: response[key]
        for key in ("model_id", "training_model_id", "run_id")
        if response.get(key)
    }
    if identity:
        state["training_model_identity"] = {"source": "finetune_response", **identity}
    remote_status = response.get("status", "queued")
    state["remote_status"] = remote_status
    state["status"] = remote_status
    if remote_status == "deployed":
        artifact = response.get("model_artifact_ref")
        if isinstance(artifact, str) and artifact.strip():
            state["model_artifact_ref"] = artifact
        else:
            state["model_artifact_ref"] = None
            state["status"] = "deployed_missing_artifact"
    else:
        state["model_artifact_ref"] = None


def submit_training(
    state_path,
    client,
    *,
    epochs=1,
    batch_size=2,
    learning_rate=None,
    lora_rank=None,
    base_model=None,
):
    """Submit exactly one managed job; existing job IDs are polled, never recreated."""
    with _locked(state_path) as path:
        state = _load(path, client)
        _clear_or_block(state)
        validate_learning_rate(learning_rate)
        validate_lora_rank(lora_rank)
        if base_model is not None and base_model != state["base_model"]:
            raise ValueError(
                f"Requested base_model differs from the workflow's ({state['base_model']})"
            )
        config = {"epochs": epochs, "batch_size": batch_size, "assistant_turns": "all"}
        overrides = {}
        if learning_rate is not None:
            config["learning_rate"] = learning_rate
            overrides["learning_rate"] = learning_rate
        if lora_rank is not None:
            config[FORGE_LORA_RANK_FIELD] = lora_rank
            overrides["lora_rank"] = lora_rank
        if state["finetune_id"]:
            if (learning_rate is not None or lora_rank is not None) and state.get(
                "training_config"
            ) != config:
                raise ValueError(
                    "Existing job training configuration differs from requested override"
                )
            _record_job(state, client.get_finetune(state["finetune_id"]))
            _save(path, state)
            return state
        if not state["entries_imported"] or not state["dataset_id"]:
            raise ValueError("Import and validate reviewed dataset before submitting training")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if batch_size != "auto" and (type(batch_size) is not int or not 1 <= batch_size <= 32):
            raise ValueError("batch_size must be auto or 1–32")
        state["training_config"] = config
        _mutate(
            path,
            state,
            "create_finetune",
            lambda: client.create_finetune(
                dataset_id=state["dataset_id"],
                model_name=state["model_name"],
                base_model=state["base_model"],
                epochs=epochs,
                batch_size=batch_size,
                **overrides,
            ),
            lambda result: _record_job(state, result),
        )
        return state


def resume_training(state_path, client):
    """Poll a saved job once. Never creates a job or changes inference routing."""
    with _locked(state_path) as path:
        state = _load(path, client)
        _clear_or_block(state)
        if not state["finetune_id"]:
            raise ValueError("No saved fine-tune ID; explicit submit_training is required")
        _record_job(state, client.get_finetune(state["finetune_id"]))
        _save(path, state)
        return state


def reconcile_remote_id(state_path, client, *, remote_id):
    """Adopt an independently observed ID after an ambiguous create, using GET only.

    Ambiguous entry uploads require manual inspection of the dataset revision/rows;
    this function deliberately cannot mark an uncertain upload as complete.
    """
    with _locked(state_path) as path:
        state = _load(path, client)
        operation = (state.get("pending_operation") or {}).get("name")
        if operation == "create_finetune":
            response = client.get_finetune(remote_id)
            if response.get("id") != remote_id:
                raise ValueError("Reconciled job ID mismatch")
            for field in ("dataset_id", "model_name", "base_model"):
                if response.get(field) != state[field]:
                    raise ValueError(f"Reconciled job does not match {field}")
            _record_job(state, response)
        elif operation == "create_dataset":
            response = client.get_dataset(remote_id)
            if response.get("name") != state["dataset_name"] or response.get("id") != remote_id:
                raise ValueError("Reconciled dataset identity/name mismatch")
            state["dataset_id"] = remote_id
            state["status"] = "dataset_created"
        else:
            raise AmbiguousOperation("Only ambiguous dataset/job creation can adopt a remote ID")
        state["pending_operation"] = None
        state["last_error"] = None
        _save(path, state)
        return state
