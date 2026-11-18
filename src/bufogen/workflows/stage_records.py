"""Idempotent Weave + W&B publication of Stage B derived records, with a local ledger."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from bufogen.settings import ROOT, project_path

LEDGER = ROOT / "runs/distill/stage-b/publications.json"
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _secret_values() -> list[str]:
    return [
        value
        for name, value in os.environ.items()
        if any(marker in name.upper() for marker in _SECRET_MARKERS) and len(value) >= 8
    ]


def assert_no_env_leak(*texts: str) -> None:
    """Refuse to publish anything containing a secret-looking environment value."""
    secrets = _secret_values()
    for text in texts:
        if any(secret in text for secret in secrets):
            raise ValueError("Refusing to publish: content contains an environment value.")


def _load(ledger: Path) -> list[dict]:
    return json.loads(ledger.read_text()) if ledger.exists() else []


def publish_record(kind: str, path: Path, *, metadata: dict, ledger: Path = LEDGER) -> dict:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    same_kind = [e for e in _load(ledger) if e["kind"] == kind]
    if same_kind and same_kind[-1]["sha256"] == sha:
        return {**same_kind[-1], "published": False}
    text = data.decode("utf-8")
    assert_no_env_leak(json.dumps(metadata, default=str), text)

    import weave

    import wandb
    from bufogen.integrations import telemetry

    telemetry.init_weave()
    name = f"bufo-stage-b-{kind}"
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        obj = weave.Dataset(name=name, rows=rows)
    else:
        content = json.loads(text) if path.suffix == ".json" else text
        obj = {"kind": kind, "sha256": sha, "metadata": metadata, "content": content}
    weave_ref = str(weave.publish(obj, name=name).uri())

    entity, project = project_path().split("/")
    with wandb.init(
        entity=entity,
        project=project,
        job_type="stage-b-record",
        group="stage-b",
        name=f"stage-b-{kind}",
        reinit=True,
        config={"kind": kind, "sha256": sha, "weave_ref": weave_ref},
    ) as run:
        artifact = wandb.Artifact(name, type="stage-b-record", metadata={**metadata, "sha256": sha})
        artifact.add_file(str(path), name=path.name)
        artifact = run.log_artifact(artifact, aliases=["latest"])
        artifact.wait()
        entry = {
            "kind": kind,
            "sha256": sha,
            "weave_ref": weave_ref,
            "wandb_artifact": f"{entity}/{project}/{artifact.name}",
            "run_id": run.id,
            "published_at": datetime.now(UTC).isoformat(),
        }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps([*_load(ledger), entry], indent=2) + "\n")
    return {**entry, "published": True}
