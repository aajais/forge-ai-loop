"""Read-only deployed-adapter configuration audit; never download weight tensors.

Records exported rank/alpha/base/targets, including per-module overrides. This is
configuration evidence, not a claim about learned numerical rank or quality.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[2]
IMMUTABLE = re.compile(r"wandb-artifact:///[^/:]+/[^/:]+/[^/:]+:v[0-9]+")
MAX_CONFIG_BYTES = 1_048_576


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def binding(state):
    uri = state.get("model_artifact_ref") or ""
    if state.get("status") != "deployed" or not IMMUTABLE.fullmatch(uri):
        raise ValueError("Audit requires a deployed immutable :vN model artifact")
    if not state.get("base_model") or not state.get("finetune_id"):
        raise ValueError("Audit requires recorded base model and Forge job identity")
    return {"model_artifact_ref": uri, "base_model": state["base_model"], "finetune_id": state["finetune_id"]}


def observed_adapter(raw, expected_base):
    config = json.loads(raw)
    if config.get("base_model_name_or_path") != expected_base:
        raise ValueError("Exported adapter base does not match the training base")
    rank = config.get("r")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("Exported adapter config must record positive integer r")
    alpha = config.get("lora_alpha")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0:
        raise ValueError("Exported adapter config must record positive lora_alpha")
    # Keep the complete JSON alongside the receipt. Missing flags are unknown,
    # not silently interpreted as false; rank_pattern can override the base r.
    return {key: config.get(key) for key in (
        "r", "lora_alpha", "lora_dropout", "target_modules", "base_model_name_or_path",
        "rank_pattern", "alpha_pattern", "use_rslora", "use_dora", "use_qalora",
        "peft_type", "peft_version", "art_lora_format")}


def audit(state_path, api):
    state_path = Path(state_path)
    state_bytes = state_path.read_bytes()
    identity = binding(json.loads(state_bytes))
    folder = state_path.parent / "adapter-audit"
    config_path = folder / "adapter_config.json"
    receipt_path = state_path.parent / "adapter-verification.json"
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved.get("source_identity") != identity:
            raise ValueError("Training artifact identity changed since frozen audit")
        raw = config_path.read_bytes()
        if sha(raw) != saved.get("adapter_config_sha256"):
            raise ValueError("Local adapter config changed since frozen audit")
        if observed_adapter(raw, identity["base_model"]) != saved.get("adapter"):
            raise ValueError("Frozen adapter observations changed")
        return saved
    artifact_name = identity["model_artifact_ref"].removeprefix("wandb-artifact:///")
    artifact = api.artifact(artifact_name)
    if artifact.qualified_name != artifact_name or not artifact.id or not artifact.digest:
        raise ValueError("W&B resolved artifact differs from the immutable requested artifact")
    entry = artifact.get_entry("adapter_config.json")
    if not isinstance(entry.size, int) or not 0 < entry.size <= MAX_CONFIG_BYTES:
        raise ValueError("Adapter configuration is missing or exceeds the metadata size limit")
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="config-", dir=folder) as temporary:
        downloaded = Path(entry.download(root=temporary))
        if not downloaded.resolve().is_relative_to(Path(temporary).resolve()):
            raise ValueError("Artifact download returned a path outside its metadata directory")
        if downloaded.stat().st_size > MAX_CONFIG_BYTES:
            raise ValueError("Downloaded adapter configuration exceeds metadata size limit")
        raw = downloaded.read_bytes()
    adapter = observed_adapter(raw, identity["base_model"])
    if config_path.exists() and config_path.read_bytes() != raw:
        raise ValueError("Existing adapter config changed; refusing to overwrite audit evidence")
    if not config_path.exists():
        with config_path.open("xb") as stream:
            stream.write(raw)
    receipt = {
        "schema_version": 1, "observed_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": identity, "source_state_sha256": sha(state_bytes),
        "artifact": {"uri": identity["model_artifact_ref"], "id": artifact.id,
                     "digest": artifact.digest, "qualified_name": artifact.qualified_name},
        "adapter_config_path": str(config_path.resolve()), "adapter_config_sha256": sha(raw),
        "adapter_config_bytes": len(raw), "artifact_file_digest": entry.digest,
        "adapter": adapter, "tensor_shapes_verified": False, "weight_tensor_bytes_downloaded": 0,
        "scope": "Exported configuration only; per-module rank_pattern overrides r. No tensor values, effective numerical rank, or performance verified.",
    }
    with receipt_path.open("xb") as stream:
        stream.write(encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/forge-iteration2")
    parser.add_argument("--candidate", choices=["e1", "e3"], action="append")
    args = parser.parse_args()
    # Load local credentials without printing them. PublicApi does only GET/query
    # operations here; no run, artifact publish, job, or registry mutation exists.
    from bufogen.settings import required
    import wandb
    api = wandb.Api(api_key=required("WANDB_API_KEY"), timeout=120)
    results = {}
    for candidate in args.candidate or ["e1", "e3"]:
        state = args.root / "training" / candidate / "forge-state.json"
        result = audit(state, api)
        results[candidate] = {"artifact": result["artifact"]["uri"], "adapter": result["adapter"],
                              "receipt": str(state.parent / "adapter-verification.json")}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
