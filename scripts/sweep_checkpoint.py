"""Add a deployed sweep job's entry to runs/distill/stage-b/checkpoints.json (no network but the
read-only adapter rank lookup). Usage: sweep_checkpoint.py <label>"""

import json
import sys

from adapter_rank import adapter_rank

from bufogen.integrations.forge import FORGE_LORA_RANK_FIELD
from bufogen.settings import ROOT

SIDECAR = ROOT / "runs/distill/stage-b/checkpoints.json"
SWEEP = ROOT / "runs/distill/stage-b/sweep"
# Forge accepts no rank field and fixes the rank per base model (observed 1 for Gemma; Forge docs
# give 8 for Qwen3.8-27B). adapter_rank_observed is read separately from the artifact.
FIXED_RANK = {
    "google/gemma-4-26B-A4B-it": 1,
    "Qwen/Qwen3.8-27B": 8,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B": 8,
}


def main(label):
    state = json.loads((SWEEP / f"forge-state-{label}.json").read_text())
    assert state["status"] == "deployed" and state["model_artifact_ref"], "job not deployed"
    config = state["training_config"]
    entry = {
        "arm_name": label,
        "artifact": state["model_artifact_ref"],
        "forge_job_id": state["finetune_id"],
        "forge_dataset_id": state["dataset_id"],
        "sft_config": {
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            # Configured rank: the field when the job carried one, else 1 (Forge accepts no rank
            # field, so the adapter default applies); adapter_rank_observed is read separately.
            "lora_rank": config.get(FORGE_LORA_RANK_FIELD) or FIXED_RANK[state["base_model"]],
            "assistant_turns": config["assistant_turns"],
        },
        "base_model": state["base_model"],
        "adapter_rank_observed": adapter_rank(state["model_artifact_ref"]),
        "checkpoint_version": state["model_artifact_ref"].rpartition(":")[2],
    }
    entries = json.loads(SIDECAR.read_text())
    slots = [i for i, e in enumerate(entries) if e["arm_name"] == label]
    if slots:  # idempotent: replace in place, keep the file order
        entries[slots[0]] = entry
    else:
        entries.append(entry)
    SIDECAR.write_text(json.dumps(entries, indent=2) + "\n")
    print(json.dumps(entry))


if __name__ == "__main__":
    main(sys.argv[1])
