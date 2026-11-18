#!/usr/bin/env bash
# Stage B paid runs. Usage: scripts/distill_stage_b.sh <step> [attempt]
# steps: heldout-ref heldout-sol data-gen <n> data-judge <n> data-decide <n> data-export lowconf-queue records
#        sweep-import <label> sweep-job <label> sweep-status <label> sweep-eval <label> sweep-checkpoint <label> sweep-pair <label> sweep-distinctness sweep-select
#        rl-update <label> <base_arm> <initial_checkpoint> <forge_finetune_id> <steps> [agent flags]  (Task 15)
#        rl-eval <label> <sft_arm> <rl_artifact> <sft_artifact> [agent flags]   (agent flags e.g. --max-tokens 20000 --disable-thinking)
#        confirm-gen confirm-score confirm-pair confirm-packet (Task 10)
#        (labels sft_j1 sft_j3 sft_j4)
# Heldout drawings are produced up front with the same agent configuration as every other arm and
# are never judged, reviewed or opened until the single confirmation run (plan Task 10).
# Appends command, wall time and exit code to runs/distill/stage-b/stage-b-log.md; the first failing
# command is logged with its exit code and then stops the script.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=runs/distill/stage-b/stage-b-log.md
CFG=config/distill_prompts.yaml
DATA=runs/distill/stage-b/sol-data
SWEEP=runs/distill/stage-b/sweep
CONF=runs/distill/stage-b/confirmation
PROJECT_ID=dd5ddbe2-b31c-43e1-ab39-b5488c78f3d8
mkdir -p runs/distill/stage-b runs/references "$SWEEP"

# Runs its arguments as argv (never re-parsed as a string), logs wall time and exit code, and returns
# that code so `set -e` stops the script after the failure is on record.
run() {
  local t0=$SECONDS rc=0
  echo "## $(date -u +%FT%TZ) START: \`$*\`" >> "$LOG"
  "$@" || rc=$?
  echo "- wall: $((SECONDS - t0))s, exit: $rc" >> "$LOG"
  return "$rc"
}

# Per-label sweep configuration: in-run base arm, Forge base model, epochs.
sweep_cfg() {
  case "$1" in
  sft_j1) base_arm=open_base; base_model=google/gemma-4-26B-A4B-it; epochs=3 ;;
  sft_j3) base_arm=open_base; base_model=google/gemma-4-26B-A4B-it; epochs=1 ;;
  sft_j4) base_arm=open_base; base_model=google/gemma-4-26B-A4B-it; epochs=3 ;;
  sft_q3) base_arm=qwen_base; base_model=Qwen/Qwen3.8-27B; epochs=1 ;;
  sft_n3) base_arm=nemotron_base; base_model=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B; epochs=1 ;;
  *) echo "unknown label"; exit 2 ;;
  esac
}

# Serving base model of a base arm (the RL student base).
base_model_of() {
  case "$1" in
  open_base) base_model=google/gemma-4-26B-A4B-it ;;
  qwen_base) base_model=Qwen/Qwen3.8-27B ;;
  nemotron_base) base_model=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B ;;
  *) echo "unknown base arm"; exit 2 ;;
  esac
}

case "${1:-}" in
heldout-ref)
  # Sol reference drawings for the 80 heldout prompts, then their reference manifest (no judging).
  run uv run bufogen --stage benchmark --benchmark-config "$CFG" --split heldout --backends closed_quality_bar --output runs/distill/stage-b/heldout-sol-ref --concurrency 4 --max-turns 100 --max-tokens 6000
  run uv run bufogen --stage build-references --output runs/distill/stage-b/heldout-sol-ref --references-out runs/references/distill-heldout-v1 ;;
heldout-sol)
  # Sol as an evaluated arm on the 80 heldout prompts (a second independent run, same configuration).
  run uv run bufogen --stage benchmark --benchmark-config "$CFG" --split heldout --backends closed_quality_bar --output runs/distill/stage-b/heldout-sol-arm --concurrency 4 --max-turns 100 --max-tokens 6000 ;;
data-gen)
  # Sol on the 256 train prompts; every attempt is its own frozen run directory.
  n=${2:?attempt number}
  run uv run bufogen --stage benchmark --benchmark-config "$CFG" --split train --backends closed_quality_bar --output "$DATA/attempt-$n" --concurrency 4 --max-turns 100 --max-tokens 6000 ;;
data-judge)
  # Calibrated judge (bufo-astra-v5, 3 samples); judge errors get one automatic re-run (cached ok rows kept).
  n=${2:?attempt number}
  judge=(uv run bufogen --stage benchmark-score --output "$DATA/attempt-$n" --judge astra --rubric bufo-astra-v5 --judge-samples 3 --references runs/references/distill-v2 --publish --concurrency 4)
  run "${judge[@]}"
  errors() { python3 -c 'import json,sys; print(sum(a["judge_errors"] for a in json.load(open(sys.argv[1]))["arms"].values()))' "$DATA/attempt-$n/astra-v5-k3/summary.json"; }
  e=$(errors) || { echo "- summary.json unreadable" >> "$LOG"; exit 1; }
  echo "- judge_errors after first run: $e" >> "$LOG"
  if [ "$e" -gt 0 ]; then
    run "${judge[@]}"
    e=$(errors) || { echo "- summary.json unreadable" >> "$LOG"; exit 1; }
    echo "- judge_errors after re-run: $e" >> "$LOG"
  fi
  [ "$e" -eq 0 ] || { echo "- judge_errors still $e after re-run" >> "$LOG"; exit 1; }
  name=$(grep -o 'bufo-astra[^"/:]*-v5-k3' "$DATA/attempt-$n/astra-v5-k3/weave-publication.json" 2>/dev/null | sort -u | head -1 || true)
  [ -n "$name" ] || { echo "- no -v5-k3 Weave evaluation name in weave-publication.json" >> "$LOG"; exit 1; }
  echo "- weave evaluation: $name" >> "$LOG"; echo "$name" ;;
data-decide)
  n=${2:?attempt number}
  run uv run python scripts/judge_review_decisions.py --scored-dir "$DATA/attempt-$n/astra-v5-k3" --rubric bufo-astra-v5 --samples 3 --out "runs/distill/stage-b/decisions/attempt-$n.jsonl" ;;
data-export)
  # export-reviewed is single-manifest: one export per attempt that has decisions, then concatenate.
  for d in "$DATA"/attempt-*; do
    a=$(basename "$d")
    [ -f "runs/distill/stage-b/decisions/$a.jsonl" ] || continue
    run uv run bufogen --stage export-reviewed --manifest "$d/manifest.jsonl" --reviews "runs/distill/stage-b/decisions/$a.jsonl" --benchmark-config "$CFG" --output "runs/distill/stage-b/export/$a"
  done
  run uv run python scripts/data_funnel.py
  run uv run python scripts/publish_stage_b_records.py ;;
lowconf-queue)
  run uv run python scripts/lowconf_queue.py
  run uv run python scripts/publish_stage_b_records.py ;;
records)
  # Free W&B/Weave logging of every derived record that exists (idempotent; ledger publications.json).
  run uv run python scripts/publish_stage_b_records.py ;;
sweep-import)
  # One dataset per distinct training file: sft_j1 imports the full accepted set (sft_j3 shares it by
  # cloning j1's state with a fresh model name); sft_j4 imports its half-data export.
  label=${2:?label}; state="$SWEEP/forge-state-$label.json"
  case "$label" in
  sft_j1) src=runs/distill/stage-b/export/accepted.jsonl ;;
  sft_j4) src=runs/distill/stage-b/export/accepted-half.jsonl ;;
  sft_j3|sft_q3|sft_n3)
    # Shares sft_j1's dataset (same dataset_id); only its model name, recorded dataset name and (q3)
    # base model change. The state file is created, never overwritten.
    sweep_cfg "$label"
    run uv run python - "$SWEEP" "$label" "$base_model" <<'PY'
import json, os, sys
d, label, base_model = sys.argv[1:4]
target = f"{d}/forge-state-{label}.json"
assert not os.path.exists(target), f"{target} already exists"
s = json.load(open(f"{d}/forge-state-sft_j1.json"))
assert s["entries_imported"] and s["dataset_id"], "sft_j1 dataset not imported"
short = label.replace("sft_", "sft-")
s.update(finetune_id=None, task_id=None, model_artifact_ref=None, training_model_identity=None,
         status="dataset_ready", remote_status=None, training_config=None, pending_operation=None)
s["model_name"] = s["model_name"].replace("sft-j1", short)
s["dataset_name"] = s["dataset_name"].replace("sft-j1", short)
s["base_model"] = base_model
s["shared_dataset_from"] = "sft_j1"
json.dump(s, open(target, "w"), indent=2)
PY
    exit 0 ;;
  *) echo "unknown label"; exit 2 ;;
  esac
  run uv run bufogen --stage forge-import --tls12 --benchmark-config "$CFG" --source "$src" --dataset-name "bufo-stageb-${label//_/-}-20261023" --name "bufo-stageb-${label//_/-}-20261023" --project-id "$PROJECT_ID" --forge-state "$state" ;;
sweep-job)
  label=${2:?label}; state="$SWEEP/forge-state-$label.json"
  sweep_cfg "$label"
  # sft_j4 may use epochs 1 if j3 wins: pass EPOCHS=1.
  run uv run bufogen --stage forge-submit --tls12 --forge-state "$state" --base-model "$base_model" --epochs "${EPOCHS:-$epochs}" --batch-size 2 --learning-rate 1e-4 ;;
sweep-status)
  label=${2:?label}
  run uv run bufogen --stage forge-status --tls12 --forge-state "$SWEEP/forge-state-$label.json" ;;
sweep-eval)
  # Needs the arm's entry in runs/distill/stage-b/checkpoints.json before the publish.
  label=${2:?label}; artifact=${ARTIFACT:?ARTIFACT=wandb-artifact:///...:vN}
  sweep_cfg "$label"
  run uv run bufogen --stage benchmark --benchmark-config "$CFG" --split validation --backends "$base_arm" --checkpoint "$label=$artifact" --output "$SWEEP/$label" --concurrency 4 --max-turns 100 --max-tokens 6000
  run uv run bufogen --stage benchmark-score --output "$SWEEP/$label" --judge astra --rubric bufo-astra-v5 --judge-samples 3 --references runs/references/distill-v2 --publish --concurrency 4 ;;
sweep-checkpoint)
  # Sidecar entry (+ adapter rank read read-only from the artifact's adapter_config.json); idempotent.
  label=${2:?label}
  run uv run python scripts/sweep_checkpoint.py "$label" ;;
sweep-pair)
  label=${2:?label}; sweep_cfg "$label"
  if [ "$base_arm" != open_base ]; then  # selection comparison: the in-run base arm
    run uv run python scripts/paired_across_runs.py --left "$SWEEP/$label/astra-v5-k3" --left-arm "$label" --right "$SWEEP/$label/astra-v5-k3" --right-arm "$base_arm" --out "$SWEEP/$label/paired-$label-vs-$base_arm.json"
  fi
  run uv run python scripts/paired_across_runs.py --left "$SWEEP/$label/astra-v5-k3" --left-arm "$label" --right runs/distill/arms-v1-students/astra-v5-k3 --right-arm open_base --out "$SWEEP/$label/paired-$label-vs-open_base.json"
  run uv run python scripts/paired_across_runs.py --left "$SWEEP/$label/astra-v5-k3" --left-arm "$label" --right runs/distill/arms-v1/astra-v5-k3 --right-arm closed_quality_bar --out "$SWEEP/$label/paired-$label-vs-closed_quality_bar.json" ;;
sweep-distinctness)
  # Free, local: was a distinct model served? (served_model only echoes the base name.)
  run uv run python scripts/sweep_distinctness.py --sweep-dir "$SWEEP" --arm sft_j1 --arm sft_j3 --arm sft_q3 --arm sft_n3 --base-arm sft_q3=qwen_base --base-arm sft_n3=nemotron_base --control-run runs/distill/arms-v1-students/astra-v5-k3 --sol-run runs/distill/arms-v1/astra-v5-k3 --export runs/distill/stage-b/export/accepted.jsonl --seed 1 --replicates 20000 --out "$SWEEP/distinctness.json" ;;
sweep-select)
  run uv run python scripts/sweep_select.py --sweep-dir "$SWEEP" --arm sft_j1 --arm sft_j3 --arm sft_q3 --arm sft_n3 --base-arm sft_q3=qwen_base --base-arm sft_n3=nemotron_base --out "$SWEEP/selection.json" ;;
rl-update)
  # Serverless RL continuing an SFT run (astra reward). Rollouts must run under the evaluation's agent
  # configuration: pass the same flags later given to rl-eval (the defaults are 6000 tokens, thinking on).
  label=${2:?label}; base_arm=${3:?base arm}; initial=${4:?initial checkpoint}; finetune=${5:?forge finetune id}; steps=${6:?steps}
  shift 6
  base_model_of "$base_arm"
  run uv run bufogen --stage train-rl --reward astra --calibration runs/distill/stage-b/rl/astra-calibration.json --dataset-ref "${DATASET_REF:?DATASET_REF=wandb-artifact:///...:vN}" --name "bufo-stageb-${label//_/-}-20261023" --base-model "$base_model" --initial-checkpoint "$initial" --forge-finetune-id "$finetune" --steps "$steps" "$@" ;;
rl-eval)
  # Validation benchmark of the RL checkpoint and the unchanged SFT checkpoint it continued from, in
  # one run at the same agent configuration and judge; the in-run paired interval is RL - SFT (plan
  # Task 9). The base arm under this configuration is compared cross-run. Needs the labels'
  # models.yaml entries before publishing.
  label=${2:?label}; sft_arm=${3:?sft arm}; artifact=${4:?rl artifact}; sft_artifact=${5:?sft artifact}
  shift 5
  RL="${RL_OUT:-runs/distill/stage-b/rl/$label}"
  run uv run bufogen --stage benchmark --benchmark-config "$CFG" --split validation --checkpoint "$label=$artifact" --checkpoint "$sft_arm=$sft_artifact" --output "$RL" --concurrency 4 --max-turns 100 "$@"
  run uv run bufogen --stage benchmark-score --output "$RL" --judge astra --rubric bufo-astra-v5 --judge-samples 3 --references runs/references/distill-v2 --publish --concurrency 4
  run uv run python scripts/paired_across_runs.py --left "$RL/astra-v5-k3" --left-arm "$label" --right "$RL/astra-v5-k3" --right-arm "$sft_arm" --out "$RL/paired-$label-vs-$sft_arm.json" ;;
confirm-gen)
  # Task 10: base Gemma on the 80 heldout prompts, same agent configuration as the Sol arm (no SFT/RL arm: no winner).
  run uv run bufogen --stage benchmark --benchmark-config "$CFG" --split heldout --backends open_base --output "$CONF" --concurrency 4 --max-turns 100 --max-tokens 6000 ;;
confirm-score)
  # The single heldout judging; both directories, no baseline arm; judge_errors must be 0 (no re-judging here).
  for d in "$CONF" runs/distill/stage-b/heldout-sol-arm; do
    run uv run bufogen --stage benchmark-score --output "$d" --judge astra --rubric bufo-astra-v5 --judge-samples 3 --references runs/references/distill-heldout-v1 --confirmation --publish --concurrency 4
    e=$(python3 -c 'import json,sys; print(sum(a["judge_errors"] for a in json.load(open(sys.argv[1]))["arms"].values()))' "$d/astra-v5-k3/summary.json") || { echo "- summary.json unreadable: $d" >> "$LOG"; exit 1; }
    echo "- judge_errors $d: $e" >> "$LOG"
    [ "$e" -eq 0 ] || { echo "- judge_errors $e in $d; stopping (no re-judge)" >> "$LOG"; exit 1; }
  done ;;
confirm-pair)
  run uv run python scripts/paired_across_runs.py --left "$CONF/astra-v5-k3" --left-arm open_base --right runs/distill/stage-b/heldout-sol-arm/astra-v5-k3 --right-arm closed_quality_bar --out "$CONF/paired-open_base-vs-closed_quality_bar.json" ;;
confirm-packet)
  # Blinded 40-item packet mixed across base and Sol; the key lives outside the packet directory.
  run uv run python scripts/make_review_packet.py --runs "$CONF" --runs runs/distill/stage-b/heldout-sol-arm --split heldout --confirmation --seed 20261027 --scored-dir astra-v5-k3 --size 40 --out "$CONF/packet" --key-dir "$CONF/key" ;;
*) echo "unknown step"; exit 2 ;;
esac
