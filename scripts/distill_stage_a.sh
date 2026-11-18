#!/usr/bin/env bash
# Stage A paid runs. Usage: scripts/distill_stage_a.sh <step>
# steps: step1 step2 step3a step3b step4a step4b step5 step6 queue pull step7 <reviewer> <decisions.json> paired dev-rubric <bufo-astra-vN> [k] rejudge <bufo-astra-vN> <k> packet-v4 <bufo-astra-vN> <k> <seed>
# Appends command, wall time and exit code to runs/distill/stage-a-log.md; the first failing
# command is logged with its exit code and then stops the script.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=runs/distill/stage-a-log.md
CFG=config/distill_prompts.yaml
CKPT=iter2_e3=wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-iter2-e3-20261016:v3
CAL=runs/distill/calibration-v1
mkdir -p runs/distill runs/references "$CAL"

# Runs its arguments as argv (never re-parsed as a string), logs wall time and exit code, and returns
# that code so `set -e` stops the script after the failure is on record.
run() {
  local t0=$SECONDS rc=0
  echo "## $(date -u +%FT%TZ) START: \`$*\`" >> "$LOG"
  "$@" || rc=$?
  echo "- wall: $((SECONDS - t0))s, exit: $rc" >> "$LOG"
  return "$rc"
}

case "${1:-}" in
step1) run uv run bufogen --stage benchmark --benchmark-config "$CFG" --split train validation --backends closed_quality_bar --output runs/distill/sol-run1 --concurrency 4 --max-turns 100 --max-tokens 6000 ;;
step2) run uv run bufogen --stage build-references --output runs/distill/sol-run1 --references-out runs/references/distill-v1 ;;
step3a) run uv run bufogen --stage benchmark --benchmark-config "$CFG" --split train validation --backends closed_quality_bar --output runs/distill/arms-v1 --concurrency 4 ;;
step3b) run uv run bufogen --stage benchmark --benchmark-config "$CFG" --split validation --backends open_base --checkpoint "$CKPT" --output runs/distill/arms-v1-students --concurrency 2 ;;
step4a) run uv run bufogen --stage benchmark-score --output runs/distill/arms-v1 --judge astra --references runs/references/distill-v1 --baseline-arm closed_quality_bar --publish ;;
step4b) run uv run bufogen --stage benchmark-score --output runs/distill/arms-v1-students --judge astra --references runs/references/distill-v1 --publish ;;
step5)
  run uv run python -c 'import json
ids = sorted(json.loads(l)["prompt_id"] for l in open("runs/distill/sol-run1/manifest.jsonl") if json.loads(l)["benchmark_metadata"]["split"] == "validation")
open("runs/distill/calibration-v1/self-equal-ids.txt", "w").write("\n".join(ids[:24]) + "\n")'
  run uv run bufogen --stage benchmark-score --output runs/distill/sol-run1 --judge astra --self-reference --prompt-ids-file "$CAL/self-equal-ids.txt" --publish ;;
step6)
  # Packet v3 (v1 was disclosed, v2 resampled two disclosed rows; both kept under $CAL).
  # Disclosed generations are excluded; the key goes outside the directory the reviewer opens.
  run uv run python scripts/make_review_packet.py --out "$CAL/packet" --key-dir "$CAL/key" --seed 20261023 --exclude-ids-file "$CAL/exclude-disclosed.txt" ;;
step7)
  # Agreement between a reviewer's downloaded decisions.json and the judge; reviewer is a label.
  [ $# -eq 3 ] || { echo "usage: $0 step7 <reviewer> <decisions.json>"; exit 2; }
  run uv run python scripts/review_agreement.py --packet "$CAL/packet" --key "$CAL/key/unblinding-key.json" --decisions "$3" --reviewer "$2" ;;
queue)
  # Weave annotation queue for the packet: one neutral call per item; the call map goes to key/.
  run uv run python scripts/weave_review_queue.py create --packet "$CAL/packet" --key-dir "$CAL/key" --name "bufo review packet v3" ;;
pull)
  # The reviewer's Weave annotations as a decisions file, then step7 reads it.
  run uv run python scripts/weave_review_queue.py pull --packet "$CAL/packet" --key-dir "$CAL/key" --out "$CAL/decisions-user.json" ;;
dev-rubric)
  # Rubric development check: re-judge only the 40 packet-v3 generations under the given rubric
  # (into astra-<suffix>/, no publish), re-key the packet with those verdicts, and compare against
  # the user's existing labels. The packet is a development set from here on.
  [ $# -eq 2 ] || [ $# -eq 3 ] || { echo "usage: $0 dev-rubric <bufo-astra-vN> [k]"; exit 2; }
  K=${3:-1}
  SUF=${2##*-}
  [ "$K" = 1 ] || SUF="$SUF-k$K"
  python3 - <<'PY'
import json
key = json.load(open("runs/distill/calibration-v1/key/unblinding-key.json"))
for run, arms in (("arms-v1", {"closed_quality_bar"}), ("arms-v1-students", {"open_base", "iter2_e3"})):
    ids = sorted(k["generation_id"] for k in key if k["backend"] in arms)
    open(f"runs/distill/calibration-v1/key/packet-v3-ids-{run}.txt", "w").write("\n".join(ids) + "\n")
PY
  run uv run bufogen --stage benchmark-score --output runs/distill/arms-v1 --judge astra --rubric "$2" --judge-samples "$K" --references runs/references/distill-v1 --generation-ids-file "$CAL/key/packet-v3-ids-arms-v1.txt"
  run uv run bufogen --stage benchmark-score --output runs/distill/arms-v1-students --judge astra --rubric "$2" --judge-samples "$K" --references runs/references/distill-v1 --generation-ids-file "$CAL/key/packet-v3-ids-arms-v1-students.txt"
  run uv run python scripts/rekey_packet.py --key "$CAL/key/unblinding-key.json" --scored "runs/distill/arms-v1/astra-$SUF/scored.jsonl" --scored "runs/distill/arms-v1-students/astra-$SUF/scored.jsonl" --out "$CAL/key/unblinding-key-$SUF.json" --packet "$CAL/packet"
  run uv run python scripts/review_agreement.py --packet "$CAL/packet" --key "$CAL/key/unblinding-key-$SUF.json" --decisions "$CAL/decisions-user.json" --reviewer "user-$SUF" ;;
rejudge)
  # Full re-judge under a revised rubric: Sol run 2 (paired against itself as baseline), the two
  # student arms, and the 24-item self-equal check, all published to Weave under names carrying
  # the rubric suffix. Writes astra-<suffix>/ and astra-self-<suffix>/; v1 directories untouched.
  [ $# -eq 3 ] || { echo "usage: $0 rejudge <bufo-astra-vN> <k>"; exit 2; }
  run uv run bufogen --stage benchmark-score --output runs/distill/arms-v1 --judge astra --rubric "$2" --judge-samples "$3" --references runs/references/distill-v1 --baseline-arm closed_quality_bar --publish
  run uv run bufogen --stage benchmark-score --output runs/distill/arms-v1-students --judge astra --rubric "$2" --judge-samples "$3" --references runs/references/distill-v1 --publish
  run uv run bufogen --stage benchmark-score --output runs/distill/sol-run1 --judge astra --rubric "$2" --judge-samples "$3" --self-reference --prompt-ids-file "$CAL/self-equal-ids.txt" --publish ;;
packet-v4)
  # Fresh blinded confirmation packet from validation rows the user has not labelled or seen
  # named, keyed with the revised judge's verdicts; key and queue record stay outside the packet.
  [ $# -eq 4 ] || { echo "usage: $0 packet-v4 <bufo-astra-vN> <k> <seed>"; exit 2; }
  SUF=${2##*-}; [ "$3" = 1 ] || SUF="$SUF-k$3"
  run uv run python scripts/make_review_packet.py --out "$CAL/packet-v4" --key-dir "$CAL/key-v4" --seed "$4" --scored-dir "astra-$SUF" --exclude-ids-file "$CAL/exclude-disclosed.txt" --exclude-ids-file "$CAL/exclude-packet-v3.txt"
  run uv run python scripts/weave_review_queue.py create --packet "$CAL/packet-v4" --key-dir "$CAL/key-v4" --name "bufo review packet v4" ;;
paired)
  # Student arms against Sol run 2 on validation; the arms live in different run directories.
  for arm in open_base iter2_e3; do
    run uv run python scripts/paired_across_runs.py --left runs/distill/arms-v1-students/astra --left-arm "$arm" --right runs/distill/arms-v1/astra --right-arm closed_quality_bar --out "$CAL/paired-validation-$arm.json"
  done ;;
*) echo "unknown step"; exit 2 ;;
esac
