import json
import subprocess
from pathlib import Path

from PIL import Image

from bufogen.evaluation.references import build_reference_manifest
from bufogen.generation import BufoGeneration, Manifest

ROOT = Path(__file__).resolve().parents[1]
STAGE_B = ROOT / "runs" / "distill" / "stage-b"


def test_heldout_dirs_hold_no_astra_output():
    """Heldout drawings stay unjudged until the single confirmation run (plan Task 10).

    The Sol heldout arm is judged exactly once, by the confirmation run (its astra-v5-k3
    directory, recorded in the stage log); every other heldout directory stays unjudged.
    """
    confirmed = (STAGE_B / "confirmation/astra-v5-k3/summary.json").is_file()
    for directory in sorted(STAGE_B.glob("heldout-*")):
        allowed = (
            directory / "astra-v5-k3" if confirmed and directory.name == "heldout-sol-arm" else None
        )
        for found in list(directory.rglob("astra*")) + list(directory.rglob("scored.jsonl")):
            assert allowed is not None and (found == allowed or allowed in found.parents), found
        if allowed is not None:
            continue
        for manifest in directory.rglob("manifest.jsonl"):
            for line in manifest.read_text().splitlines():
                assert not json.loads(line).get("scores"), (manifest, "judged heldout row")


def test_build_references_does_not_judge(tmp_path):
    run = tmp_path / "run"
    (run / "pngs").mkdir(parents=True)
    (run / "benchmark.json").write_text(
        json.dumps({"run_id": "r1", "configuration_hash": "abc", "configuration": {}})
    )
    image = run / "pngs/h.png"
    Image.new("RGB", (4, 4), "red").save(image)
    Manifest(run / "manifest.jsonl").append(
        BufoGeneration(
            prompt_id="h",
            prompt="p",
            backend="sol",
            model="m",
            group_id="g",
            svg="",
            valid=True,
            latency_s=0.0,
            png_path=str(image),
            benchmark_metadata={"status": "ok", "split": "heldout", "family": "f"},
        )
    )
    manifest = build_reference_manifest(run, tmp_path / "out", backend="sol")
    assert manifest["rows"]["h"]
    assert all("scores" not in entry for entry in manifest["rows"].values())
    assert not list((tmp_path / "out").rglob("astra*"))


def test_stage_b_script_is_strict_and_defines_heldout_steps():
    script = ROOT / "scripts/distill_stage_b.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "set -euo pipefail" in text
    assert "heldout-ref)" in text and "heldout-sol)" in text
    # Train-data steps judge; no heldout step (or any text before data-gen) may.
    heldout_part = text.split("data-gen)")[0]
    assert "benchmark-score" not in heldout_part
    # The confirmation steps (last in the script) are the only ones after it that touch heldout.
    assert "heldout" not in text.split("data-gen)")[1].split("confirm-gen)")[0]


def test_stage_script_confirmation_steps_are_wired_safely():
    text = (ROOT / "scripts/distill_stage_b.sh").read_text()
    body = text.split("confirm-gen)")[1]
    score = body.split("confirm-score)")[1].split("confirm-pair)")[0]
    assert "--confirmation" in score and "--baseline-arm" not in score
    assert "--judge-samples 3" in score and "heldout-sol-arm" in score
    gen = body.split("confirm-score)")[0]
    assert "--split heldout" in gen and "--backends open_base" in gen
    assert "benchmark-score" not in gen and "--checkpoint" not in gen
    packet = body.split("confirm-packet)")[1]
    assert "--split heldout --confirmation" in packet and '--key-dir "$CONF/key"' in packet


def test_stage_script_rl_steps_use_the_per_label_base_and_agent_flags():
    text = (ROOT / "scripts/distill_stage_b.sh").read_text()
    update = text.split("rl-update)")[1].split("rl-eval)")[0]
    assert "--stage train-rl --reward astra" in update and "astra-calibration.json" in update
    assert '--base-model "$base_model"' in update and 'base_model_of "$base_arm"' in update
    assert '"$@"' in update and "--max-tokens 6000" not in update
    evaluation = text.split("rl-eval)")[1].split("confirm-gen)")[0]
    assert '--checkpoint "$label=$artifact"' in evaluation
    assert '--checkpoint "$sft_arm=$sft_artifact"' in evaluation and "--backends" not in evaluation
    assert '--right-arm "$sft_arm"' in evaluation
    assert "--judge-samples 3" in evaluation and "--rubric bufo-astra-v5" in evaluation
    assert "--references runs/references/distill-v2 --publish" in evaluation
    assert "paired_across_runs.py" in evaluation and '"$@"' in evaluation
    assert "heldout" not in update + evaluation and "--max-tokens 6000" not in evaluation
