import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from test_review_packet import decisions_for, make_rows, write

from bufogen.curation.review_packet import create_packet, select_review_rows

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/review_agreement.py"
spec = importlib.util.spec_from_file_location("review_agreement", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_review_agreement_takes_paths_and_reviewer_as_arguments(tmp_path, capsys):
    rows = select_review_rows(make_rows(tmp_path), size=40)
    out = tmp_path / "packet"
    create_packet(rows, out, key_dir=tmp_path / "key")
    decisions = write(tmp_path, decisions_for(out))
    mod.main(
        [
            "--packet",
            str(out),
            "--key",
            str(tmp_path / "key" / "unblinding-key.json"),
            "--decisions",
            str(decisions),
            "--reviewer",
            "user",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["gate"]["passed"] is True
    assert set(printed) == {"reviewer", "n", "strict_pass_agreement", "gate", "report"}
    assert (tmp_path / "key" / "agreement-user.json").exists()
    with pytest.raises(ValueError, match="reviewer"):
        mod.main(
            [
                "--packet",
                str(out),
                "--key",
                str(tmp_path / "key" / "unblinding-key.json"),
                "--decisions",
                str(decisions),
                "--reviewer",
                "x'); import os; ('",
            ]
        )


def test_stage_a_script_parses_and_is_strict():
    script = ROOT / "scripts/distill_stage_a.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "set -euo pipefail" in text
    assert "eval" not in text
    assert "scripts/review_agreement.py" in text


def test_stage_a_run_logs_failure_then_stops_paired_loop(tmp_path):
    # A copy of the script in a scratch tree, with a fake `uv` that always fails.
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts/distill_stage_a.sh"
    script.write_text((ROOT / "scripts/distill_stage_a.sh").read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "uv"
    fake.write_text('#!/bin/sh\necho "$@" >> "$(dirname "$0")/calls"\nexit 3\n')
    fake.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(["bash", str(script), "paired"], env=env, capture_output=True)
    assert result.returncode == 3
    log = (tmp_path / "runs/distill/stage-a-log.md").read_text()
    assert log.count("START") == 1 and "exit: 3" in log
    assert len((bin_dir / "calls").read_text().splitlines()) == 1
    # Reviewer and decisions reach Python as argv, never interpolated into code.
    result = subprocess.run(
        ["bash", str(script), "step7", "a b", "/x/'quoted' path.json"],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 3
    call = (bin_dir / "calls").read_text().splitlines()[-1]
    assert call.startswith("run python scripts/review_agreement.py ")
    assert "--reviewer a b" in call and "--decisions /x/'quoted' path.json" in call
