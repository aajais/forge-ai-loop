import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rekey_packet", ROOT / "scripts/rekey_packet.py")
rekey_packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rekey_packet)


def scored_row(gid, strict, vs, version="bufo-astra-v2", status="ok"):
    return json.dumps(
        {
            "id": gid,
            "scores": {
                "astra_judge": {
                    "status": status,
                    "strict_pass": strict,
                    "vs_reference": vs,
                    "rubric_version": version,
                }
            },
        }
    )


def test_rekey_replaces_verdicts_and_counts_changes(tmp_path):
    key_dir = tmp_path / "key"
    key_dir.mkdir()
    key = key_dir / "unblinding-key.json"
    key.write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "generation_id": "g0",
                    "backend": "a",
                    "judge_strict_pass": True,
                    "judge_vs_reference": "equal",
                    "packet_sha256": "s",
                },
                {
                    "index": 1,
                    "generation_id": "g1",
                    "backend": "b",
                    "judge_strict_pass": False,
                    "judge_vs_reference": "worse",
                    "packet_sha256": "s",
                },
            ]
        )
    )
    scored = tmp_path / "scored.jsonl"
    scored.write_text(
        scored_row("g0", False, "worse") + "\n" + scored_row("g1", False, "worse") + "\n"
    )
    out = key_dir / "unblinding-key-v2.json"
    result = rekey_packet.rekey(key, [scored], out)
    assert result == {"n": 2, "strict_verdicts_changed": 1, "out": str(out)}
    rekeyed = json.loads(out.read_text())
    assert rekeyed[0]["judge_strict_pass"] is False and rekeyed[0]["judge_vs_reference"] == "worse"
    assert all(
        k["rubric_version"] == "bufo-astra-v2" and k["packet_sha256"] == "s" for k in rekeyed
    )


def test_rekey_refuses_missing_rows_mixed_versions_and_other_dirs(tmp_path):
    key_dir = tmp_path / "key"
    key_dir.mkdir()
    key = key_dir / "unblinding-key.json"
    key.write_text(json.dumps([{"index": 0, "generation_id": "g0", "packet_sha256": "s"}]))
    scored = tmp_path / "scored.jsonl"
    scored.write_text(scored_row("other", True, "equal") + "\n")
    with pytest.raises(ValueError, match="no verdict"):
        rekey_packet.rekey(key, [scored], key_dir / "k2.json")
    scored.write_text(
        scored_row("g0", True, "equal")
        + "\n"
        + scored_row("g0", True, "equal", "bufo-astra-v1")
        + "\n"
    )
    with pytest.raises(ValueError, match="one rubric version"):
        rekey_packet.rekey(key, [scored], key_dir / "k2.json")
    scored.write_text(scored_row("g0", True, "equal") + "\n")
    with pytest.raises(ValueError, match="beside the original key"):
        rekey_packet.rekey(key, [scored], tmp_path / "elsewhere.json")
