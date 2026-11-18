import json

import pytest

from bufogen.curation.review_packet import PACKET_VERSION, agreement_report
from bufogen.curation.review_queue import (
    FEEDBACK_TYPES,
    SPECS,
    call_ref,
    check_dirs,
    decisions_from_feedback,
    numbered_requirements,
    parse_missing,
)

ITEMS = [
    {"index": 0, "label": "Item 01", "requirements": ["a", "b", "c"]},
    {"index": 1, "label": "Item 02", "requirements": ["x"]},
]
QUEUE = {
    "project": "ent/proj",
    "packet_sha256": "abc",
    "calls": [{"index": 0, "call_id": "c0"}, {"index": 1, "call_id": "c1"}],
}


def fb(call_id, kind, value, when="2026-10-22T10:00:00Z"):
    return {
        "weave_ref": call_ref("ent/proj", call_id),
        "feedback_type": FEEDBACK_TYPES[kind],
        "payload": {"value": value},
        "created_at": when,
    }


def test_parse_missing_accepts_numbers_and_none_words():
    assert parse_missing("1,3", 3) == ([0, 2], "")
    assert parse_missing(" 3 1 ", 3) == ([0, 2], "")
    assert parse_missing("2;2", 3) == ([1], "")
    for word in ("", None, "none", "None", "-", "0", "n/a"):
        assert parse_missing(word, 3) == ([], "")


def test_parse_missing_keeps_free_text_as_note():
    assert parse_missing("2,3(bad airplane), 5", 5) == ([1, 2, 4], "bad airplane")
    assert parse_missing("1 and 2", 3) == ([0, 1], "and")
    assert parse_missing("planes weird in A and B", 4) == ([], "planes weird in A and B")


@pytest.mark.parametrize("text", ["4", "0,4", "7"])
def test_parse_missing_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_missing(text, 3)


def test_numbered_requirements():
    assert numbered_requirements(["a", "b"]) == "1. a\n2. b"


def test_specs_are_strings_the_ui_can_render():
    for spec in SPECS.values():
        assert spec["field_schema"]["type"] == "string"
        assert "arm" not in spec["description"].lower()


def test_decisions_from_feedback_uses_latest_value_per_field():
    rows = [
        fb("c0", "pass", "pass", "2026-10-22T09:00:00Z"),
        fb("c0", "pass", "fail", "2026-10-22T11:00:00Z"),
        fb("c0", "vs_reference", "worse"),
        fb("c0", "missing", "1, 3"),
        fb("c1", "pass", "pass"),
        fb("c1", "vs_reference", "equal"),
        fb("c1", "missing", "none"),
        {
            "weave_ref": call_ref("ent/proj", "c1"),
            "feedback_type": "wandb.runnable.x",
            "payload": {"output": 1},
            "created_at": "2026-10-22T12:00:00Z",
        },
    ]
    payload = decisions_from_feedback(ITEMS, QUEUE, rows, packet_sha256="abc")
    assert payload["version"] == PACKET_VERSION and payload["packet_sha256"] == "abc"
    assert payload["decisions"] == [
        {"index": 0, "pass": False, "vs_reference": "worse", "missing": [0, 2], "note": ""},
        {"index": 1, "pass": True, "vs_reference": "equal", "missing": [], "note": ""},
    ]


def test_decisions_from_feedback_treats_absent_missing_as_none():
    rows = [
        fb("c0", "pass", "pass"),
        fb("c0", "vs_reference", "equal"),
        fb("c1", "pass", "fail"),
        fb("c1", "vs_reference", "worse"),
    ]
    payload = decisions_from_feedback(ITEMS, QUEUE, rows, packet_sha256="abc")
    assert [d["missing"] for d in payload["decisions"]] == [[], []]


def test_decisions_from_feedback_reports_unlabelled_items():
    rows = [fb("c0", "pass", "pass"), fb("c0", "vs_reference", "equal")]
    with pytest.raises(ValueError, match="1 of 2 items unlabelled: \\['Item 02'\\]"):
        decisions_from_feedback(ITEMS, QUEUE, rows, packet_sha256="abc")


def test_decisions_from_feedback_rejects_other_packet():
    with pytest.raises(ValueError, match="does not match"):
        decisions_from_feedback(ITEMS, QUEUE, [], packet_sha256="other")


def test_check_dirs_keeps_record_out_of_packet(tmp_path):
    with pytest.raises(ValueError):
        check_dirs(tmp_path / "packet", tmp_path / "packet" / "key")
    check_dirs(tmp_path / "packet", tmp_path / "key")


def test_pulled_decisions_feed_agreement_report(tmp_path):
    packet, key = tmp_path / "packet", tmp_path / "key"
    packet.mkdir(), key.mkdir()
    items = [
        {
            "index": i,
            "label": f"Item 0{i + 1}",
            "prompt": "p",
            "requirements": ["r1", "r2"],
            "candidate": f"images/item-0{i + 1}.png",
            "reference": f"images/ref-0{i + 1}.png",
        }
        for i in range(2)
    ]
    items_bytes = json.dumps(items).encode()
    (packet / "items.json").write_bytes(items_bytes)
    import hashlib

    sha = hashlib.sha256(items_bytes).hexdigest()
    (key / "unblinding-key.json").write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "backend": "x",
                    "judge_strict_pass": True,
                    "judge_vs_reference": "equal",
                    "packet_sha256": sha,
                },
                {
                    "index": 1,
                    "backend": "y",
                    "judge_strict_pass": False,
                    "judge_vs_reference": "worse",
                    "packet_sha256": sha,
                },
            ]
        )
    )
    queue = dict(QUEUE, packet_sha256=sha)
    rows = [
        fb("c0", "pass", "pass"),
        fb("c0", "vs_reference", "equal"),
        fb("c0", "missing", ""),
        fb("c1", "pass", "fail"),
        fb("c1", "vs_reference", "worse"),
        fb("c1", "missing", "2"),
    ]
    payload = decisions_from_feedback(items, queue, rows, packet_sha256=sha)
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(payload))
    report = agreement_report(
        packet, decisions, reviewer="user", key_path=key / "unblinding-key.json"
    )
    assert report["strict_pass_agreement"] == 1.0 and report["gate"]["passed"]
    assert not (packet / "weave-queue.json").exists()
