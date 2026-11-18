"""Weave annotation queue for a blinded review packet.

The queue holds one neutral `bufo_review_item` call per packet item: label, prompt,
numbered requirements and the two drawings (A candidate, B reference). Nothing in the
call names an arm, model or judge verdict; the call-to-index map lives in the key
directory. The reviewer's annotations are pulled back into a `bufo-review-v2`
decisions file, which `agreement_report` reads exactly as it reads the HTML download.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from bufogen.curation.review_packet import PACKET_VERSION, _inside

OP_NAME = "bufo_review_item"
SPEC_VERSION = "v2"
SPECS = {
    "missing": {
        "name": f"bufo_review_missing_{SPEC_VERSION}",
        "description": (
            "Numbers of the requirements that drawing A does NOT meet, comma separated "
            "(for example 1,3). Write none if A meets all of them."
        ),
        "field_schema": {"type": "string", "maxLength": 40},
    },
    "pass": {
        "name": f"bufo_review_pass_{SPEC_VERSION}",
        "description": "Does drawing A pass as a usable Bufo for this prompt?",
        "field_schema": {"type": "string", "enum": ["pass", "fail"]},
    },
    "vs_reference": {
        "name": f"bufo_review_vs_reference_{SPEC_VERSION}",
        "description": "Drawing A compared with drawing B.",
        "field_schema": {"type": "string", "enum": ["better", "equal", "worse"]},
    },
}
FEEDBACK_TYPES = {k: f"wandb.annotation.{v['name']}" for k, v in SPECS.items()}
DISPLAY_FIELDS = ["inputs.label", "inputs.prompt", "inputs.requirements", "inputs.A", "inputs.B"]
GUIDELINES = (
    "A is the candidate, B is the reference. For each item: list the numbers of the "
    "requirements A does not meet (none if it meets all), say whether A passes, then "
    "compare A with B. The missing field is only for requirements that are literally unmet "
    "(the thing is not there or not doing what the requirement says); a badly drawn, oddly "
    "proportioned or awkwardly layered element that still meets the words is a pass/fail or "
    "comparison call, and you can note it in words after the numbers. Judge the drawing, not "
    "the style; B is one acceptable answer, not the only one."
)
_NONE_WORDS = {"", "none", "no", "n/a", "na", "-", "0"}


def numbered_requirements(requirements: list[str]) -> str:
    """One line per requirement, numbered from 1; a string renders expanded in the queue UI."""
    return "\n".join(f"{i + 1}. {text}" for i, text in enumerate(requirements))


def parse_missing(text: str | None, n_requirements: int) -> tuple[list[int], str]:
    """'1,3' -> ([0, 2], ''); 'none' -> ([], ''); '2,3(bad hat), 5' -> ([1, 2, 4], 'bad hat').

    Every integer in the text is a requirement number; whatever is left is the reviewer's
    note. Text with no number at all is a note with nothing missing. Raises ValueError only
    when a number is out of range.
    """
    raw = (text or "").strip()
    if raw.lower() in _NONE_WORDS:
        return [], ""
    numbers = [int(t) for t in re.findall(r"\d+", raw)]
    note = " ".join(re.sub(r"[\d,;()\[\]:]+", " ", raw).split())
    if not numbers:
        # Free text without a number is a note, not a requirement miss ("planes look odd").
        return [], note
    indices = sorted({n - 1 for n in numbers})
    if any(not 0 <= i < n_requirements for i in indices):
        raise ValueError(f"requirement number out of range 1..{n_requirements}: {text!r}")
    return indices, note


def call_ref(project: str, call_id: str) -> str:
    return f"weave:///{project}/call/{call_id}"


def _created(row: dict) -> datetime:
    value = row.get("created_at")
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def latest_values(feedback_rows: list[dict]) -> dict[tuple[str, str], str]:
    """(weave_ref, feedback_type) -> latest payload value, by created_at."""
    latest: dict[tuple[str, str], tuple[datetime, str]] = {}
    for row in feedback_rows:
        key = (row["weave_ref"], row["feedback_type"])
        value = (row.get("payload") or {}).get("value")
        if value is None:
            continue
        stamp = _created(row)
        if key not in latest or stamp >= latest[key][0]:
            latest[key] = (stamp, value)
    return {k: v for k, (_, v) in latest.items()}


def decisions_from_feedback(
    items: list[dict], queue: dict, feedback_rows: list[dict], *, packet_sha256: str
) -> dict:
    """Build a bufo-review-v2 decisions payload; raises ValueError listing unlabelled items."""
    if queue.get("packet_sha256") != packet_sha256:
        raise ValueError("queue record does not match this packet")
    values = latest_values(feedback_rows)
    by_index = {item["index"]: item for item in items}
    decisions, incomplete = [], []
    for entry in queue["calls"]:
        index, ref = entry["index"], call_ref(queue["project"], entry["call_id"])
        item = by_index[index]
        got = {k: values.get((ref, t)) for k, t in FEEDBACK_TYPES.items()}
        if got["pass"] not in ("pass", "fail") or got["vs_reference"] not in (
            "better",
            "equal",
            "worse",
        ):
            incomplete.append(item["label"])
            continue
        missing, note = parse_missing(got["missing"], len(item["requirements"]))
        decisions.append(
            {
                "index": index,
                "pass": got["pass"] == "pass",
                "vs_reference": got["vs_reference"],
                "missing": missing,
                "note": note,
            }
        )
    if incomplete:
        raise ValueError(f"{len(incomplete)} of {len(items)} items unlabelled: {incomplete}")
    decisions.sort(key=lambda d: d["index"])
    return {"version": PACKET_VERSION, "packet_sha256": packet_sha256, "decisions": decisions}


def packet_sha256(packet_dir: Path) -> str:
    return hashlib.sha256((Path(packet_dir) / "items.json").read_bytes()).hexdigest()


def queue_record_path(key_dir: Path) -> Path:
    return Path(key_dir) / "weave-queue.json"


def check_dirs(packet_dir: Path, key_dir: Path) -> None:
    if _inside(key_dir, packet_dir):
        raise ValueError("key_dir must not be inside the packet directory")


# --- Weave adapters (network) -------------------------------------------------------------


def _review_item_op():
    import weave

    @weave.op(name=OP_NAME)
    def bufo_review_item(label: str, prompt: str, requirements: str, A, B) -> dict:  # noqa: N803
        return {"status": "awaiting review"}

    return bufo_review_item


def create_queue(packet_dir: Path, key_dir: Path, *, name: str) -> dict:
    """Publish specs, log one neutral call per item, create the queue; record goes to key_dir."""
    import weave
    from PIL import Image
    from weave.flow.annotation_spec import AnnotationSpec
    from weave.trace_server.trace_server_interface import (
        AnnotationQueueAddCallsReq,
        AnnotationQueueCreateReq,
    )

    from bufogen.integrations.telemetry import init_weave
    from bufogen.settings import project_path

    packet_dir, key_dir = Path(packet_dir), Path(key_dir)
    check_dirs(packet_dir, key_dir)
    record = queue_record_path(key_dir)
    if record.exists():
        raise FileExistsError(f"{record} exists; one queue per packet")
    sha = packet_sha256(packet_dir)
    items = json.loads((packet_dir / "items.json").read_text())
    client = init_weave()
    project = project_path()
    spec_refs = []
    for spec in SPECS.values():
        ref = weave.publish(AnnotationSpec(**spec), name=spec["name"])
        spec_refs.append(ref.uri())
    op = _review_item_op()
    calls = []
    for item in items:
        a = Image.open(packet_dir / item["candidate"]).convert("RGBA")
        b = Image.open(packet_dir / item["reference"]).convert("RGBA")
        _, call = op.call(
            item["label"], item["prompt"], numbered_requirements(item["requirements"]), a, b
        )
        calls.append({"index": item["index"], "call_id": call.id})
    client.flush()  # calls are batched; the queue can only reference calls the server has
    created = client.server.annotation_queue_create(
        AnnotationQueueCreateReq(
            project_id=project, name=name, description=GUIDELINES, scorer_refs=spec_refs
        )
    )
    added = client.server.annotation_queue_add_calls(
        AnnotationQueueAddCallsReq(
            project_id=project,
            queue_id=created.id,
            call_ids=[c["call_id"] for c in calls],
            display_fields=DISPLAY_FIELDS,
        )
    )
    if added.added_count != len(calls):
        raise RuntimeError(f"queue got {added.added_count} of {len(calls)} calls")
    entity, proj = project.split("/", 1)
    out = {
        "project": project,
        "queue_id": created.id,
        "name": name,
        "url": f"https://wandb.ai/{entity}/{proj}/weave/annotation-queues/{created.id}",
        "packet_sha256": sha,
        "spec_refs": spec_refs,
        "added": added.added_count,
        "calls": calls,
    }
    key_dir.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(out, indent=2) + "\n")
    return out


def fetch_feedback(queue: dict) -> list[dict]:
    from weave.trace_server.interface.query import Query
    from weave.trace_server.trace_server_interface import FeedbackQueryReq

    from bufogen.integrations.telemetry import init_weave

    client = init_weave()
    refs = [call_ref(queue["project"], c["call_id"]) for c in queue["calls"]]
    query = Query(
        **{
            "$expr": {
                "$in": [{"$getField": "weave_ref"}, [{"$literal": r} for r in refs]],
            }
        }
    )
    res = client.server.feedback_query(
        FeedbackQueryReq(project_id=queue["project"], query=query, limit=10_000)
    )
    return [
        r for r in res.result if str(r.get("feedback_type", "")).startswith("wandb.annotation.")
    ]


def pull_decisions(packet_dir: Path, key_dir: Path, out: Path) -> dict:
    packet_dir, key_dir = Path(packet_dir), Path(key_dir)
    check_dirs(packet_dir, key_dir)
    queue = json.loads(queue_record_path(key_dir).read_text())
    items = json.loads((packet_dir / "items.json").read_text())
    payload = decisions_from_feedback(
        items, queue, fetch_feedback(queue), packet_sha256=packet_sha256(packet_dir)
    )
    Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    return {"n": len(payload["decisions"]), "out": str(out)}
