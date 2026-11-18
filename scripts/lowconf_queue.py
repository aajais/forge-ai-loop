"""Build a neutral Weave review queue of low-confidence judge rows (status ok, confidence < 0.6).

Packet: runs/distill/stage-b/lowconf/packet; unblinding key (outside the packet):
runs/distill/stage-b/lowconf/key. With zero such rows, writes NONE.txt and no queue.
"""

import hashlib
import json
import re
from pathlib import Path

from bufogen.curation.judge_reviews import is_low_confidence
from bufogen.curation.review_packet import create_packet, load_scored_rows

STAGE_B = Path("runs/distill/stage-b")
SCORED = "astra-v5-k3"


def select_lowconf(rows: list) -> list:
    """Rows whose judge status is ok and whose confidence is a number below the floor."""
    return [row for row in rows if is_low_confidence(row)]


def packet_digest(packet: Path) -> str | None:
    """sha256 over the packet's relative file names and bytes (digest only; contents stay private)."""
    if not packet.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(p for p in packet.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(packet)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    root = STAGE_B
    lowconf, scored_total, scanned = [], 0, 0
    dirs = sorted(
        (p for p in (root / "sol-data").glob("attempt-*") if re.fullmatch(r"attempt-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[1]),
    )
    for d in dirs:
        if not (d / SCORED / "scored.jsonl").exists():
            continue
        rows = load_scored_rows(d, SCORED)
        scanned += 1
        scored_total += len(rows)
        lowconf.extend(select_lowconf(rows))
    out = root / "lowconf"
    out.mkdir(parents=True, exist_ok=True)
    none = out / "NONE.txt"
    if lowconf:
        none.unlink(missing_ok=True)
    else:
        (out / "NONE.txt").write_text(
            f"attempts scanned: {scanned}\nscored rows: {scored_total}\nlow-confidence rows: 0\n"
        )
        (out / "summary.json").write_text(
            json.dumps({"count": 0, "attempts_scanned": scanned}, indent=2) + "\n"
        )
        print("no low-confidence rows; wrote NONE.txt")
        return
    from bufogen.curation.review_queue import create_queue

    packet, key = out / "packet", out / "key"
    create_packet(lowconf, packet, key_dir=key)
    queue = create_queue(packet, key, name="bufo low-confidence review (stage B)")
    summary = {
        "count": len(lowconf),
        "attempts_scanned": scanned,
        "queue_id": queue["queue_id"],
        "queue_url": queue["url"],
        "packet_sha256": packet_digest(packet),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": len(lowconf), "queue_id": queue["queue_id"], "url": queue["url"]}))


if __name__ == "__main__":
    main()
