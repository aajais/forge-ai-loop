"""Record root's actual visual inspection of the four validation contact sheets."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path("runs/forge-gap/review/reviewer-root")
inputs = json.loads((root / "inputs.json").read_text())
evidence = [
    "Both raised hands overlap the bottoms of two orange batons; airplane behind intact Bufo. Confirmed full PNG hand contacts.",
    "Two orange batons held in raised hands; airplane behind; blue vest attached and head/hands in frame.",
    "Each raised hand grips an orange baton, airplane behind; entire body and both feet visible.",
    "Main Bufo holds two batons; smaller idle observer at viewer right; plane behind; exactly two Bufos.",
    "Two held orange batons and plane visible; foreground rail crosses lower legs, leaving face and hands clear.",
    "Two held orange batons and plane behind; upper sign reads READY; continuous recognizable Bufo body.",
    "Raised hand grips red flower stem entering blue vase; lower hand touches vase; coherent florist scene.",
    "One hand grips stem into blue vase, other rests on vase; blue vest attached; head and hands inside frame.",
    "Red flower stem visibly enters blue vase with separate hand contacts; whole body and both feet shown.",
    "Reject: red flower is held aloft but its stem does not visibly extend into the vase. Raised forearm passes in front of vase instead. Full PNG confirms missing stem-to-vase relationship; right observer is present.",
    "Stem extends from red bloom through gripping hand into blue vase; other hand touches vase; rail covers lower legs.",
    "Red bloom stem enters blue vase; other hand touches vase; upper sign reads READY.",
    "One hand grips cone, other holds scoop utensil against pink ice cream atop cone; blue ice-cream cart visible.",
    "Cone and scoop utensil held separately, pink ice cream touches cone; blue vest and cart visible, hands/head inside frame.",
    "Cone gripped and scoop utensil held against pink ice cream; cart visible; both feet and whole body in frame.",
    "Main Bufo holds scoop utensil and cone with pink ice cream; smaller idle Bufo at right, cart behind; exactly two characters.",
    "Cone gripped and scoop handle held, pink ice cream on cone; cart visible; rail obscures lower legs without hiding hands.",
    "Pink ice cream sits on held cone with scoop utensil over it; cart present and upper sign reads READY.",
    "Gold key gripped at bow, tip enters blue padlock keyhole; other hand grips lock; recognizable continuous Bufo.",
    "Gold key held and inserted into blue padlock, other hand on lock; blue vest attached and hands/head in frame.",
    "Held gold key enters held blue padlock; complete Bufo body and both feet visible in frame.",
    "Main Bufo inserts held gold key into blue padlock; small idle observer on viewer right; exactly two Bufos.",
    "Held key tip enters blue padlock with other hand gripping lock; foreground rail covers lower legs, face/action clear.",
    "Held gold key enters held blue padlock; upper blue sign reads READY; face and action remain clear.",
]
assert len(inputs) == len(evidence) == 24
out = root / "decisions.jsonl"
assert not out.exists(), "Never overwrite review decisions silently"
reviews = []
for item, note in zip(inputs, evidence, strict=True):
    assert hashlib.sha256(Path(item["png_path"]).read_bytes()).hexdigest() == item["png_sha256"]
    accept = item["index"] != 9
    reviews.append(
        {
            "schema_version": 1,
            "source_generation_id": item["source_generation_id"],
            "prompt_id": item["prompt_id"],
            "partition": "validation",
            "prompt_family": item["family"],
            "decision": "accept" if accept else "reject",
            "reviewer_kind": "assistant",
            "reviewer_id": "codex/reviewer-root",
            "source_ref": item["png_path"],
            "evidence": note,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "visual_quality_pass": accept,
            "png_sha256": item["png_sha256"],
        }
    )
out.write_text("".join(json.dumps(r) + "\n" for r in reviews))
print(json.dumps({"reviewed": len(reviews), "accepted": 23, "rejected": 1}))
