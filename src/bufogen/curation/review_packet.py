"""Blinded one-page human review packet and judge agreement report."""

import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from bufogen.generation import BufoGeneration

PACKET_VERSION = "bufo-review-v2"
AGREEMENT_THRESHOLD = 0.8
_VS = ("better", "equal", "worse")
_PATH_FIELDS = ("png_path", "svg_path", "input_image_path")


def freeze(path: Path, data: bytes) -> None:
    """Write once; rewriting with different bytes is an error."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen packet differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _save_json(path: Path, value) -> bytes:
    data = (json.dumps(value, indent=2) + "\n").encode()
    freeze(path, data)
    return data


def load_scored_rows(run_dir: Path, scored_dir: str = "astra") -> list[BufoGeneration]:
    """Read <run>/<scored_dir>/scored.jsonl, resolving relative paths against the run directory.

    Scored rows keep the manifest's png_path spelling, which is relative to the run
    directory; Manifest.read() would resolve it against the scored directory instead.
    scored_dir names the judge run (astra, astra-v4-k3, ...); it must be a plain name.
    """
    if not re.fullmatch(r"astra(-self)?(-v\d+)?(-k\d+)?", scored_dir):
        raise ValueError(f"scored_dir must name an astra scoring directory, got {scored_dir!r}")
    run_dir = Path(run_dir).resolve()
    path = run_dir / scored_dir / "scored.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = BufoGeneration(**json.loads(line))
        for name in _PATH_FIELDS:
            value = getattr(row, name)
            if value and not Path(value).is_absolute():
                setattr(row, name, str((run_dir / value).resolve()))
        reference = (row.benchmark_metadata or {}).get("reference")
        if (
            reference
            and reference.get("png_path")
            and not Path(reference["png_path"]).is_absolute()
        ):
            reference["png_path"] = str((run_dir / reference["png_path"]).resolve())
        rows.append(row)
    return rows


def load_scored_runs(run_dirs, scored_dir: str = "astra") -> list[BufoGeneration]:
    """Scored rows of several run directories (one per confirmation arm), deduped by id.

    Arm identity stays on each row's backend, which only the unblinding key records.
    The first run that carries a generation id wins.
    """
    rows, seen = [], set()
    for run_dir in run_dirs:
        for row in load_scored_rows(run_dir, scored_dir):
            if row.id not in seen:
                seen.add(row.id)
                rows.append(row)
    return rows


def _judge(row: BufoGeneration) -> dict:
    return row.scores.get("astra_judge") or {}


def _eligible(row: BufoGeneration) -> bool:
    reference = (row.benchmark_metadata or {}).get("reference") or {}
    return bool(
        row.png_path
        and Path(row.png_path).is_file()
        and reference.get("png_path")
        and Path(reference["png_path"]).is_file()
        and _judge(row).get("status") == "ok"
        # A row whose generation did not complete fails strict_pass whatever it shows,
        # so a reviewer could only disagree with it for a non-visual reason.
        and (row.benchmark_metadata or {}).get("status") == "completed"
    )


CONFIRMATION_SPLITS = frozenset({"heldout", "test"})


def select_review_rows(
    rows: list[BufoGeneration],
    *,
    size: int = 40,
    seed: int = 20261021,
    confirmation: bool = False,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> list[BufoGeneration]:
    """Stratified review sample; heldout/test rows need confirmation=True.

    ``exclude_ids`` removes generations that were disclosed to the reviewer (for example
    through an earlier packet's write-up) before sampling.
    """
    if size < 4:
        raise ValueError("size must be >= 4")
    exclude_ids = set(exclude_ids)
    rows = [r for r in rows if r.id not in exclude_ids]
    if not confirmation and any(
        (r.benchmark_metadata or {}).get("split") in CONFIRMATION_SPLITS for r in rows
    ):
        raise ValueError(
            "rows include a heldout/test split; reviewing them is reserved for the "
            "confirmation run (confirmation=True)"
        )
    eligible = sorted((r for r in rows if _eligible(r)), key=lambda r: r.id)
    if len(eligible) < size:
        raise ValueError(f"need {size} eligible rows, found {len(eligible)}")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    arms = sorted({r.backend for r in eligible})
    arm_min = size // (2 * len(arms))
    class_min = math.ceil(0.4 * size)

    chosen: list[BufoGeneration] = []
    remaining = list(eligible)

    def passed(row):
        return bool(_judge(row).get("strict_pass"))

    def take(row):
        chosen.append(row)
        remaining.remove(row)

    for arm in arms:
        for _ in range(arm_min):
            candidates = [r for r in remaining if r.backend == arm]
            if not candidates:
                break
            n_pass = sum(passed(r) for r in chosen)
            want_pass = n_pass <= len(chosen) - n_pass
            same = [r for r in candidates if passed(r) == want_pass]
            take((same or candidates)[0])
    # Each phase is capped at the room left, so no row added for a class minimum is
    # cut afterwards (the arm phase takes at most size / 2 rows).
    for want in (True, False):
        have = sum(passed(r) == want for r in chosen)
        room = size - len(chosen)
        for row in [r for r in remaining if passed(r) == want][
            : min(room, max(0, class_min - have))
        ]:
            take(row)
    chosen.extend(remaining[: size - len(chosen)])
    if len(chosen) != size:
        raise AssertionError(f"selection produced {len(chosen)} rows, expected {size}")
    return chosen


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image review</title>
<style>
body{font-family:system-ui,sans-serif;margin:0 auto;max-width:960px;padding:12px 12px 80px;background:#fafafa;color:#111}
.item{background:#fff;border:1px solid #ccc;border-radius:8px;padding:12px;margin:16px 0}
.pair{display:flex;gap:12px;flex-wrap:wrap}
.pair figure{flex:1 1 280px;margin:0;text-align:center}
.pair img{max-width:100%;height:auto;background:#fff;border:1px solid #ddd}
fieldset{border:0;padding:0;margin:8px 0}
textarea{width:100%;box-sizing:border-box}
footer{position:fixed;left:0;right:0;bottom:0;background:#222;color:#fff;padding:10px 16px;display:flex;justify-content:space-between;align-items:center}
button{font-size:1rem;padding:8px 14px}
@media (max-width:600px){.pair{flex-direction:column}}
</style></head><body>
<h1>Image review</h1>
<p>For each item, A is the candidate and B is the reference. Tick every requirement that A does NOT meet (ticked means missing), judge whether A passes, then compare A with B.</p>
<form id="review" autocomplete="off" onsubmit="return false"><div id="items"></div></form>
<footer><span id="progress"></span><button id="download" disabled>Download decisions</button></footer>
<script type="application/json" id="packet-data">__ITEMS__</script>
<script>
"use strict";
var PACKET_SHA256 = "__SHA__";
var items = JSON.parse(document.getElementById("packet-data").textContent);
function el(tag, text, cls) {
  var e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
function radios(item, name, options, labelFor) {
  var box = document.createElement("fieldset");
  box.appendChild(el("legend", labelFor));
  options.forEach(function (opt) {
    var label = document.createElement("label");
    var input = document.createElement("input");
    input.type = "radio";
    input.name = name + "-" + item.index;
    input.value = opt.value;
    input.addEventListener("change", function () {
      update();
    });
    label.appendChild(input);
    label.appendChild(document.createTextNode(" " + opt.text + "  "));
    box.appendChild(label);
  });
  return box;
}
function figure(src, caption) {
  var f = document.createElement("figure");
  var img = document.createElement("img");
  img.src = src;
  img.alt = caption;
  f.appendChild(img);
  f.appendChild(el("figcaption", caption));
  return f;
}
items.forEach(function (item) {
  var card = el("section", undefined, "item");
  card.appendChild(el("h2", item.label));
  card.appendChild(el("p", item.prompt));
  card.appendChild(el("p", "Tick each requirement that A does NOT meet (missing):"));
  var list = document.createElement("ul");
  item.requirements.forEach(function (r, j) {
    var li = document.createElement("li");
    var lab = document.createElement("label");
    var cb = document.createElement("input");
    cb.type = "checkbox";
    cb.name = "missing-" + item.index;
    cb.value = String(j);
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(" missing: " + r));
    li.appendChild(lab);
    list.appendChild(li);
  });
  card.appendChild(list);
  var pair = el("div", undefined, "pair");
  pair.appendChild(figure(item.candidate, "A"));
  pair.appendChild(figure(item.reference, "B"));
  card.appendChild(pair);
  card.appendChild(radios(item, "pass", [{value: "pass", text: "Pass"}, {value: "fail", text: "Fail"}], "Does A pass?"));
  card.appendChild(radios(item, "vs", [{value: "better", text: "better"}, {value: "equal", text: "equal"}, {value: "worse", text: "worse"}], "A vs B"));
  var note = document.createElement("textarea");
  note.rows = 2;
  note.placeholder = "Optional note";
  note.name = "note-" + item.index;
  note.setAttribute("aria-label", "Optional note for " + item.label);
  card.appendChild(note);
  document.getElementById("items").appendChild(card);
});
function choice(i, name) {
  var e = document.querySelector('input[name="' + name + "-" + i.index + '"]:checked');
  return e ? e.value : null;
}
function noteOf(i) {
  var e = document.querySelector('textarea[name="note-' + i.index + '"]');
  return e ? e.value : "";
}
function missingOf(i) {
  var out = [];
  var boxes = document.querySelectorAll('input[name="missing-' + i.index + '"]:checked');
  for (var n = 0; n < boxes.length; n++) out.push(Number(boxes[n].value));
  return out;
}
function labelled() {
  return items.filter(function (i) {
    return choice(i, "pass") && choice(i, "vs");
  }).length;
}
function update() {
  var n = labelled();
  document.getElementById("progress").textContent = "labelled " + n + " / " + items.length;
  document.getElementById("download").disabled = n !== items.length;
}
document.getElementById("download").addEventListener("click", function () {
  if (labelled() !== items.length) return;
  var out = {
    version: "__VERSION__",
    packet_sha256: PACKET_SHA256,
    decisions: items.map(function (i) {
      return {index: i.index, pass: choice(i, "pass") === "pass", vs_reference: choice(i, "vs"), missing: missingOf(i), note: noteOf(i)};
    })
  };
  var blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = "decisions.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
});
update();
window.addEventListener("pageshow", update);
</script></body></html>
"""


def _inside(path: Path, directory: Path) -> bool:
    path, directory = Path(path).resolve(), Path(directory).resolve()
    return path == directory or directory in path.parents


def create_packet(
    rows: list[BufoGeneration], out_dir: Path, seed: int = 20261021, *, key_dir: Path
) -> dict:
    """Write the blinded packet to out_dir and its unblinding key to key_dir.

    key_dir must not be inside out_dir: the reviewer opens out_dir, so nothing that
    names an arm, model or judge verdict may live there.
    """
    out_dir, key_dir = Path(out_dir), Path(key_dir)
    if _inside(key_dir, out_dir):
        raise ValueError("key_dir must not be inside the packet directory")
    ordered = sorted(rows, key=lambda r: r.id)
    random.Random(seed).shuffle(ordered)
    items, key = [], []
    for index, row in enumerate(ordered):
        number = f"{index + 1:02d}"
        reference = row.benchmark_metadata["reference"]
        freeze(out_dir / "images" / f"item-{number}.png", Path(row.png_path).read_bytes())
        freeze(out_dir / "images" / f"ref-{number}.png", Path(reference["png_path"]).read_bytes())
        items.append(
            {
                "index": index,
                "label": f"Item {number}",
                "prompt": row.prompt,
                "requirements": list(row.benchmark_metadata["requirements"]),
                "candidate": f"images/item-{number}.png",
                "reference": f"images/ref-{number}.png",
            }
        )
        judge = _judge(row)
        key.append(
            {
                "index": index,
                "generation_id": row.id,
                "prompt_id": row.prompt_id,
                "backend": row.backend,
                "model": row.model,
                "judge_strict_pass": bool(judge.get("strict_pass")),
                "judge_vs_reference": judge.get("vs_reference"),
            }
        )
    items_bytes = _save_json(out_dir / "items.json", items)
    sha = hashlib.sha256(items_bytes).hexdigest()
    _save_json(key_dir / "unblinding-key.json", [k | {"packet_sha256": sha} for k in key])
    inline = (
        json.dumps(items).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    page = (
        _HTML.replace("__ITEMS__", inline)
        .replace("__SHA__", sha)
        .replace("__VERSION__", PACKET_VERSION)
    )
    freeze(out_dir / "packet.html", page.encode())
    return {
        "n": len(items),
        "packet_sha256": sha,
        "dir": str(out_dir),
        "key": str(key_dir / "unblinding-key.json"),
    }


def agreement_report(
    packet_dir: Path, decisions_path: Path, *, reviewer: str, key_path: Path
) -> dict:
    """Compare a reviewer's decisions with the judge; the report is written beside the key."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", reviewer or ""):
        raise ValueError("reviewer must match [A-Za-z0-9_-]+")
    packet_dir, key_path = Path(packet_dir), Path(key_path)
    if _inside(key_path, packet_dir):
        raise ValueError("key_path must not be inside the packet directory")
    sha = hashlib.sha256((packet_dir / "items.json").read_bytes()).hexdigest()
    payload = json.loads(Path(decisions_path).read_text())
    if payload.get("version") != PACKET_VERSION or payload.get("packet_sha256") != sha:
        raise ValueError("decisions do not match this packet")
    items = json.loads((packet_dir / "items.json").read_text())
    n_requirements = {item["index"]: len(item["requirements"]) for item in items}
    key = json.loads(key_path.read_text())
    if any(k.get("packet_sha256") != sha for k in key):
        raise ValueError("unblinding key does not match this packet")
    by_index = {k["index"]: k for k in key}
    decisions = {}
    for d in payload.get("decisions", []):
        index = d.get("index")
        if index not in by_index or index in decisions:
            raise ValueError(f"unknown or duplicate decision index {index}")
        missing = d.get("missing")
        if (
            not isinstance(d.get("pass"), bool)
            or d.get("vs_reference") not in _VS
            or not isinstance(missing, list)
            or any(type(j) is not int or not 0 <= j < n_requirements.get(index, 0) for j in missing)
            or len(set(missing)) != len(missing)
        ):
            raise ValueError(f"invalid decision at index {index}")
        decisions[index] = d
    if len(decisions) != len(by_index):
        raise ValueError(f"incomplete review: {len(decisions)} of {len(by_index)}")

    n = len(by_index)
    confusion = Counter()
    pass_agree = vs_agree = 0
    judge_pass_reviewer_fail, judge_fail_reviewer_pass, judge_pass_reviewer_missing = [], [], []
    arms: dict[str, list[int]] = {}
    for index in sorted(by_index):
        k, d = by_index[index], decisions[index]
        jp, rp = k["judge_strict_pass"], d["pass"]
        confusion[f"judge_{'pass' if jp else 'fail'}_reviewer_{'pass' if rp else 'fail'}"] += 1
        pass_agree += jp == rp
        vs_agree += k["judge_vs_reference"] == d["vs_reference"]
        if jp and not rp:
            judge_pass_reviewer_fail.append(index)
        if jp and d["missing"]:
            judge_pass_reviewer_missing.append(index)
        if rp and not jp:
            judge_fail_reviewer_pass.append(index)
        arm = arms.setdefault(k["backend"], [0, 0])
        arm[0] += 1
        arm[1] += jp == rp
    strict = pass_agree / n
    # Spec gate: agreement, and no drawing the judge passes that the reviewer marks as
    # missing a requirement. A judge-pass/reviewer-fail for craft alone is reported as a
    # secondary count but does not fail the gate.
    gate = {
        "agreement_ok": strict >= AGREEMENT_THRESHOLD,
        "no_judge_pass_reviewer_missing": not judge_pass_reviewer_missing,
    }
    gate["passed"] = gate["agreement_ok"] and gate["no_judge_pass_reviewer_missing"]
    report = {
        "reviewer": reviewer,
        "n": n,
        "strict_pass_agreement": strict,
        "vs_reference_agreement": vs_agree / n,
        "judge_pass_reviewer_missing": judge_pass_reviewer_missing,
        "judge_pass_reviewer_fail": judge_pass_reviewer_fail,
        "judge_fail_reviewer_pass": judge_fail_reviewer_pass,
        "confusion": {
            f"judge_{j}_reviewer_{r}": confusion[f"judge_{j}_reviewer_{r}"]
            for j in ("pass", "fail")
            for r in ("pass", "fail")
        },
        "gate": gate,
        "by_arm": {a: {"n": c[0], "agreement": c[1] / c[0]} for a, c in sorted(arms.items())},
    }
    (key_path.parent / f"agreement-{reviewer}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
