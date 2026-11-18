import json
import math
import re

import pytest
from PIL import Image

from bufogen.curation.review_packet import (
    agreement_report,
    create_packet,
    load_scored_rows,
    select_review_rows,
)
from bufogen.generation import BufoGeneration

BACKENDS = ("alpha-backend", "beta-backend")
MODELS = {"alpha-backend": "zeta-model-1", "beta-backend": "omega-model-2"}


def make_rows(tmp_path, n=60, relative=False):
    rows = []
    for i in range(n):
        arm = BACKENDS[i % 2]
        png = tmp_path / f"gen{i}.png"
        ref = tmp_path / f"ref{i}.png"
        Image.new("RGB", (4, 4), (i, 0, 0)).save(png)
        Image.new("RGB", (4, 4), (0, i, 0)).save(ref)
        passed = i % 3 != 0
        rows.append(
            BufoGeneration(
                prompt_id=f"p{i}",
                prompt=f"draw thing {i}",
                backend=arm,
                model=MODELS[arm],
                group_id="g",
                svg="<svg/>",
                valid=True,
                latency_s=1.0,
                id=f"id{i:03d}",
                png_path=png.name if relative else str(png),
                scores={
                    "astra_judge": {
                        "status": "ok",
                        "strict_pass": passed,
                        "vs_reference": ("better", "equal", "worse")[i % 3],
                    }
                },
                benchmark_metadata={
                    "status": "completed",
                    "split": "validation",
                    "requirements": [f"req {i}a", f"req {i}b"],
                    "reference": {
                        "png_path": str(ref),
                        "png_sha256": "x",
                        "generation_id": f"r{i}",
                        "is_self": False,
                    },
                },
            )
        )
    return rows


def test_select_rows_stratified_and_deterministic(tmp_path):
    rows = make_rows(tmp_path)
    rows[0].scores["astra_judge"]["status"] = "judge_error"
    rows[1].png_path = None
    a = select_review_rows(rows, size=40)
    b = select_review_rows(list(reversed(rows)), size=40)
    assert [r.id for r in a] == [r.id for r in b]
    assert len(a) == len({r.id for r in a}) == 40
    assert "id000" not in {r.id for r in a} and "id001" not in {r.id for r in a}
    passes = sum(r.scores["astra_judge"]["strict_pass"] for r in a)
    assert 16 <= passes <= 24
    for arm in BACKENDS:
        assert sum(r.backend == arm for r in a) >= 10
    with pytest.raises(ValueError):
        select_review_rows(rows[:30], size=40)


def test_select_excludes_disclosed_generations(tmp_path):
    rows = make_rows(tmp_path)
    baseline = select_review_rows(rows, size=40)
    disclosed = {baseline[0].id, baseline[5].id}
    sample = select_review_rows(rows, size=40, exclude_ids=disclosed)
    assert len(sample) == 40 and not disclosed & {r.id for r in sample}
    # Excluding rows the sample never held still yields a complete, valid sample.
    untouched = select_review_rows(rows, size=40, exclude_ids={"not-a-row"})
    assert [r.id for r in untouched] == [r.id for r in baseline]


def test_select_requires_completed_generation(tmp_path):
    rows = make_rows(tmp_path, n=44)
    for row in rows[:4]:
        row.benchmark_metadata["status"] = "generation_error"
    chosen = select_review_rows(rows, size=40)
    assert {r.id for r in rows[:4]}.isdisjoint({r.id for r in chosen})
    with pytest.raises(ValueError, match="need 41 eligible rows, found 40"):
        select_review_rows(rows, size=41)


@pytest.mark.parametrize("n_arms", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("size", [4, 5, 7, 10, 13, 40])
def test_select_never_cuts_class_minimum_rows(tmp_path, n_arms, size):
    # Arms are skewed: every arm but the last is all-pass and the last is all-fail,
    # so the class phase must add fails after the arm phase.
    rows = make_rows(tmp_path, n=240)
    for i, row in enumerate(rows):
        arm = i % n_arms
        row.backend = f"arm{arm}"
        row.scores["astra_judge"]["strict_pass"] = arm != n_arms - 1 if n_arms > 1 else i % 2 == 0
    chosen = select_review_rows(rows, size=size)
    assert len(chosen) == len({r.id for r in chosen}) == size
    class_min = math.ceil(0.4 * size)
    passes = sum(r.scores["astra_judge"]["strict_pass"] for r in chosen)
    assert passes >= class_min and size - passes >= class_min


def packet(tmp_path, **kw):
    rows = select_review_rows(make_rows(tmp_path), size=40)
    out = tmp_path / "packet"
    kw.setdefault("key_dir", tmp_path / "key")
    create_packet(rows, out, **kw)
    return rows, out


def key_of(out):
    return json.loads((out.parent / "key" / "unblinding-key.json").read_text())


def agree(out, decisions, reviewer="me"):
    return agreement_report(
        out, decisions, reviewer=reviewer, key_path=out.parent / "key" / "unblinding-key.json"
    )


def test_key_lives_outside_packet(tmp_path):
    rows = select_review_rows(make_rows(tmp_path), size=40)
    out = tmp_path / "packet"
    for bad in (out, out / "key", out / "a" / "b"):
        with pytest.raises(ValueError, match="key_dir"):
            create_packet(rows, out, key_dir=bad)
    create_packet(rows, out, key_dir=tmp_path / "key")
    assert not (out / "unblinding-key.json").exists()
    assert (tmp_path / "key" / "unblinding-key.json").exists()
    assert not any("key" in p.name for p in out.rglob("*"))


def test_agreement_writes_report_beside_key_not_in_packet(tmp_path):
    _, out = packet(tmp_path)
    agree(out, write(tmp_path, decisions_for(out)))
    assert (tmp_path / "key" / "agreement-me.json").exists()
    assert not list(out.glob("agreement-*"))
    with pytest.raises(ValueError, match="key_path"):
        agreement_report(
            out,
            write(tmp_path, decisions_for(out)),
            reviewer="me",
            key_path=out / "unblinding-key.json",
        )


def test_packet_files_and_neutral_names(tmp_path):
    _, out = packet(tmp_path)
    items = json.loads((out / "items.json").read_text())
    assert len(items) == 40
    assert (
        items[0]["candidate"] == "images/item-01.png"
        and items[0]["reference"] == "images/ref-01.png"
    )
    assert (out / "images" / "item-40.png").exists() and (out / "images" / "ref-40.png").exists()
    key = key_of(out)
    assert {"index", "generation_id", "prompt_id", "backend", "model"} <= set(key[0])
    text = (out / "items.json").read_text() + (out / "packet.html").read_text()
    for name in (*BACKENDS, *MODELS.values(), "backend", "model"):
        assert name not in text
    # shuffled, not arm-ordered
    arms = [k["backend"] for k in key]
    assert arms != sorted(arms) and arms != [BACKENDS[i % 2] for i in range(40)]


def test_packet_html_is_self_contained(tmp_path):
    _, out = packet(tmp_path)
    html = (out / "packet.html").read_text()
    assert "http://" not in html and "https://" not in html
    assert "fetch(" not in html and "innerHTML" not in html
    assert 'type="application/json"' in html
    for src in re.findall(r'src="([^"]*)"', html):
        assert src.startswith("images/")
    assert "disabled" in html and "position:fixed" in html


def test_packet_freeze_rejects_differing_rewrite(tmp_path):
    rows, out = packet(tmp_path)
    create_packet(rows, out, key_dir=tmp_path / "key")  # identical rewrite is fine
    with pytest.raises(ValueError):
        create_packet(rows, out, seed=1, key_dir=tmp_path / "key")


def decisions_for(out, flips=(), missing=None):
    missing = missing or {}
    key = key_of(out)
    sha = __import__("hashlib").sha256((out / "items.json").read_bytes()).hexdigest()
    decisions = []
    for k in key:
        p = k["judge_strict_pass"]
        if k["index"] in flips:
            p = not p
        decisions.append(
            {
                "index": k["index"],
                "pass": p,
                "vs_reference": k["judge_vs_reference"],
                "missing": list(missing.get(k["index"], [])),
                "note": "",
            }
        )
    return {"version": "bufo-review-v2", "packet_sha256": sha, "decisions": decisions}


def write(tmp_path, payload):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(payload))
    return path


def judge_state(out):
    key = key_of(out)
    return [k["index"] for k in key if k["judge_strict_pass"]], [
        k["index"] for k in key if not k["judge_strict_pass"]
    ]


def test_agreement_requires_every_item(tmp_path):
    _, out = packet(tmp_path)
    payload = decisions_for(out)
    payload["decisions"] = payload["decisions"][:39]
    with pytest.raises(ValueError, match="incomplete review: 39 of 40"):
        agree(out, write(tmp_path, payload), reviewer="me")


def test_agreement_gate_logic(tmp_path):
    _, out = packet(tmp_path)
    passed, failed = judge_state(out)
    # exactly 0.8 agreement via 8 judge-fail/reviewer-pass flips: gate passes
    report = agree(out, write(tmp_path, decisions_for(out, flips=failed[:8])), reviewer="me")
    assert report["strict_pass_agreement"] == 0.8
    assert report["gate"] == {
        "agreement_ok": True,
        "no_judge_pass_reviewer_missing": True,
        "passed": True,
    }
    assert report["judge_fail_reviewer_pass"] == sorted(failed[:8])
    assert (out.parent / "key" / "agreement-me.json").exists()
    assert sum(report["confusion"].values()) == 40
    assert sum(v["n"] for v in report["by_arm"].values()) == 40
    # just under 0.8
    low = agree(out, write(tmp_path, decisions_for(out, flips=failed[:9])), reviewer="low")
    assert low["gate"]["agreement_ok"] is False and low["gate"]["passed"] is False
    # A judge-pass/reviewer-fail with no requirement ticked missing (a craft fail) is
    # reported but does not break the spec gate.
    one = agree(out, write(tmp_path, decisions_for(out, flips=passed[:1])), reviewer="one")
    assert one["gate"] == {
        "agreement_ok": True,
        "no_judge_pass_reviewer_missing": True,
        "passed": True,
    }
    assert one["judge_pass_reviewer_fail"] == [passed[0]]
    assert one["judge_pass_reviewer_missing"] == []
    # A judge pass where the reviewer ticks a requirement missing breaks the gate,
    # even at full pass/fail agreement.
    miss = agree(
        out, write(tmp_path, decisions_for(out, missing={passed[0]: [1]})), reviewer="miss"
    )
    assert miss["strict_pass_agreement"] == 1.0
    assert miss["gate"] == {
        "agreement_ok": True,
        "no_judge_pass_reviewer_missing": False,
        "passed": False,
    }
    assert miss["judge_pass_reviewer_missing"] == [passed[0]]
    assert miss["judge_pass_reviewer_fail"] == []
    # Missing requirements on a judge fail are expected and never break the gate.
    fine = agree(
        out, write(tmp_path, decisions_for(out, missing={failed[0]: [0, 1]})), reviewer="fine"
    )
    assert fine["gate"]["passed"] is True and fine["judge_pass_reviewer_missing"] == []


def test_agreement_rejects_v1_decisions_and_bad_missing(tmp_path):
    _, out = packet(tmp_path)
    v1 = decisions_for(out)
    v1["version"] = "bufo-review-v1"
    for d in v1["decisions"]:
        del d["missing"]
    with pytest.raises(ValueError, match="decisions do not match"):
        agree(out, write(tmp_path, v1))
    first = key_of(out)[0]["index"]
    for bad in ([2], [-1], [0, 0], ["0"], [True], None, "0"):
        payload = decisions_for(out)
        for d in payload["decisions"]:
            if d["index"] == first:
                d["missing"] = bad
        with pytest.raises(ValueError, match=f"invalid decision at index {first}"):
            agree(out, write(tmp_path, payload))


def test_agreement_rejects_wrong_packet_hash(tmp_path):
    _, out = packet(tmp_path)
    payload = decisions_for(out)
    payload["packet_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        agree(out, write(tmp_path, payload), reviewer="me")


def test_load_scored_rows_resolves_relative_paths_against_run_dir(tmp_path):
    run = tmp_path / "run"
    (run / "astra").mkdir(parents=True)
    rows = make_rows(run, n=3, relative=True)
    (run / "astra" / "scored.jsonl").write_text(
        "".join(json.dumps(r.to_dict()) + "\n" for r in rows)
    )
    loaded = load_scored_rows(run)
    assert [r.png_path for r in loaded] == [str((run / f"gen{i}.png").resolve()) for i in range(3)]


def test_html_records_missing_requirements(tmp_path):
    _, out = packet(tmp_path)
    html = (out / "packet.html").read_text()
    assert 'version: "bufo-review-v2"' in html
    # Each requirement checkbox is named per item, valued by requirement index, and
    # labelled so that ticking it means the requirement is NOT met.
    assert 'cb.name = "missing-" + item.index;' in html
    assert "cb.value = String(j);" in html
    assert "missing" in html.split("item.requirements.forEach")[0].lower()
    assert "does NOT meet" in html
    assert "input[name=\"missing-' + i.index + '\"]:checked" in html
    assert "missing: missingOf(i)" in html


def test_html_derives_state_from_dom(tmp_path):
    _, out = packet(tmp_path)
    html = (out / "packet.html").read_text()
    assert 'autocomplete="off"' in html
    assert 'input[name="' in html and '"]:checked' in html
    assert "state[" not in html
    assert "setTimeout(function () { URL.revokeObjectURL" in html
    assert "aria-label" in html


def test_agreement_input_validation(tmp_path):
    _, out = packet(tmp_path)
    good = decisions_for(out)
    dup = json.loads(json.dumps(good))
    dup["decisions"][1]["index"] = dup["decisions"][0]["index"]
    with pytest.raises(ValueError, match="duplicate"):
        agree(out, write(tmp_path, dup), reviewer="me")
    unknown = json.loads(json.dumps(good))
    unknown["decisions"][0]["index"] = 999
    with pytest.raises(ValueError, match="unknown"):
        agree(out, write(tmp_path, unknown), reviewer="me")
    bad = json.loads(json.dumps(good))
    bad["decisions"][3]["vs_reference"] = "meh"
    with pytest.raises(ValueError, match="invalid decision at index 3"):
        agree(out, write(tmp_path, bad), reviewer="me")
    with pytest.raises(ValueError, match="reviewer"):
        agree(out, write(tmp_path, good), reviewer="../x")


def test_select_small_size_and_skewed_arms(tmp_path):
    rows = make_rows(tmp_path)
    with pytest.raises(ValueError, match="size must be"):
        select_review_rows(rows, size=3)
    # arm beta is all-pass
    for r in rows:
        if r.backend == BACKENDS[1]:
            r.scores["astra_judge"]["strict_pass"] = True
    a = select_review_rows(rows, size=40)
    assert len(a) == 40
    assert all(sum(r.backend == arm for r in a) >= 10 for arm in BACKENDS)
    # too few fails overall
    for r in rows[:50]:
        r.scores["astra_judge"]["strict_pass"] = True
    b = select_review_rows(rows, size=40)
    assert len(b) == len({r.id for r in b}) == 40
    assert all(sum(r.backend == arm for r in b) >= 10 for arm in BACKENDS)
    for size in (4, 5, 7, 9):
        assert len(select_review_rows(rows, size=size)) == size


def test_agreement_rejects_key_from_another_packet(tmp_path):
    rows, out = packet(tmp_path)
    other = tmp_path / "other"
    create_packet(rows, other / "packet", seed=7, key_dir=other / "key")
    with pytest.raises(ValueError, match="key does not match"):
        agreement_report(
            out,
            write(tmp_path, decisions_for(out)),
            reviewer="me",
            key_path=other / "key" / "unblinding-key.json",
        )


_DOM_HARNESS = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const data = html.match(/<script type="application\/json" id="packet-data">([\s\S]*?)<\/script>/)[1];
const code = html.match(/<script>\n([\s\S]*?)<\/script>/)[1];
const all = [];
function make(tag) {
  const e = {tag, children: [], listeners: {}, className: "", textContent: "", value: "",
    checked: false, disabled: false, name: "", type: "", setAttribute() {},
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(k, f) { this.listeners[k] = f; }, click() {}, remove() {}};
  all.push(e);
  return e;
}
const byId = {"packet-data": {textContent: data}, items: make("div"),
  progress: make("span"), download: make("button")};
function match(sel) {
  const m = sel.match(/^(\w+)\[name="([^"]+)"\](:checked)?$/);
  return all.filter((e) => e.tag === m[1] && e.name === m[2] && (!m[3] || e.checked));
}
let saved = null;
global.document = {getElementById: (id) => byId[id], createElement: make,
  createTextNode: (t) => ({text: t}), body: make("body"),
  querySelector: (s) => match(s)[0] || null, querySelectorAll: (s) => match(s)};
global.window = {addEventListener() {}};
global.Blob = function (parts) { saved = parts.join(""); };
global.URL = {createObjectURL: () => "blob:x", revokeObjectURL() {}};
global.setTimeout = () => {};
eval(code);
const items = JSON.parse(data);
items.forEach((item) => {
  for (const name of ["pass", "vs"]) {
    const radio = all.find((e) => e.name === name + "-" + item.index);
    radio.checked = true;
  }
});
const first = all.filter((e) => e.name === "missing-" + items[0].index);
first[1].checked = true;
byId.download.listeners.click();
process.stdout.write(saved);
"""


def test_packet_page_serialises_missing_requirements(tmp_path):
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node not installed")
    _, out = packet(tmp_path)
    harness = tmp_path / "harness.cjs"
    harness.write_text(_DOM_HARNESS)
    result = subprocess.run(
        ["node", str(harness), str(out / "packet.html")],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["version"] == "bufo-review-v2"
    first, *rest = payload["decisions"]
    assert first["missing"] == [1]
    assert all(d["missing"] == [] for d in rest)
    assert set(first) == {"index", "pass", "vs_reference", "missing", "note"}
    # The page's output is accepted by agreement_report as-is.
    agree(out, write(tmp_path, payload))


@pytest.mark.parametrize("split", ["heldout", "test"])
def test_select_refuses_heldout_without_confirmation(tmp_path, split):
    rows = make_rows(tmp_path)
    rows[5].benchmark_metadata["split"] = split
    with pytest.raises(ValueError, match="confirmation"):
        select_review_rows(rows, size=40)
    assert len(select_review_rows(rows, size=40, confirmation=True)) == 40


def test_load_scored_rows_reads_named_judge_dir_and_rejects_odd_names(tmp_path):
    run = tmp_path / "run"
    (run / "astra-v4-k3").mkdir(parents=True)
    rows = make_rows(run, n=2, relative=True)
    (run / "astra-v4-k3" / "scored.jsonl").write_text(
        "".join(json.dumps(r.to_dict()) + "\n" for r in rows)
    )
    loaded = load_scored_rows(run, "astra-v4-k3")
    assert [r.png_path for r in loaded] == [str((run / f"gen{i}.png").resolve()) for i in range(2)]
    with pytest.raises(FileNotFoundError):
        load_scored_rows(run)
    for bad in ("../astra", "astra/..", "scored", "astra-v4-k3/x"):
        with pytest.raises(ValueError):
            load_scored_rows(run, bad)


# --- multi-run confirmation packet (Task 10 Step 3), synthetic fixtures only ---

ARMS = ("arm-base", "arm-sol", "arm-sft", "arm-rl")


def make_confirmation_runs(tmp_path, arms=ARMS, per_arm=30, split="heldout", shared_ids=False):
    """One run directory per arm, each with astra-v5-k3/scored.jsonl and relative paths."""
    runs = []
    for a, arm in enumerate(arms):
        run = tmp_path / f"runs/{arm}"
        (run / "astra-v5-k3").mkdir(parents=True, exist_ok=True)
        rows = make_rows(run, n=per_arm, relative=True)
        for i, row in enumerate(rows):
            row.backend = arm
            row.model = f"model-of-{arm}"
            row.prompt_id = f"hp{i:03d}"
            row.benchmark_metadata["split"] = split
            if not shared_ids:
                row.id = f"{arm}-{i:03d}"
        (run / "astra-v5-k3" / "scored.jsonl").write_text(
            "".join(json.dumps(r.to_dict()) + "\n" for r in rows)
        )
        runs.append(run)
    return runs


def load_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/make_review_packet.py"
    spec = importlib.util.spec_from_file_location("make_review_packet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(module, tmp_path, runs, *extra):
    argv = ["--out", str(tmp_path / "packet"), "--key-dir", str(tmp_path / "key")]
    argv += ["--seed", "20261027", "--scored-dir", "astra-v5-k3"]
    for run in runs:
        argv += ["--runs", str(run)]
    module.main(argv + list(extra))


def test_load_scored_runs_merges_dirs_and_dedupes_ids(tmp_path):
    from bufogen.curation.review_packet import load_scored_runs

    runs = make_confirmation_runs(tmp_path, arms=ARMS[:2], per_arm=5, shared_ids=True)
    # Shared ids across two runs collapse to the first run's row.
    rows = load_scored_runs(runs, "astra-v5-k3")
    assert len(rows) == 5 and {r.backend for r in rows} == {"arm-base"}
    runs = make_confirmation_runs(tmp_path, arms=ARMS[:2], per_arm=5)
    rows = load_scored_runs(runs, "astra-v5-k3")
    assert len(rows) == 10 and {r.backend for r in rows} == set(ARMS[:2])
    assert all(r.png_path.startswith(str(tmp_path)) for r in rows)


def test_multi_run_selection_balances_arms(tmp_path):
    from collections import Counter

    from bufogen.curation.review_packet import load_scored_runs

    runs = make_confirmation_runs(tmp_path)
    rows = load_scored_runs(runs, "astra-v5-k3")
    chosen = select_review_rows(rows, size=40, seed=20261027, confirmation=True)
    counts = Counter(r.backend for r in chosen)
    assert len(chosen) == 40 and len({r.id for r in chosen}) == 40
    assert set(counts) == set(ARMS) and min(counts.values()) >= 40 // (2 * len(ARMS))


def test_cli_heldout_without_confirmation_refused_before_reading(tmp_path):
    module = load_script()
    missing = tmp_path / "does-not-exist"  # reading it would raise FileNotFoundError
    with pytest.raises(SystemExit):
        run_cli(module, tmp_path, [missing], "--split", "heldout")
    assert not (tmp_path / "packet").exists() and not (tmp_path / "key").exists()


def test_cli_confirmation_requires_heldout_split_and_runs(tmp_path):
    module = load_script()
    runs = make_confirmation_runs(tmp_path)
    with pytest.raises(SystemExit):
        run_cli(module, tmp_path, runs, "--split", "validation", "--confirmation")
    with pytest.raises(SystemExit):
        run_cli(module, tmp_path, [], "--split", "heldout", "--confirmation")


def test_cli_heldout_confirmation_builds_blinded_packet(tmp_path):
    module = load_script()
    runs = make_confirmation_runs(tmp_path)
    run_cli(module, tmp_path, runs, "--split", "heldout", "--confirmation")
    packet_dir, key_dir = tmp_path / "packet", tmp_path / "key"
    items = json.loads((packet_dir / "items.json").read_text())
    key = json.loads((key_dir / "unblinding-key.json").read_text())
    assert len(items) == len(key) == 40
    assert {k["backend"] for k in key} == set(ARMS)
    assert all({"prompt_id", "backend", "judge_strict_pass"} <= set(k) for k in key)
    blob = "".join(p.read_text() for p in packet_dir.rglob("*") if p.suffix in {".json", ".html"})
    for arm in ARMS:
        assert arm not in blob and f"model-of-{arm}" not in blob
    assert not re.search(r"hp\d{3}", blob)
    raw = (packet_dir / "items.json").read_text()
    for field in ("strict_pass", "vs_reference", "prompt_id", "backend", "generation_id"):
        assert field not in raw
    assert set(items[0]) == {"index", "label", "prompt", "requirements", "candidate", "reference"}
    assert not list(packet_dir.rglob("unblinding-key.json"))


def test_cli_split_filters_rows(tmp_path):
    module = load_script()
    runs = make_confirmation_runs(tmp_path, split="validation")
    with pytest.raises(ValueError, match="eligible"):
        run_cli(module, tmp_path, runs, "--split", "heldout", "--confirmation")
