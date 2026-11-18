"""Anti-anchoring probe: a near-copy of the reference that breaks one requirement must fail it.

The reference drawing's SVG gets its exact sign text changed to a different word, is rendered
through the sandbox renderer, and is judged against the unmodified reference PNG. The calibrated
judge must mark the altered requirement unsatisfied and call the near-copy worse than the
reference. Never touches heldout rows.
"""

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

from bufogen.settings import ROOT

RUBRIC = "bufo-astra-v5"
SAMPLES = 3
MIN_UNSATISFIED = 11
MIN_WORSE = 10
REQUIRED = re.compile(r"reads exactly (.+)$")
REPLACEMENTS = {
    "READY": "LATER",
    "GATE 4": "DOCK 7",
    "FRESH": "STALE",
    "ICE": "COLD",
    "OPEN": "SHUT",
}
SOURCE = ROOT / "runs/distill/sol-run1/manifest.jsonl"
REFERENCES = ROOT / "runs/references/distill-v1/references.json"
OUT = ROOT / "runs/distill/stage-b/anchoring"


def mutate_text(svg: str, old: str, new: str) -> str:
    """Replace the exact content of the single ``<text>`` element holding ``old``."""
    pattern = re.compile(r"(<text\b[^>]*>)" + re.escape(old) + r"(</text>)")
    count = len(pattern.findall(svg))
    if count == 0:
        raise ValueError(f"text {old!r} is absent from the SVG")
    if count > 1:
        raise ValueError(f"text {old!r} is ambiguous: {count} <text> elements")
    return pattern.sub(lambda m: m.group(1) + new + m.group(2), svg)


def required_text(requirements: list[str]) -> tuple[str, str] | None:
    for requirement in requirements:
        match = REQUIRED.search(requirement)
        if match:
            return requirement, match.group(1).strip()
    return None


def forbidden_words(prompts: list[dict]) -> set[str]:
    """Every exact text any prompt (all splits) requires; only requirement text is read."""
    words = set()
    for prompt in prompts:
        for requirement in prompt["requirements"]:
            match = REQUIRED.search(requirement)
            if match:
                words.add(match.group(1).strip())
    return words


def check_replacement(old: str, new: str, forbidden: set[str]) -> None:
    if new == old or new in forbidden:
        raise ValueError(f"replacement {new!r} for {old!r} collides with a required text")


def select_rows(rows: list[dict], train_n: int = 4) -> list[dict]:
    """All validation rows plus the first ``train_n`` train rows by prompt_id. Never heldout."""
    usable = [
        r
        for r in rows
        if r["status"] == "completed"
        and r["split"] in ("train", "validation")
        and required_text(r["requirements"])
    ]
    usable.sort(key=lambda r: r["prompt_id"])
    return [r for r in usable if r["split"] == "validation"] + [
        r for r in usable if r["split"] == "train"
    ][:train_n]


def manifest_entry(prompt_id, split, original, replacement, requirement, ref_sha, copy_sha) -> dict:
    return {
        "prompt_id": prompt_id,
        "split": split,
        "original_word": original,
        "replacement": replacement,
        "altered_requirement": requirement,
        "reference_sha256": ref_sha,
        "near_copy_sha256": copy_sha,
    }


def build_report(entries: list[dict], verdicts: dict[str, dict]) -> dict:
    errors = unsatisfied = worse = strict = 0
    for entry in entries:
        verdict = verdicts.get(entry["prompt_id"])
        if not verdict or verdict.get("status") != "ok":
            errors += 1
            continue
        strict += bool(verdict.get("strict_pass"))
        worse += verdict.get("vs_reference") == "worse"
        unsatisfied += any(
            c["requirement"] == entry["altered_requirement"] and not c["satisfied"]
            for c in verdict["prompt_checks"]
        )
    n = len(entries)
    gate_u, gate_w = unsatisfied >= MIN_UNSATISFIED, worse >= MIN_WORSE
    return {
        "rubric_version": RUBRIC,
        "samples": SAMPLES,
        "rows": n,
        "altered_unsatisfied": unsatisfied,
        "vs_reference_worse": worse,
        "strict_pass": strict,
        "judge_errors": errors,
        "thresholds": {"altered_unsatisfied": MIN_UNSATISFIED, "vs_reference_worse": MIN_WORSE},
        "gate_unsatisfied": gate_u,
        "gate_worse": gate_w,
        "gate_passed": gate_u and gate_w,
    }


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def render_all(picked: list[dict], references: dict, out: Path, forbidden: set[str]):
    from bufogen.rendering.renderer import Renderer

    renderer = Renderer()
    entries, rows, skipped = [], [], []
    try:
        for r in picked:
            pid = r["prompt_id"]
            requirement, word = required_text(r["requirements"])
            new = REPLACEMENTS.get(word)
            if new is None:
                skipped.append({"prompt_id": pid, "reason": f"no replacement for {word!r}"})
                continue
            try:
                check_replacement(word, new, forbidden)
                svg = mutate_text(Path(r["svg_path"]).read_text(), word, new)
            except ValueError as exc:
                skipped.append({"prompt_id": pid, "reason": str(exc)})
                continue
            result = await renderer.render(svg)
            if not result.png_bytes:
                skipped.append({"prompt_id": pid, "reason": f"render failed: {result.error}"})
                continue
            (out / "near-copies" / f"{pid}.svg").write_text(svg)
            png = out / "near-copies" / f"{pid}.png"
            png.write_bytes(result.png_bytes)
            ref = references[pid]
            ref_png = REFERENCES.parent / ref["png_path"]
            entries.append(
                manifest_entry(
                    pid,
                    r["split"],
                    word,
                    new,
                    requirement,
                    ref["png_sha256"],
                    sha(result.png_bytes),
                )
            )
            rows.append(
                {
                    "id": f"anchoring-{pid}",
                    "valid": True,
                    "png_path": str(png),
                    "prompt": r["prompt"],
                    "prompt_id": pid,
                    "backend": "anchoring_probe",
                    "model": "near-copy",
                    "benchmark_metadata": {
                        "status": "completed",
                        "split": r["split"],
                        "family": r["family"],
                        "requirements": r["requirements"],
                        "reference": {
                            "png_path": str(ref_png),
                            "png_sha256": ref["png_sha256"],
                            "generation_id": ref["generation_id"],
                            "is_self": False,
                        },
                    },
                }
            )
    finally:
        await renderer.close()
    return entries, rows, skipped


async def judge_all(rows: list[dict]) -> dict[str, dict]:
    from bufogen.evaluation.scorers.astra_judge import AstraReferenceJudge

    judge = AstraReferenceJudge(rubric_version=RUBRIC, samples=SAMPLES)
    gate = asyncio.Semaphore(4)

    async def one(row):
        async with gate:
            return row["prompt_id"], await judge.score(row, rescore=True)

    return dict(await asyncio.gather(*(one(r) for r in rows)))


async def main_async(out: Path, only: set[str] | None = None):
    (out / "near-copies").mkdir(parents=True, exist_ok=True)
    gens = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    flat = [
        {
            "prompt_id": g["prompt_id"],
            "prompt": g["prompt"],
            "svg_path": g["svg_path"],
            "split": g["benchmark_metadata"]["split"],
            "status": g["benchmark_metadata"]["status"],
            "family": g["benchmark_metadata"]["family"],
            "requirements": g["benchmark_metadata"]["requirements"],
        }
        for g in gens
    ]
    references = json.loads(REFERENCES.read_text())["rows"]
    from bufogen.evaluation.benchmark import load_benchmark

    forbidden = forbidden_words(load_benchmark(ROOT / "config/distill_prompts.yaml"))
    picked = select_rows(flat)
    if only:
        picked = [r for r in picked if r["prompt_id"] in only]
    entries, rows, skipped = await render_all(picked, references, out, forbidden)
    verdicts = await judge_all(rows)
    if only:
        # Re-run of selected rows: keep the other rows' saved entries and verdicts.
        old = json.loads((out / "manifest.json").read_text())
        entries = [e for e in old["rows"] if e["prompt_id"] not in only] + entries
        skipped = old["skipped"] + skipped
        saved = {
            d["prompt_id"]: d
            for d in map(json.loads, (out / "verdicts.jsonl").read_text().splitlines())
            if d["prompt_id"] not in only
        }
        verdicts = {
            k: {k2: v for k2, v in d.items() if k2 != "prompt_id"} for k, d in saved.items()
        } | verdicts
    (out / "manifest.json").write_text(json.dumps({"rows": entries, "skipped": skipped}, indent=2))
    (out / "verdicts.jsonl").write_text(
        "".join(json.dumps({"prompt_id": k} | v) + "\n" for k, v in verdicts.items())
    )
    report = build_report(entries, verdicts) | {"skipped": skipped}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="re-run only these prompt ids")
    args = parser.parse_args()
    asyncio.run(main_async(OUT, set(args.only) if args.only else None))
