"""Free, local check: were the sweep students distinguishable from the base model they were scored with?

`served_model` is only the server's echo of the base name, so it cannot show that an adapter was
applied. This compares, per student arm, its rows against the `open_base` rows of the SAME run
directory (paired by prompt id, sign-flip permutation test on output tokens, SVG length and element
count), against base-vs-base controls (the null: two runs of the same model), and fingerprints the
SVG element ids of Sol's training trajectories. No model calls of any kind.
"""

import argparse
import collections
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

FEATURES = ("output_tokens", "svg_length", "element_tags")
P_DISTINCT, P_CONTROL, TOP_IDS = 0.01, 0.05, 30


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    by = collections.defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by[row["backend"]][row["prompt_id"]] = row
    return by


def _number(value):
    return value if isinstance(value, int | float) else 0


def features(row):
    svg = row.get("svg") or ""
    return {
        "output_tokens": _number(row.get("output_tokens")),
        "svg_length": len(svg),
        "element_tags": len(re.findall(r"<[a-zA-Z]+", svg)),
    }


def element_ids(row):
    return re.findall(r'id="([^"]+)"', row.get("svg") or "")


def paired_permutation(left, right, key, rng, replicates):
    """Two-sided paired sign-flip permutation test on per-prompt differences left - right."""
    ids = sorted(set(left) & set(right))
    diffs = [features(left[i])[key] - features(right[i])[key] for i in ids]
    observed = statistics.mean(diffs)
    hits = sum(
        abs(statistics.mean(d if rng.random() < 0.5 else -d for d in diffs)) >= abs(observed)
        for _ in range(replicates)
    )
    return {"n": len(ids), "mean_diff": round(observed, 1), "p": round(hits / replicates, 4)}


def top_training_ids(export_path, top=TOP_IDS):
    """Most frequent element ids across the assistant render_svg SVGs of the training export."""
    counts, svgs = collections.Counter(), 0
    for line in Path(export_path).read_text().splitlines():
        if not line.strip():
            continue
        for message in json.loads(line)["messages"]:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                svg = json.loads(call["function"]["arguments"]).get("svg") or ""
                if "<svg" in svg:
                    svgs += 1
                    counts.update(re.findall(r'id="([^"]+)"', svg))
    return [name for name, _ in counts.most_common(top)], svgs


def fingerprint_share(rows, top):
    shares = [
        sum(i in top for i in ids) / len(ids) for ids in map(element_ids, rows.values()) if ids
    ]
    return {
        "rows": len(rows),
        "rows_with_ids": len(shares),
        "mean_share": round(statistics.mean(shares), 3) if shares else 0.0,
    }


def analyze(
    *,
    sweep_dir,
    arms,
    control_run,
    sol_run,
    export,
    seed=1,
    replicates=20000,
    scored="astra-v5-k3",
    base_arms=None,
):
    """base_arms maps an arm to the in-run base arm it is compared with (default open_base).

    The null controls are Gemma open_base vs open_base only (they are the only repeated base
    runs); arms with another base are compared after them so earlier numbers are unchanged."""
    base_arms = {arm: (base_arms or {}).get(arm, "open_base") for arm in arms}
    sweep_dir, control_run, sol_run = Path(sweep_dir), Path(control_run), Path(sol_run)
    rng = random.Random(seed)
    inputs, student, base = {}, {}, {}
    for arm in arms:
        path = sweep_dir / arm / scored / "scored.jsonl"
        by = load_rows(path)
        inputs[str(path)] = sha256(path)
        student[arm], base[arm] = by[arm], by[base_arms[arm]]
    control_path = control_run / "scored.jsonl"
    control_by = load_rows(control_path)
    inputs[str(control_path)] = sha256(control_path)
    sol_path = sol_run / "scored.jsonl"
    sol = load_rows(sol_path)["closed_quality_bar"]
    inputs[str(sol_path)] = sha256(sol_path)
    inputs[str(export)] = sha256(export)

    def pair(left, right):
        return {key: paired_permutation(left, right, key, rng, replicates) for key in FEATURES}

    gemma = [a for a in arms if base_arms[a] == "open_base"]
    other = [a for a in arms if base_arms[a] != "open_base"]
    pairs = {f"{arm}_vs_open_base": pair(student[arm], base[arm]) for arm in gemma}
    bases = {f"{arm}:open_base": base[arm] for arm in gemma}
    bases[f"{control_run.parent.name}:open_base"] = control_by["open_base"]
    names = list(bases)
    controls = {
        f"{a}_vs_{b}": pair(bases[a], bases[b]) for i, a in enumerate(names) for b in names[i + 1 :]
    }
    for arm in other:
        pairs[f"{arm}_vs_{base_arms[arm]}"] = pair(student[arm], base[arm])
    informational = {}
    if "iter2_e3" in control_by:
        informational["iter2_e3_vs_open_base_prior_run"] = pair(
            control_by["iter2_e3"], control_by["open_base"]
        )
    top, svgs = top_training_ids(export)
    fingerprint = {
        "top_ids": top,
        "export_svgs": svgs,
        "sol_validation_rows": fingerprint_share(sol, top),
        **{f"{arm}_student": fingerprint_share(student[arm], top) for arm in arms},
        **{f"{name}": fingerprint_share(rows, top) for name, rows in bases.items()},
        **{f"{a}:{base_arms[a]}": fingerprint_share(base[a], top) for a in other},
    }
    controls_quiet = all(c["output_tokens"]["p"] > P_CONTROL for c in controls.values())
    verdict = {
        arm: (
            "distinct"
            if pairs[f"{arm}_vs_{base_arms[arm]}"]["output_tokens"]["p"] < P_DISTINCT
            and controls_quiet
            else "not_separable"
        )
        for arm in arms
    }
    return {
        "schema_version": 1,
        "rule": (
            f"distinct if output_tokens paired p < {P_DISTINCT} versus the same-run open_base while "
            f"every base-vs-base control has p > {P_CONTROL}; else not_separable"
        ),
        "base_arms": base_arms,
        "seed": seed,
        "replicates": replicates,
        "inputs": inputs,
        "student_vs_base": pairs,
        "base_vs_base_controls": controls,
        "informational": informational,
        "sol_training_id_fingerprint": fingerprint,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("runs/distill/stage-b/sweep"))
    parser.add_argument("--arm", action="append", dest="arms")
    parser.add_argument(
        "--control-run", type=Path, default=Path("runs/distill/arms-v1-students/astra-v5-k3")
    )
    parser.add_argument("--sol-run", type=Path, default=Path("runs/distill/arms-v1/astra-v5-k3"))
    parser.add_argument(
        "--export", type=Path, default=Path("runs/distill/stage-b/export/accepted.jsonl")
    )
    parser.add_argument(
        "--base-arm",
        action="append",
        default=[],
        help="ARM=BASE_ARM, the in-run base an arm is compared with (default open_base)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/distill/stage-b/sweep/distinctness.json")
    )
    args = parser.parse_args()
    report = analyze(
        sweep_dir=args.sweep_dir,
        arms=args.arms or ["sft_j1", "sft_j3"],
        control_run=args.control_run,
        sol_run=args.sol_run,
        export=args.export,
        seed=args.seed,
        replicates=args.replicates,
        base_arms=dict(item.split("=", 1) for item in args.base_arm),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"]))


if __name__ == "__main__":
    main()
