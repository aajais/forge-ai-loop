"""Offline paired statistics; families are the resampling units."""

import numpy as np


def strict_pass(row):
    return int(
        row.benchmark_metadata["status"] == "completed"
        and row.valid
        and row.scores["scene_quality"].get("quality_failure") is False
    )


def paired_family_interval(
    rows, left, right, *, families=16, per_family=5, replicates=20000, seed=20261015
):
    if left == right or replicates < 1:
        raise ValueError("Require distinct arms and positive bootstrap replicates")
    lookup = {(r.prompt_id, r.backend): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate paired rows")
    a_rows = [r for r in rows if r.backend == left]
    b_rows = [r for r in rows if r.backend == right]
    if not a_rows or {r.prompt_id for r in a_rows} != {r.prompt_id for r in b_rows}:
        raise ValueError("Incomplete paired prompt coverage")
    clusters = {}
    for a in a_rows:
        b = lookup[a.prompt_id, right]
        family = a.benchmark_metadata["family"]
        if family != b.benchmark_metadata["family"]:
            raise ValueError("Paired family mismatch")
        clusters.setdefault(family, []).append(strict_pass(a) - strict_pass(b))
    if len(clusters) != families or any(len(v) != per_family for v in clusters.values()):
        raise ValueError("Unexpected family/variant coverage")
    differences = np.array([np.mean(clusters[f]) for f in sorted(clusters)])
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, families, size=(replicates, families))].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "difference": float(differences.mean()),
        "ci95": [float(lo), float(hi)],
        "families": families,
        "prompts": len(a_rows),
        "replicates": replicates,
        "seed": seed,
        "method": "paired family-cluster percentile bootstrap",
        "family_differences": {f: float(np.mean(v)) for f, v in sorted(clusters.items())},
        "degenerate_interval": bool(lo == hi),
    }
