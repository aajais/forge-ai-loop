"""Offline paired statistics; families are the resampling units.

Algorithm copied from docs/forge-gap-work/paired_statistics.py (frozen there by its own
tests); here the family count and prompts per family are inferred from the rows. Unlike the
docs version, families may hold different prompt counts: each family contributes its mean
difference with equal weight, and ``balanced``/``prompts_per_family`` record any imbalance.
"""

import numpy as np


def astra_pass(row) -> int:
    return int(row.scores.get("astra_judge", {}).get("strict_pass") is True)


def paired_family_interval(
    rows, left: str, right: str, *, pass_fn, replicates: int = 20000, seed: int = 20261015
) -> dict:
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
        clusters.setdefault(family, []).append(pass_fn(a) - pass_fn(b))
    families = len(clusters)
    sizes = {len(v) for v in clusters.values()}
    differences = np.array([np.mean(clusters[f]) for f in sorted(clusters)])
    prompt_difference = float(np.mean([d for v in clusters.values() for d in v]))
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, families, size=(replicates, families))].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "difference": float(differences.mean()),
        "prompt_difference": prompt_difference,
        "ci95": [float(lo), float(hi)],
        "families": families,
        "prompts": len(a_rows),
        "balanced": len(sizes) == 1,
        "prompts_per_family": {f: len(v) for f, v in sorted(clusters.items())},
        "replicates": replicates,
        "seed": seed,
        "method": "paired family-cluster percentile bootstrap",
        "family_differences": {f: float(np.mean(v)) for f, v in sorted(clusters.items())},
        "degenerate_interval": bool(lo == hi),
    }
