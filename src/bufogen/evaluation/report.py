from collections import defaultdict
from pathlib import Path
from statistics import mean

from bufogen.generation import BufoGeneration
from bufogen.models import generation_model_presentation


def compare_themes(rows: list[BufoGeneration], *, weak_themes: set[str] | None = None) -> str:
    weak_themes = weak_themes or set()
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row.theme, row.backend][row.prompt_id] = row.scores.get("vlm_judge", {}).get(
            "overall", 0.0
        )
    themes = sorted({r.theme for r in rows} | weak_themes)
    backends = sorted({r.backend for r in rows} - {"open_base"})
    labels = {r.backend: generation_model_presentation(r)["model_label"] for r in rows}
    lines = [
        "# Bufo evaluation",
        "",
        "Paired VLM overall scores on identical prompt IDs. Positive deltas favor the candidate.",
        "",
        "| Theme | Candidate | Paired prompts | Baseline | Candidate | Delta | Flagged weak |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for theme in themes:
        for backend in backends or ["open_tuned"]:
            base, candidate = grouped[theme, "open_base"], grouped[theme, backend]
            common = base.keys() & candidate.keys()
            if not common:
                lines.append(
                    f"| {theme} | {labels.get(backend, 'Unconfigured model')} | 0 | — | — | No paired evidence | {theme in weak_themes} |"
                )
                continue
            b, c = mean(base[p] for p in common), mean(candidate[p] for p in common)
            lines.append(
                f"| {theme} | {labels[backend]} | {len(common)} | {b:.3f} | {c:.3f} | {c - b:+.3f} | {theme in weak_themes} |"
            )
    lines += [
        "",
        "## Validity and latency",
        "",
        "| Backend | Valid renders | Mean latency |",
        "|---|---:|---:|",
    ]
    for backend in sorted({r.backend for r in rows}):
        subset = [r for r in rows if r.backend == backend]
        lines.append(
            f"| {labels[backend]} | {sum(r.valid for r in subset)}/{len(subset)} | {mean(r.latency_s for r in subset):.1f}s |"
        )
    lines += [
        "",
        "A small demo set is directional evidence, not a statistical claim of model improvement.",
    ]
    return "\n".join(lines) + "\n"


async def evaluate_checkpoints(
    prompts: list[dict], backends: dict, output_dir: Path, *, weak_themes=None
):
    from bufogen.evaluation.suite import evaluate_manifest
    from bufogen.workflows.pipeline import run_pipeline

    rows = await run_pipeline(prompts, list(backends), output_dir, backends=backends)
    rows = await evaluate_manifest(rows, output_dir / "eval")
    path = output_dir / "report.md"
    path.write_text(compare_themes(rows, weak_themes=weak_themes))
    return path
