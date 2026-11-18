import asyncio
import json
from pathlib import Path

from bufogen.generation import BufoGeneration
from bufogen.models import generation_model_presentation


class ScoringSuite:
    def __init__(self):
        from bufogen.evaluation.scorers.clip_score import ClipScore
        from bufogen.evaluation.scorers.svg_validity import SvgValidity
        from bufogen.evaluation.scorers.vlm_judge import VLMJudge

        self.validity, self.clip, self.judge = SvgValidity(), ClipScore(), VLMJudge()

    async def score(
        self, row: BufoGeneration, *, reference_png: Path | None = None
    ) -> BufoGeneration:
        output = row.to_dict()
        output["scores"] = {}  # Explicit rescoring never trusts previous scores.
        if reference_png is not None:
            output["reference_png"] = str(reference_png)
        validity = self.validity.score(output)
        output["valid"] = validity["valid"]
        clip = await asyncio.to_thread(self.clip.score, output)
        judge = await self.judge.score(output)
        row.scores = {"svg_validity": validity, "clip_score": clip, "vlm_judge": judge}
        return row


async def evaluate_manifest(rows: list[BufoGeneration], output_dir: Path) -> list[BufoGeneration]:
    if not rows:
        raise ValueError("No generations to evaluate. Run the generation stage first.")
    from bufogen.integrations.telemetry import init_weave

    init_weave()
    import weave

    suite = ScoringSuite()
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored.jsonl"
    # Write incrementally to retain completed judgments if a later API request fails.
    with scored_path.open("w") as file:
        for row in rows:
            await suite.score(row)
            file.write(json.dumps(row.to_dict()) + "\n")
            file.flush()

    @weave.op()
    def recorded_generation(generation: dict) -> dict:
        return generation

    summaries = {}
    for backend in sorted({r.backend for r in rows}):
        subset = [r for r in rows if r.backend == backend]
        model_label = generation_model_presentation(subset[0])["model_label"]
        dataset = weave.Dataset(
            name=f"bufo-{model_label}", rows=[{"generation": r.to_dict()} for r in subset]
        )
        evaluation = weave.Evaluation(
            name=f"bufo-{model_label}-evaluation",
            dataset=dataset,
            scorers=[suite.validity, suite.clip, suite.judge],
        )
        summaries[backend] = await evaluation.evaluate(recorded_generation)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    return rows


async def evaluate_identity_manifest(
    rows: list[BufoGeneration], output_dir: Path
) -> list[BufoGeneration]:
    """Score real renders, attach production feedback, then compare in Weave Evals.

    No CLIP/GPU installation is needed. Scores label quality failures without
    removing any generations, including invalid renders, from the evaluation.
    """
    if not rows:
        raise ValueError("No generations to evaluate. Run the generation stage first.")
    import weave

    from bufogen.evaluation.scorers.bufo_identity import BufoIdentity
    from bufogen.integrations.observability import record_generation, visual_generation
    from bufogen.integrations.telemetry import init_weave

    client = init_weave()
    scorer = BufoIdentity()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scored.jsonl").open("w") as file:
        for row in rows:
            if not row.weave_call_id:
                record_generation(row)
            if not row.weave_call_id:
                raise RuntimeError("Weave did not create a visual generation trace.")
            client.flush()
            call = client.get_call(row.weave_call_id)
            # Fresh judgment attached as native scorer feedback to the production
            # image trace; the evaluation below replays the saved result.
            result = await call.apply_scorer(scorer, {"rescore": True})
            row.scores["bufo_identity"] = result.result
            file.write(json.dumps(row.to_dict()) + "\n")
            file.flush()

    summaries = {}
    evaluation_refs = {}
    for backend in sorted({row.backend for row in rows}):
        subset = [row for row in rows if row.backend == backend]
        model_label = generation_model_presentation(subset[0])["model_label"]
        dataset = weave.Dataset(
            name=f"bufo-identity-{model_label}",
            rows=[{"prompt": row.prompt, "generation": row.to_dict()} for row in subset],
        )
        evaluation = weave.Evaluation(
            name=f"bufo-identity-{model_label}-evaluation", dataset=dataset, scorers=[scorer]
        )
        evaluation_refs[backend] = weave.publish(evaluation).uri()
        summaries[backend] = await evaluation.evaluate(visual_generation)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    (output_dir / "evaluations.json").write_text(json.dumps(evaluation_refs, indent=2))
    client.flush()
    return rows
