"""Update current W&B display tables from unchanged saved evaluation rows."""
import json
from pathlib import Path

import wandb

from bufogen.workflows.loop_report import build_loop_report
from bufogen.models import frozen_model_presentations
from bufogen.models import generation_model_presentation
from bufogen.settings import project_path

root = Path("runs/forge-loop")
directory = root / "heldout-baselines"
receipt_path = root / "heldout-baseline-visualization.json"
receipt = json.loads(receipt_path.read_text())
report = build_loop_report(root, benchmark_dirs=[directory])
benchmark = report["benchmarks"][0]
assert benchmark["comparison_fingerprint"] == receipt["comparison_fingerprint"]
arms = frozen_model_presentations(benchmark["models"])
rows = [json.loads(line) for line in (directory / "scored.jsonl").read_text().splitlines()]
assert len(rows) == 300
entity, project = project_path().split("/")
with wandb.init(entity=entity, project=project, id=receipt["run_id"], resume="must") as run:
    run.name = "google/gemma-4-26B-A4B-it vs gpt-6.1-sol vs ft-google/gemma-4-26B-A4B-it"
    images = wandb.Table(columns=["model_id", "checkpoint", "prompt_id", "prompt", "image",
                                 "strict_pass", "constraint_coverage", "failure_reason", "trace"])
    for row in rows:
        identity = generation_model_presentation(row, config=arms[row["backend"]])
        score = row["scores"]["scene_quality"]
        png = row.get("png_path")
        images.add_data(identity["model_label"], (identity["model_version"] or "").removeprefix("wandb-artifact:///"),
                        row["prompt_id"], row["prompt"], wandb.Image(png) if png and Path(png).is_file() else None,
                        score["status"] == "ok" and not score["quality_failure"],
                        score.get("constraint_coverage"), score.get("reason") or row.get("error"),
                        row.get("conversation_url"))
    metrics = wandb.Table(columns=["model_id", "attempts", "strict_passes", "strict_pass_rate",
                                  "constraint_coverage_scored_only", "generation_errors"])
    for key, score in benchmark["arms"].items():
        identity = benchmark["model_presentations"][key]
        metrics.add_data(identity["model_label"], score["total"], score["provisional_passes"],
                         score["provisional_pass_rate_all_attempts"],
                         score["metric_means_scored_only"]["constraint_coverage"], score["generation_errors"])
    run.log({"heldout_images":images, "heldout_summary":metrics,
             "heldout_pass_rate":wandb.plot.bar(metrics, "model_id", "strict_pass_rate", title="Strict scene pass rate — all attempts"),
             "constraint_coverage":wandb.plot.bar(metrics, "model_id", "constraint_coverage_scored_only", title="Visible requirement coverage — scored images")})
    receipt["display_model_ids"] = [benchmark["model_presentations"][k]["model_label"] for k in benchmark["arms"]]
    receipt["display_revision_note"] = "Model ID labels updated; source generations, scores, failures and comparison fingerprint unchanged."
    receipt_path.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"run_url":run.url, "rows":len(rows), "display_model_ids":receipt["display_model_ids"]}), flush=True)
