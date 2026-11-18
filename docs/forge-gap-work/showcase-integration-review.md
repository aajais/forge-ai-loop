# Final showcase integration review

The completed comparison publisher and `write_loop_report` can be used without
changing the frozen experiment runtime. As of this review, `selection.json` is
absent; do not label either deployed candidate as selected yet.

After selection and confirmation finish, the following local-only assembly uses
the actual selected Forge receipt and final adjudicated reviews. It writes
derived presentation files, preserving all original evidence:

```python
import json
from pathlib import Path
from bufogen.workflows.loop_report import write_loop_report

root = Path("runs/forge-gap")
output = root / "showcase"
output.mkdir(parents=True, exist_ok=True)
selection = json.loads((root / "selection.json").read_text())
candidate_dir = {
    "open_tuned_epoch1": "e1",
    "open_tuned_epoch3": "e3",
}[selection["selected_arm"]]
forge_state = root / "training" / candidate_dir / "forge-state.json"
state = json.loads(forge_state.read_text())
assert state["model_artifact_ref"] == selection["selected_artifact"]
assert state["entry_counts"] == {"train": 124, "val": 23}

rows = [json.loads(line) for line in
        (root / "training/reviewed-manifest.jsonl").read_text().splitlines()
        if line.strip()]
reviews = [row["training_review"] for row in rows]
assert len(reviews) == len({r["source_generation_id"] for r in reviews}) == 152
review_path = output / "final-adjudicated-reviews.jsonl"
review_path.write_text("".join(json.dumps(r) + "\n" for r in reviews))

write_loop_report(
    root,
    output,
    benchmark_dirs=[
        root / "teacher-curriculum",
        root / "validation-candidates",
        root / "confirmation-selected",
    ],
    forge_state=forge_state,
    review_paths=[review_path],
    rl_state=root / "rl-direct-result.json",
)
```

The function accepts `calibration_path` if historical calibration should also be
shown. `runs/forge-loop/calibration/rl-calibration.json` belongs to the earlier
pilot; identify it as historical evidence, not calibration of the new candidate
comparison.

Presentation caveats:

- There are 152 final reviews: 124 accepted training targets, 23 accepted
  validation targets, and five rejects. The report aggregates 147 accepted
  targets across partitions; do not describe these as 147 training examples.
  Original decisions plus adjudications must not both be passed as review
  sources: the API correctly rejects conflicting decisions for one generation.
- The RL recovery receipt has no `status` field. This does not cause an API
  error or an incorrect failure label: the Improve card uses the Forge state
  status, and embeds the complete RL receipt under Evidence. Explicitly state
  that the earlier pilot lineage reached backend step 2, immutable `v2`, and
  served a valid PNG; Registry linking failed and RL quality improvement was
  not measured. It is separate from the new epoch-1/epoch-3 SFT candidates.
- The nested RL job's `status: created` is its original submission receipt, not
  a current backend status. Preserve that receipt and explain completion using
  `training_completed`, `completed_step`, and the verified immutable artifact.
- `publish_comparison.py` writes `validation-visualization.json` and
  `confirmation-visualization.json`. The current loop report reads older
  `pilot-visualization.json` and `heldout-visualization.json` names. Link the new
  receipts' actual W&B run URLs explicitly in the showcase documentation.
  Likewise link the verified RL serving receipt's conversation URL and PNG;
  a local receipt path alone is not rendered as a clickable evidence link.
- Benchmark metrics keep their own denominators and fingerprints; the gallery
  shows at most 12 prompts from each run. Use the publisher's complete W&B image
  tables for all matched cases. Do not imply that the small gallery is complete
  evaluation coverage or that the RL checkpoint was one of those evaluated
  candidate arms.

Reviewed sources: `docs/forge-gap-work/publish_comparison.py`,
`src/bufogen/loop_report.py`, the selection receipt schema, the final reviewed
manifest, both deployed Forge state receipts, and `rl-direct-result.json`.
No report was published, no runtime was edited, and no training was launched
during this review.
