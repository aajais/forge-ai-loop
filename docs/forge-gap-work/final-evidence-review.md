# Independent final evidence review

Reviewed the completed experiment's `results.md`, README outcome, and local
showcase HTML, JSON, review export, and assembly receipt. **No substantive
correctness finding** was identified in the reviewed snapshot. The remaining
publication transition is called out below; this review does not cover the
in-progress LR runner or authorize a new training job.

## Evidence verified

- Every source and derived-output hash in the showcase assembly receipt
  matched its current file.
- Rebuilt the report data read-only with `build_loop_report`; all four
  benchmarks passed its frozen manifest, scored-row binding, coverage, and
  recomputed-summary validation. Saved benchmark arms, rows, models, model
  labels, fingerprints, and run IDs matched source evidence exactly.
- The 152 final review records exactly matched `training_review` in the final
  adjudicated source manifest, and all review PNG hashes matched actual files.
  Counts are 124 accepted train, 23 accepted validation, and five rejected.
- All 493 retained benchmark PNGs matched their manifest-bound checksums.
  The HTML's 103 gallery images matched the exact expected first-12-prompt
  selection, row order, PNG bytes, model label, and prompt. Its additional
  image matched the separate RL serving receipt's PNG, SHA256
  `12d45ff68e238119afe2dc036ddd3a387fe1bd4b44ed4e870423f9543d3c8574`.
- Local HTML links resolved. No SVG/base64 contents were dumped during review.
- Displayed model names use `gpt-6.1-sol`,
  `google/gemma-4-26B-A4B-it`, and `ft-google/gemma-4-26B-A4B-it`.
  Tuned artifact versions distinguish the candidates. Internal backend keys
  remain in provenance rather than replacing the user-facing model labels.

## Interpretation verified

Confirmation evidence supports 65/80 teacher passes, 2/80 base passes, and
7/80 selected one-epoch passes, with all failed attempts retained. The reported
6.25-point improvement, interval [0, 13.75], and remaining 72.5-point teacher
gap, interval [57.5, 86.25], match the paired-results receipt. Its confirmation
hashes match the frozen files. The documentation correctly describes the
family-cluster bootstrap as approximate and does not claim population parity.

Coverage means retain their scored-only denominators, including 76 available
base images versus 80 selected-checkpoint images. The lower selected-model
coverage is disclosed. The 0/16 training-fit result is explicitly an in-sample
diagnostic. Assistant reviews are not represented as human approvals, and the
teacher/judge identity and incomplete independent calibration are disclosed.

The earlier RL pilot's immutable `v2`, completed backend step 2, serving PNG,
and Registry-link failure remain separate from the expanded-data SFT lineage.
The presentation explicitly makes no claim of measured RL quality improvement.
Rank one is presented as a capacity hypothesis; no rank-32 model, new training
job, or unsupported access is claimed.

## Publication transition to finish

The reviewed assembly was timestamped `2026-10-15T20:02:00.794271+00:00` and
truthfully marked confirmation publication pending. During this review, the
actual native receipt completed with 240 feedback entries, three evaluations,
and no publication errors. The saved showcase JSON still contains the earlier
empty confirmation evaluation list; rebuild the showcase from completed
receipts, rather than changing only its banner, so the three evaluation links
are included.

Once the W&B PNG comparison publisher has a verified completed receipt, replace
the pending text and add its actual run URL in the results, showcase, and
README as appropriate. Regenerate assembly hashes after all derived output
changes. These are expected finalization steps, not evidence falsification in
the reviewed snapshot. Preserve original benchmark manifests, scores,
selection, candidate states, and RL failure/recovery receipts.

Only this review note was written. No runtime, data source, model route,
publication, or training state was changed by the review.

## Root finalization verification

After the independent review above, the root agent completed the publication
transition. Publisher exited successfully with W&B run `504b5fe7`, 240 rows and
artifact `wandb-smle/bufo-gen/bufo-gap-confirmation-evidence:v0`. The showcase JSON
was rebuilt from all three completed native evaluations, its HTML gained the
verified comparison link, and assembly hashes were regenerated. The prior
assembly receipt is retained as `assembly-receipt-before-publication.json`.
Root verification checked every current evidence hash, both derived file hashes,
three native confirmation evaluations and zero publication errors. This addendum
records root verification separately from the independent snapshot review.
