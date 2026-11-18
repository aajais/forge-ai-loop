# Independent review of the follow-up evaluator

Reviewed 2026-10-15. Scope: `evaluate_lr_candidates.py`,
`tests/test_lr_evaluation.py`, and the existing benchmark/publication helpers
they call. Read-only review; no remote jobs, generation, scoring or runtime edits.
The initial 20 focused evaluator tests passed. After the correction below,
all 27 focused evaluator tests passed during independent re-review.

## Finding resolved after re-review

**P2 — A transient feedback publication failure prevents successful completion
even after a successful resume.**

The evaluator rejects any nonempty `weave-publication.json.errors` after native
publication. The shared benchmark `_publish` records per-generation exceptions
under `errors[generation_id]`. On a successful later retry it records
`feedback[generation_id]`, but does not remove that earlier error. Only
`evaluation:<arm>` errors are explicitly cleared on success. Consequently a
successfully resumed run remains rejected forever by the new completion check.
The check's message currently recommends resuming, which cannot resolve this
saved state on its own.

The standalone completion validator should distinguish unresolved failures from
historical errors whose exact generation IDs now have successful feedback
receipts, while preserving the history. It should also compare feedback ID
membership against the actual generated rows, not just equal cardinality. This
allows recovery without changing the frozen generation runtime or deleting
evidence. Add an offline regression for a stale error followed by complete
successful feedback, plus a wrong-ID/same-count feedback rejection.

This finding was sent to the code owner before this report. The owner fixed it
in the standalone evaluator; no runtime file was changed. Independent re-review
confirmed that `reconcile_publication` requires the exact generation-ID set,
nonempty unique call IDs, and matching recorded trace IDs where present. Only
generation-keyed historical errors with those successful receipts are resolved;
unknown and evaluation errors remain blocking. The original publication file is
preserved, and an exclusive reconciliation record binds its SHA256 plus the
manifest SHA256. Existing reconciliation records must match exactly on resume.

New regressions cover successful recovery without history deletion, equal-count
wrong-ID feedback, mismatched/empty/duplicate call IDs, and unresolved unknown or
evaluation errors. The P2 finding is resolved; no blocking findings remain in
the reviewed scope.

## Other reviewed boundaries

No additional blocking issue was found in these paths:

- Training-fit uses the frozen 16 training base IDs and verifies accepted exact
  targets; validation uses the exact 24 frozen validation prompts. Output roots
  cannot overlap the original experiment. No confirmation or selection stage is
  exposed, and historical results are not pooled with new samples.
- Actual candidate checks cover deployed status, base identity, exact requested
  recipe, source/upload hashes, row counts, immutable artifact version and live
  Forge/W&B identity. The original control's exact artifact/state is retained.
  The accepted training-plan adapter also checks its source/control bindings and
  required candidate lineage. A generic future declaration does not submit or
  assume a working rank route; actual receipts must match before generation.
- Exactly three distinct immutable adapters are evaluated, all displayed as
  `ft-google/gemma-4-26B-A4B-it` with separate internal arm IDs. Common generation
  source, policy, judge and serving fields must match the frozen comparison.
- Protocols are saved before generation and must match on resume. Existing peer
  stage protocols require identical candidate identity, environment and common
  configuration. The benchmark retains interrupted attempts as failures rather
  than automatically replacing them.
- Complete manifest/scored coverage and score-to-generation bindings are
  revalidated through `_benchmark`. Missing images and generation failures
  remain in the all-attempt denominator; scored-only coverage remains labeled.

The publication correction is approved for interrupted-run recovery within the
reviewed receipt contract. This audit does not authorize submission of
the optional LR experiment while the higher-rank route is being clarified.
