# Model ID presentation

Implemented 2026-10-15. Runtime ownership released to the parent before its next
training experiment; this change does not alter system prompts, rendering,
scorers, generation budgets, actual model request IDs, or immutable checkpoints.

## Result

- Base and reference model labels are their actual configured IDs, initially
  `google/gemma-4-26B-A4B-it` and `gpt-6.1-sol`.
- Fine-tuned labels use `ft-` plus the base ID. `OPEN_TUNED_BASE_MODEL` and
  `OPEN_RL_BASE_MODEL` explicitly configure this identity independently of the
  serving artifact. Their defaults match this project's trained base.
- Dropdown entries, responses and report tables show the immutable artifact URI
  separately. SFT and SFT+RL remain distinguishable even with the same ft label.
- Trace agent names now contain the display model ID instead of internal arm keys.
  New trajectories record base-model identity and tuning method. Neither the
  response provenance's actual model nor its requested serving URI is replaced.
- Forge remains explicitly labeled as a routing alias until the response includes
  resolved serving model evidence. History serialization then displays the actual
  resolved ID. An agent-root span started before resolution retains its explicitly
  pending alias label; no guessed model is presented as known.
- Current UI, generated HTML comparisons/showcase, Markdown comparison reports,
  new evaluation names/metadata, README and ARIA instructions use model IDs.
  Weave sanitizes `/` in object URI names; the original model ID remains in the
  evaluation name field and explicit metadata.
- Internal arm keys (`open_base`, etc.) remain stable machine identifiers and CLI
  arguments. Frozen benchmark JSON, historical manifests and published remote
  traces/artifacts were not rewritten. Legacy frozen comparison labels derive
  from recorded model IDs and the paired recorded base; missing identity is shown
  as unrecorded rather than substituted from today's model environment.

## Verification

- New regression tests first failed for stale labels, missing response identity and
  internal-key trace names, then passed after implementation.
- Full suite: 270 passed (before the final new-evaluation-name-only change).
- Final targeted suite: 48 passed, including model status, historical response
  identity, tracing, benchmark reports, publication and CLI invariants.
- Node UI controller: 4 passed, including visible checkpoint version and historical
  model ID preservation when configuration changes.
- Repository Ruff check passed. Existing Weave/Pydantic deprecation warnings remain.
- Regenerated `runs/forge-loop/showcase/index.html` locally from saved evidence.
  This performs no paid generation or network mutation. No browser visual claim.

Review diff: `docs/forge-gap-work/model-names.diff`. Before copies and changed-file
inventory are adjacent. Source fingerprints will change for the next experiment;
the parent is preparing a fresh controlled comparison rather than pooling old
and new fingerprints.

Independent review follow-up: replaced remaining report prose references to Sol
with `gpt-6.1-sol`. Forge responses that resolve to an artifact now show an ft
label and preserve that artifact separately. A missing base ID, or a legacy
`base_model` value equal to the routing alias, produces the explicit unrecorded
base label; it never produces `ft-bufo-gen`. The Forge config no longer defaults
its base identity to its alias. Eight model naming tests and scoped Ruff passed
after these corrections. The latest models/report-only diff is
`model-names-reviewed-display.diff`; `agent.py` belongs to the RL worker now.
