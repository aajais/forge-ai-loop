# Fixed blind validation diagnostic

An assistant reviewer inspected all 16 predefined attempts with model IDs and automated judge scores hidden. The sample was fixed before review: the base variant of `airport_marshaller`, `florist_arranging`, `ice_cream_cart`, and `locksmith`, for each of four model arms. Three missing generations remained visible as failure placeholders. Original decisions were frozen before unblinding and were not edited.

| Model ID | Separate checkpoint metadata | Strict passes / attempts | Missing images | Requirements met / all required | Recognizable Bufo / generated images | Connected body / generated images |
|---|---|---:|---:|---:|---:|---:|
| `gpt-6.1-sol` | — | 4/4 | 0 | 13/13 | 4/4 | 4/4 |
| `google/gemma-4-26B-A4B-it` | — | 0/4 | 0 | 1/13 | 4/4 | 4/4 |
| `ft-google/gemma-4-26B-A4B-it` | 1 epoch, artifact `bufo-gemma-gap-e1-20261015:v1` | 0/4 | 0 | 2/13 | 4/4 | 4/4 |
| `ft-google/gemma-4-26B-A4B-it` | 3 epochs, artifact `bufo-gemma-gap-e3-20261015:v3` | 0/4 | 3 | 1/13 | 1/1 | 1/1 |

All 13 generated images contained a recognizable, connected Bufo according to this reviewer. The strict failures were instead missing or detached props, unclear grips, missed stem/vase, key/keyhole or scoop/cone contacts, and airplane/cart shapes that were not sufficiently recognizable. This distinction supports keeping character identity and prompt-specific requirements as separate measurements.

The three-epoch candidate's coverage among generated images alone was 1/3 requirements, versus 1/13 when all four attempts are retained. Reporting only its generated image would conceal three failures; the main comparison retains the four-scenario denominator. Identity and connectivity use generated-image denominators explicitly, because no character is observable in a missing image.

This is a diagnostic sample of **four scenarios per arm**, not population calibration, a statistically reliable ranking, or proof of a fine-tuning improvement. The one-epoch candidate's additional satisfied requirement is only a sample observation. This review does not override the preregistered full-validation selection rule, and these validation prompts are not the untouched confirmation evaluation.

## Artifacts and provenance

- Local side-by-side review: `runs/forge-gap/validation-blind-review/comparison.html`. It shows actual images or the original failure placeholders, exact model IDs, separate epoch labels, and the reviewer's requirement evidence.
- Machine-readable unblinded counts and per-scenario evidence: `runs/forge-gap/validation-blind-review/diagnostic-summary.json`.
- Immutable original blind decisions: `runs/forge-gap/validation-blind-review/reviewer/decisions.jsonl`, SHA256 `da11b8b82cc047f8baf6db2b30f417b632829e20c685ce39ebcc89827cc57f3f`.
- Reviewer inputs SHA256: `2b3da5051df2a48716da1df7ff222d1580c4c8898f078bb0d6b1b727306d2306`.
- Separate private mapping SHA256: `39a1291d96aa57f030d1faca6140c3ce262129b9bd89f9210405663332dcd4fe`.
- Frozen source benchmark SHA256: `9432a9114b465236accf4b665374da97510ded7973baf099c0afebe7177aebea`.

The summary additionally binds the packet receipt, source manifest at unblinding, selected source provenance, each original available PNG, and each reviewer PNG. Validation checked the 16 unique scenario/arm pairs, exact requirement order, strict-pass consistency, all hashes, and the link between packet IDs and original generation IDs. No automated judge scores were consulted to create this diagnostic.
