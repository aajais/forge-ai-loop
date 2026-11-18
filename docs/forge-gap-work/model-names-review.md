# Model naming review

Reviewed `model-names.diff`, `model-names-report.md`, the current naming helper, UI serialization/JS, report and evaluation changes, and naming tests. Review was read-only apart from this report. Concurrent RL changes in `agent.py` were outside scope.

## Verdict

**Approved after correction.** The configured direct-model UI paths implement the requested names correctly, and the two findings below were corrected on re-review.

### Follow-up verification

The alias branch now applies the shared presentation function to actual resolved serving IDs. A resolved artifact uses its recorded base for the `ft-` label and preserves the artifact URI separately. Unknown lineage, including the alias itself erroneously occupying the base field, explicitly displays `ft-model (base ID unrecorded)` rather than inventing a base. Multiple resolved IDs remain represented. The two showcase sentences now identify `gpt-6.1-sol` for the completed pilot instead of the shorthand.

Reviewer reran `.venv/bin/pytest -q tests/test_model_names.py`: **8 passed**. Added coverage includes a resolved artifact with known base, missing base, and alias-valued base. The initial findings below are retained as review history; neither remains blocking for this implementation.

## Findings

1. **Medium: a resolved Forge adapter loses fine-tuned presentation.** In `src/bufogen/models.py`, the early `alias` branch returns `resolved_serving_models` directly as the label and always sets `model_version=None`. A real adapter URI therefore appears as the model's name instead of `ft-<base ID>`, even if recorded base identity exists. Reproduced with a `forge_proxy` generation whose recorded resolved model is `wandb-artifact:///t/p/a:v1` and recorded base is `google/gemma-4-26B-A4B-it`; result was `{'model_label': 'wandb-artifact:///t/p/a:v1', 'model_version': None, 'routing_alias': 'bufo-gen'}`. The current base route is unaffected. Handle resolved artifact IDs through the same adapter presentation logic, keeping actual artifact/alias separate. If lineage is unknown, display an explicit unknown-base label rather than guessing the current base. Preserve multiple resolved identities if a conversation's routing really changed.

2. **Low: generated showcase prose retains old shorthand.** `src/bufogen/loop_report.py` still emits “no Sol-parity claim” and “Sol teacher and Sol judge share a model family” in the report payload. Replace with recorded IDs or provider-neutral prose that does not become wrong under environment overrides. Historical source evidence should remain unchanged.

## Verified behavior

- `OPEN_MODEL` and `SOL_MODEL` overrides propagate to current status labels and trace agent names.
- `OPEN_TUNED_BASE_MODEL` and `OPEN_RL_BASE_MODEL` determine `ft-` labels independently of the serving artifact URI. Exact request model IDs are not rewritten.
- Direct fine-tuned choices show the checkpoint/version separately, including a tuning-method distinction between SFT and SFT+RL.
- New trajectories preserve base-model identity and tuning method. Recorded identity is preferred over the current environment for historical UI rows.
- Historical frozen benchmark data is used to derive presentation labels; source benchmark files are not rewritten by report generation. Existing tests assert their bytes remain unchanged.
- Forge aliases stay explicitly pending until actual response model evidence exists; a pending trace root does not invent a serving model.
- Current generated galleries, comparison tables, and evaluation metadata use model IDs. Internal arm IDs remain stable grouping keys.

## Verification performed by reviewer

- `.venv/bin/pytest -q tests/test_model_names.py tests/test_benchmark.py tests/test_forge_cli.py`: **38 passed**, existing Weave/Pydantic deprecation warnings only.
- `node --test tests/chat_ui.test.cjs`: **4 passed**.
- Read-only reproduction above confirms the routed-adapter edge case.

No browser visual verification or remote W&B table publication was performed in this review. The parent is handling updated remote visualization separately.
