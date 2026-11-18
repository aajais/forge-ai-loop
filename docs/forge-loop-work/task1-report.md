# Task 1 — Authentic trajectory capture and reviewed export

Status: implementation complete; initial root-assigned independent review feedback addressed; final reviewer refresh pending. No paid model calls, Git operations, or external publication were performed by this worker.

## Changed files

- `src/bufogen/generation.py`: optional `trajectory` and `training_review` fields. Old JSONL rows without either field remain readable through dataclass defaults.
- `src/bufogen/agent.py`: freezes instructions once; constructs the actual SDK tools once and captures their `params_json_schema`; persists the final cumulative `result.to_input_list()` across the repair loop. Uses authenticated history reconstruction instead of the legacy synthetic `previous_drawing_*` calls.
- `src/bufogen/trajectories.py`: new portable text/function-tool conversion, snapshot capture, continuation input, and strict export contract.
- `src/bufogen/curate.py`: imports provenance-bearing review decisions; applies explicit review and split gates; uses authentic export instead of raw SVG assistant messages; records skip diagnostics. Legacy automatic metric thresholds are optional opt-in gates, not a prerequisite for a genuinely reviewed target.
- `tests/test_trajectories.py`: capture, repair, provenance, invalid association, image retention, legacy, mixed-settings/model, and export tests.
- `tests/test_training_contract.py`: score-only exclusion, decision import, review identity/evidence/timestamp validation, explicit negative review, held-out family/ID exclusion, legacy diagnostics, and score-free reviewed acceptance.
- `tests/test_learning.py`: updates the obsolete raw-SVG training expectation to actual tool calls plus the real friendly final reply.

## Capture and export contract

`trajectory` schema version 1 contains frozen `system_instructions`, OpenAI-compatible `tools`, portable `messages`, `provenance`, and `diagnostics`. Provenance identifies SDK capture source, prompt/model/group/version, actual final output, context policy, authenticated-history status, cumulative `source_models`, history-snapshot compatibility, and `prior_conversation`.

The input/output snapshot comes from the final SDK result, which already includes prior run input and repair attempts. It is not concatenated with older complete snapshots. Parallel function calls become one assistant tool-call message with separate matching tool responses. Call IDs, original compact SVG arguments, failed render results, repair feedback, and final model text survive. Renderer-injected canonical geometry and application media publication are not invented as model actions. Hidden reasoning items are excluded.

`export_training_example(row)` returns messages/tools, source generation/group/model identifiers, prompt/version/theme and provenance. It rejects missing legacy trajectories, unauthenticated history, orphan/duplicate/unresolved calls, undeclared tools, assistant raw SVG, missing/expanded/overlong SVG arguments, failed final rendering, composition warnings, mismatched provenance/system/final output, and unsupported content conversion.

Multimodal SDK inputs are retained in `source_items` with an explicit unsupported-export diagnostic. Follow-up chats can replay those original image inputs and actual tool calls; no attachment is silently discarded. They cannot enter the v1 text/tool SFT export.

Incomplete SDK snapshots (including old test doubles that return an empty `to_input_list()`) retain the actual submitted input for continuation, are explicitly marked unauthenticated, and cannot train. Existing legacy rows retain conversational text and editable SVG as *application-supplied user context*, never fabricated tool calls; their descendants remain nonexportable.

Follow-ups preserve model selection and chat usability. All captures containing prior conversation history are nonexportable in version 1: review of the latest image cannot qualify earlier unreviewed, rejected, or held-out assistant targets. Repairs within one generate call remain eligible. Mixed-model, prompt-version, system-instruction, and tool-schema provenance is additionally recorded and rejected, with exclusions persisting into descendants.

## Review-decision schema

`import_review_decisions(rows, jsonl_path)` validates and joins by generation ID, returning new dataclass rows without modifying the supplied objects. Unknown/duplicate IDs fail closed. Each JSONL object requires:

```json
{
  "schema_version": 1,
  "source_generation_id": "actual-generation-id",
  "prompt_id": "actual-prompt-id",
  "decision": "accept",
  "reviewer_kind": "human",
  "reviewer_id": "actual-reviewer-id",
  "reviewed_at": "2026-10-15T14:00:00+00:00",
  "source_ref": "actual-review-source-reference",
  "evidence": "Actual visual review findings",
  "partition": "train",
  "prompt_family": "assigned-family"
}
```

Allowed decisions: `accept`, `reject`. Reviewer kinds: `human`, `aria_reviewed`, `assistant` (assistant review explicitly authorized by the user via root). Allowed partitions: `train`, `validation`, `test`, `heldout`; only `train` can be selected. The sample above documents schema only and is not a review decision for any real output.

`select_examples` requires an explicit valid acceptance, successful render and authentic export; excludes supplied held-out IDs/families and nontraining partitions; deduplicates prompt/drawing pairs. Missing review and negative review cannot train even with perfect automatic scores. `min_judge` and `min_clip` default to `None`; callers may explicitly enable those legacy thresholds. Scores and original review evidence remain in exported metadata. A caller-supplied `diagnostics=[]` collects each skipped generation ID and a reason.

## Verification evidence

Test-first observations before implementation:

- Initial trajectory suite: 9 expected failures from the missing trajectory module/field, including legacy rejection, associations, raw SVG, repair persistence and continuation.
- Review contract additions: 4 expected failures (missing decision import/diagnostics API and automatic-only acceptance still occurring).
- Incomplete snapshot regression: failing test for unsupported `expected_input` parameter, then green implementation; existing screenshot-follow-up test also reproduced missing historical image context before the fix.
- Optional automatic metrics: failed because a genuinely reviewed row without legacy scores was excluded; passed after defaults changed to optional thresholds.
- Mixed settings/models: each new test failed before the history-snapshot/lineage support existed, then passed after implementation.
- Independent-review ancestor qualification regression: a same-model/same-settings follow-up initially exported without raising; explicit prior-conversation rejection made the test pass. This prevents latest-row acceptance from laundering earlier unreviewed, rejected or held-out assistant targets.

Final command:

```text
.venv/bin/pytest -q tests/test_trajectories.py tests/test_training_contract.py tests/test_pipeline.py tests/test_ui.py tests/test_learning.py
42 passed in 1.08s
```

This includes the exact requested four-file suite plus the changed learning tests. Existing pipeline/UI tests cover PNG persistence, concurrent/session behavior, screenshot propagation and follow-ups.

```text
.venv/bin/ruff check src/bufogen/trajectories.py src/bufogen/agent.py src/bufogen/generation.py src/bufogen/curate.py tests/test_trajectories.py tests/test_training_contract.py tests/test_learning.py
All checks passed!
```

Changed Python files also formatted with Ruff. Local installed SDK code confirmed `RunResult.to_input_list()` cumulative semantics and `FunctionTool.params_json_schema`/`strict_json_schema`. No synthetic fixture was represented as a real teacher or review result.

## Integration concerns / deliberate limits

1. Initial independent review identified ancestor target qualification; the fix and regression test are implemented. Root/reviewer must refresh the final changes; this report does not claim final review approval.
2. Real provider capture and actual Forge upload still need the root-owned bounded live pilot. This task ran no live paid calls.
3. Multimodal SFT conversion intentionally fails closed; image-bearing conversations remain usable and preserved. A future image-training contract must explicitly support retained attachments.
4. The old `curate()` function still generates new rows before importing a supplied review file. Reviews for old IDs cannot match newly generated IDs. Use `Manifest.read()` → `import_review_decisions()` → `select_examples()` for export of an existing reviewed queue; Task 5 is expected to expose this as the end-to-end route. `curate()` now writes `curation-diagnostics.json` and fails without reviewed examples instead of silently inventing acceptance.
5. The legacy `prompting.generation_input()` helper is unchanged outside this worker's ownership; only its existing tests call it now. Production `BufoAgent` uses the authentic `trajectories.conversation_input()` helper. Future callers must use the latter to avoid synthetic historical tool actions.
6. Review provenance is an explicit audit record, not a cryptographic identity verification system. Callers must supply real decisions and the correct held-out IDs/families from the partition manifest. No review labels were manufactured here.
7. Local exports deliberately include richer provenance/source fields. Task 3 maps them to Forge's supported upload fields rather than sending the full local record blindly.

8. Reviewed prior-conversation targets are deliberately unsupported in v1, even when all turns use the same model/settings. Supporting those later requires checking all target-bearing ancestors against their review/split records or explicit per-turn training masks; simply accepting the latest PNG is insufficient. Single-request render/repair trajectories support the initial pilot.
