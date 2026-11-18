# Task 3 independent review

Reviewed 2026-10-15 by `review_forge`. Read task brief, implementation report, full changed source/tests and supplied diff, and verified endpoint contracts against the researched `docs/forge-preflight.md`. No remote mutations or dependency changes were made.

## Spec decision: changes requested

The principal managed-Forge contract is implemented: exact Gemma student, explicit existing project identity, reviewed authentic tool trajectories, distinct validation mapping, family/group split exclusion, documented upload fields, dry-run before append, all assistant turns, persisted mutation intents, poll-only resume, no duplicate job retries after ambiguity, and actual deployed artifact handling. Direct captured Sol uploads are the specified supported alternative to unverified relabeling. The one issue below weakens required source lineage preservation.

1. **Medium — protect all authoritative lineage fields when mapping metadata** (`src/bufogen/forge.py:291–294`). `metadata.update(example['metadata'])` overwrites validated local `source_model`, `source_generation_id`, `partition`, `prompt_family`, and other source fields. Only `training_review` is restored. An otherwise valid reviewed export carrying nested metadata can therefore upload contradictory lineage. Reproduced using the actual reviewed Sol smoke: nested metadata changed uploaded source model to `different-model`, source generation ID to `unrelated`, and partition to `heldout`, while the row ID, provenance and actual split still identified the original Sol training row. This does not alter the actual training split, but breaks trustworthy downstream attribution and comparisons. Reject conflicting reserved keys or apply authoritative local fields after extension metadata. Add a regression showing authoritative fields cannot be overwritten and nonconflicting custom metadata survives.

## Quality decision: changes requested (same finding)

No separate blocking quality findings. The client/workflow boundary is clear, write uncertainty fails closed, errors omit credentials, and state locking prevents concurrent duplicate mutations. Existing source hashes and mapped hashes protect resume compatibility. Deliberately unsupported ambiguous-upload reconciliation is clearly documented rather than silently acknowledging an uncertain append. Existing job IDs are retained even for failed jobs; deployed-without-artifact is not promoted.

## Verification

- `.venv/bin/pytest -q tests/test_forge.py tests/test_learning.py`: **23 passed**.
- Ruff over changed Python source/tests: **all checks passed**.
- Actual local `trajectory-smoke` manifest + assistant review passed `import_review_decisions`, `select_examples`, and `reviewed_entries`: **one train row with five authentic messages**. This is a local mapping check, not a remote upload or completed training claim.
- Root independently reported actual client preflight success saved at `runs/forge-loop/client-preflight.json`; this review did not repeat that network operation.

After the metadata collision fix and focused regression pass, this task should be ready for live reviewed-dataset import and controlled training submission. Service entitlement/minimum data/deployment success remain live validation steps, not proven by these tests.

## Final re-review after correction

**Spec: PASS. Quality: PASS.** No outstanding findings.

The metadata mapping now validates custom metadata as an object and applies authoritative local source fields last. The new regression exercises conflicting model/generation/group IDs, partition, family, prompt/version and reviewer evidence together, and confirms unrelated custom annotations survive. Independently inspected the corrected source and regression, then reran both checks:

- `.venv/bin/pytest -q tests/test_forge.py tests/test_learning.py`: **24 passed**.
- Ruff over changed Python source/tests: **all checks passed**.

This supersedes the initial changes-requested decisions above; the earlier finding remains as review history. Ready for the parent's controlled live import/submission validation. No remote upload, training acceptance or deployment success is implied by this sign-off.
