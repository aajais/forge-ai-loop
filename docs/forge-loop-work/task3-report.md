# Task 3: managed Forge workflow

Implemented 2026-10-15. No remote mutations, training jobs, provider creation, routing changes, or Git operations were performed by this task. Pre-edit snapshots are in `docs/forge-loop-work/task3-before/`.

## Files and contract

- `src/bufogen/forge.py`: synchronous `httpx` client with explicit entity and project alias, injectable `httpx.MockTransport`, sanitized API errors, catalog/project preflight, explicit project import, dataset creation/validation/upload, and fine-tune create/get. Uses the documented endpoints in `docs/forge-preflight.md`.
- `src/bufogen/forge_loop.py`: atomic JSON state, nonblocking process file lock, source and mapped-upload hashes, persisted mutation intent, dataset/job identity, one-shot poll, deployed artifact capture, and explicit remote-ID reconciliation.
- `src/bufogen/curate.py`: `select_examples(..., allowed_partitions={'train', 'validation'})` permits explicitly reviewed validation rows. Default remains train-only; test/heldout partitions cannot be enabled.
- `.env.example`: managed Forge endpoint/alias/state settings. Existing routing is not changed.
- `tests/test_forge.py`, `tests/test_learning.py`: mocked external-boundary and curation checks.

### Minimal invocation after root reviews real PNGs

First produce a JSONL file containing `select_examples` exports, not raw generation manifests. Explicitly pass the authoritative held-out IDs/families both to selection and initialization. Validation families must be disjoint from training families.

```python
import os
from pathlib import Path
from bufogen.settings import required
from bufogen.integrations.forge import ForgeClient
from bufogen.workflows.forge_loop import (
    initialize_workflow, import_reviewed_dataset, submit_training, resume_training,
)

state_path = Path('runs/forge-loop/forge-state.json')
with ForgeClient(
    api_key=required('WANDB_API_KEY'),
    entity=required('WANDB_ENTITY'),
    project_alias=os.getenv('FORGE_PROJECT_ALIAS', 'bufo-gen'),
) as client:
    eligibility = client.preflight(
        project='bufo-gen',
        project_id='dd5ddbe2-b31c-43e1-ab39-b5488c78f3d8',
    )
    state = initialize_workflow(
        state_path, client,
        source_path=Path('runs/forge-loop/pilot/reviewed-examples.jsonl'),
        dataset_name='bufo-sol-pilot-20261015',
        model_name='bufo-gemma-sol-pilot-20261015',
        project='bufo-gen',
        project_id='dd5ddbe2-b31c-43e1-ab39-b5488c78f3d8',
        heldout_ids=heldout_ids,
        heldout_families=heldout_families,
        lineage={'preflight': eligibility},
    )
    state = import_reviewed_dataset(state_path, client)
    # Root explicitly triggers submission after reviewing import evidence.
    state = submit_training(state_path, client, epochs=1, batch_size=2)
    # Later calls poll once; no hidden loops or paid job retries.
    state = resume_training(state_path, client)
```

The example's source path is a caller-chosen local output, not a claim that that file already exists. The existing project ID is real and supplied by the parent. `preflight` checks the exact alias/entity/project/ID and model catalog. Catalog eligibility explicitly does not claim project-specific job acceptance. `initialize_workflow` performs local validation only; callers run preflight explicitly before remote mutations. Import never submits training.

### Public APIs

`ForgeClient(api_key=..., entity=..., project_alias=..., base_url=..., training_base_url=..., transport=None, timeout=60)` is a context manager; `.close()` is also available. Defaults are the official Model Distillation and Serverless Training bases. `.preflight(base_model=STUDENT_MODEL, project=None, project_id=None)` returns `project`, `base_model`, `catalog_eligible`, and `job_acceptance_verified=False`. `.setup_project(project=..., targets=[...])` is an explicit initial-routing mutation and is not used for this already-imported project.

`reviewed_entries(examples, heldout_ids=(), heldout_families=())` preserves actual messages/tools and moves rich local fields to metadata. It rechecks explicit accepted review identity/time/evidence, train/validation partitions, real tool-call integrity, frozen system prompt, capture provenance, source model consistency, final reply, family/group split separation, duplicate IDs and documented row/object limits. Validation maps to `val`. Held-out/test rows are rejected. It does not synthesize tool calls, review decisions, model IDs, or artifact references.

Workflow functions return the entire saved state dictionary:

- `initialize_workflow(state_path, client, *, source_path, dataset_name, model_name, project_id, project=None, base_model=STUDENT_MODEL, heldout_ids=(), heldout_families=(), lineage=None)` validates source before creating state. Repeated initialization only accepts the same frozen identity/configuration.
- `import_reviewed_dataset(state_path, client)` creates the dataset once, dry-runs exact mapped entries, then commits a single append and checks counts. Repeated completed imports return saved state without writes.
- `submit_training(state_path, client, *, epochs=1, batch_size=2)` requires completed import and trains every assistant turn (`assistant_turns='all'`). Existing job IDs are polled rather than recreated.
- `resume_training(state_path, client)` polls an existing ID once; never submits. Returns the actual `model_artifact_ref` only for a `deployed` response containing a nonempty reference. A deployed response without an artifact becomes `deployed_missing_artifact`.
- `reconcile_remote_id(state_path, client, *, remote_id)` verifies an independently discovered dataset/job ID via GET after an ambiguous create. For jobs it requires the exact dataset, model name and base model. It cannot acknowledge an ambiguous entry upload.

State contains explicit endpoint/entity/project identity; absolute source path and exact bytes SHA-256; canonical mapped upload SHA-256; review/heldout lineage; dataset/relabel/fine-tune/task IDs; upload revision/results; config; remote status/progress/failure type; dataset/model artifacts; pending operation and sanitized last error. `training_model_identity` is populated only from actual returned `model_id`, `training_model_id` or `run_id` fields, with `source='finetune_response'`. The project/task ID is never asserted to be the training model ID. The documented response does not guarantee underlying training identity, so RL continuation must independently verify it when absent.

## Safety and known limitations

Before each remote write, the workflow durably persists an intent under a process lock. Transport failures, HTTP 5xx, malformed successful acknowledgements, and process interruption leave that intent unresolved. Resuming blocks another mutation until the remote state is reconciled. Definite HTTP rejection clears the intent and retains actionable status. The client does not automatically retry writes.

A failed fine-tune remains attached to its original ID and is only polled. `training_failure_type=RemoteTrainingFailed` is retained; the workflow will not silently recreate the job. Creating a genuinely new run requires a separate explicit workflow/state path.

The pilot implementation supports one atomic upload of at most 5,000 entries. An ambiguous upload cannot automatically be retried or marked complete: inspect the remote revision and entries, then add a carefully verified reconciliation path if needed. Source edits after initialization fail closed, including whitespace-only edits. Keep source JSONL immutable and available while polling.

Teacher relabeling is deliberately not used because a verified tool-call relabel contract was not established. The supported fallback uploads actual locally captured Sol trajectories. `relabel_run_id` stays null. No custom GPU SFT fallback is implemented.

The service's effective minimum training count, project-specific Gemma entitlement, final deployment artifact, and underlying Serverless Training identity remain live-service questions. This task's mocked tests do not claim actual job acceptance or training success.

## Verification evidence

Initial focused run failed at collection because the new `bufogen.forge` module did not exist. After implementation, focused checks passed. Additional tests for documented nested API errors and `training_failure_type` were observed failing, then corrected and rerun successfully.

- `.venv/bin/pytest -q tests/test_forge.py tests/test_learning.py`: **23 passed**.
- `.venv/bin/ruff check src/bufogen/forge.py src/bufogen/forge_loop.py src/bufogen/curate.py tests/test_forge.py tests/test_learning.py`: **All checks passed**.
- Full suite before the final two schema-specific tests: **202 passed**, 258 existing Weave/Pydantic deprecation warnings. Final full-suite result is recorded below when complete.

Tests cover sanitized auth/nested errors, disabled-model preflight, explicit project matching, malformed/unreviewed/held-out rows, real tools and metadata, validation opt-in, family leakage, local failure before mutation, dry-run before commit, dry-run rejection with retained dataset ID, source hash changes, resume without duplicate jobs/imports, transport/503 ambiguity, verified ID reconciliation, file lock contention, missing deployed artifacts, and terminal remote failure retention.

Final full-suite verification: `.venv/bin/pytest -q` → **204 passed, 258 existing Weave/Pydantic deprecation warnings in 2.93s**.

Independent-review correction: nested custom metadata previously could overwrite validated source identity/partition fields. `reviewed_entries` now merges every validated local source field after custom metadata, preserving unrelated annotations while retaining authoritative source model, generation/group IDs, prompt/version, partition/family, and review. The regression was observed failing before correction; afterward **24 focused tests passed** and Ruff passed. Root also verified the live read-only client preflight successfully; evidence is `runs/forge-loop/client-preflight.json`.
