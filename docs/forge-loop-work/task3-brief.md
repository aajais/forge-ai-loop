## Global Constraints

- Teacher: configured Sol reference; student: `google/gemma-4-26B-A4B-it`. Never silently replace Gemma with another training family.
- Use Forge Model Distillation as the primary SFT path, not the older custom GPU sandbox script.
- Preserve rendered PNG outputs even when image quality fails. Never manufacture success, labels, calibrated metrics or trained artifacts.
- Keep render tool calls before the final assistant text and PNG; preserve conversation, model selection, image input and concurrent chats.
- Match prompt version, character assets, context policy and tool budgets across comparison arms. Partition prompt families before generating teacher targets.
- User has authorized paid task execution and no dollar cap. Bound pilots for evidence and scale only after data validation; no production routing changes are implied.
- User performs ARIA curation and has additionally authorized assistant-reviewed pilot targets. Record assistant reviews distinctly with image evidence, reviewer identity and timestamp; do not impersonate human reviews.
- Do not print secrets, commit credentials or contact colleagues. Document platform enablement requirements without sending messages.
- This directory has no Git metadata. Work in place, retain file snapshots/review reports in `docs/forge-loop-work`, and do not initialize or pretend to commit a repository.

## Review Focus

1. Repair trajectories and follow-ups must retain real tool IDs/results, without invented calls or expanded canonical SVG definitions.
2. Missing/negative human review and held-out families must never become accepted SFT targets.
3. Failed generations/judges must remain in denominators, and comparison must reject incompatible run settings.
4. Resuming Forge operations must not create duplicate training jobs or promote undeployed models.
5. RL must capture only student actions, not search/judge models, and must not silently restart from base when claiming SFT continuation.

## Task 3: Forge client and resumable managed training

**Files:** new `src/bufogen/forge.py`, new `src/bufogen/forge_loop.py`, new `tests/test_forge.py`, `.env.example` and CLI integration later in Task 5.

**Interfaces:** Follow documented API schemas from `docs/forge-preflight.md`. `ForgeClient` accepts explicit entity/project alias, base URL and injected HTTP transport for tests. Workflow state persists project/dataset/relabel/finetune IDs, exact source dataset hash, model, status and deployed artifact. No speculative API endpoints.

- [ ] Test sanitized authorization failures, disabled model, malformed dataset, transport interruption and resume without duplicate mutations using `httpx.MockTransport`.
  ```python
  def test_resume_reuses_job(saved_state, fake_client):
      # Existing job ID must be polled, never recreated.
      resume_training(saved_state, fake_client)
      assert fake_client.create_calls == 0
  ```
- [ ] Implement read-only project/model eligibility preflight, explicit project setup, reviewed trajectory import and teacher relabel where its tool-call contract is supported. Otherwise import actual locally collected Sol trajectories through the documented dataset API.
- [ ] Submit/poll managed SFT, save remote IDs before subsequent work, distinguish transport uncertainty from definite rejection, and return only real deployed model identifiers.
- [ ] Preserve Forge/Weave/dataset/model lineage in state and emit actionable unsupported/permission status. Never substitute custom sandbox SFT for a failed Forge request.
- [ ] Run focused tests; independent review.


## Verified product contract and existing resources

Read docs/forge-preflight.md for exact official schemas and live eligibility. Root already imported project successfully: runs/forge-loop/project-setup.json, alias bufo-gen, project ID dd5ddbe2-b31c-43e1-ab39-b5488c78f3d8, v1 baseline Gemma route. Reuse it; no duplicate project/route changes. Actual live Sol trajectory smoke at runs/forge-loop/trajectory-smoke with explicit assistant review. New library must map local richer export fields to documented Forge upload fields, use tools and all assistant turns, preserve source review metadata and immutable hashes. No API key/provider creation for Sol necessary: upload actual collected Sol trajectories. Built-in wandb-inference provider ready.

User authorizes real paid pilot after reviewed targets; root handles initial job submission to avoid duplicates. You implement/test only; no remote mutations. Error messages must retain actionable status without dumping headers/tokens. State transitions atomic, unknown post timeout fail closed before retry (reconcile by ID/list or require explicit ID). Get exact real deployed artifact, do not invent URI from job name. Keep dataset/test separation and validation rows (val) distinct; heldout test never uploads.

Validation data: Task2 adding four single-turn validation prompts in disjoint families (separate from 12 train and100 heldout). Forge needs split train|val. You may extend curate.select_examples with explicit allowed_partitions train/validation (default train preservesexisting callers) and preserve actual partition in returnedexamples; reject heldout/test regardless. Do not relabel validation as train to bypassgates. Add focusedtests ifyouextend it. Actual reviewed validation examples go to Forge val; heldouttestneveruploaded.
