# Task 4 independent review

## Spec review: PASS

Reviewed the task brief, implementation report, snapshot diff, production source, installed ART 0.5.20 implementation, tests, and the actual deployed Forge job / immutable SFT artifact records. The implementation meets the scoped reward, student capture, train-only task data, and continuation contracts.

- Reward revalidates the exact generation, requirements and scorer configuration, recomputes checklist coverage, rejects malformed/nonfinite scores, and gives invalid renders or missing Bufo zero reward. It does not retain the legacy CLIP/VLM/length reward.
- Calibration rebuilds observations instead of trusting an editable readiness flag; joins review provenance and image hashes; keeps judge failures and disagreements; requires positive and negative evidence. It correctly labels this limited pilot agreement rather than population calibration.
- Environment tasks preserve requirements and family/split identity. Validation/heldout tasks and edits are excluded from conversion; the loader rejects these if they appear in a purported training artifact.
- Current ART context-manager capture is used. Search is disabled, the judge executes after capture exits, and captured request model/protocol/tool associations are checked. The code correctly acknowledges that ART exchange records do not independently prove the HTTP endpoint; that is bound at client configuration.
- SFT continuation requires an actual deployed Forge job, immutable matching artifact ID/digest, original base and logged-by run. Registration uses that storage identity; a positive service checkpoint and identical step artifact are required before rollout. A recreated empty model fails instead of silently claiming continuation. No experimental adapter fork is misrepresented as training initialization.
- The driver explicitly does not promise automatic crash resumption. Exceptions mark local state failed and do not promote an artifact. This review did not perform remote mutations or establish service-side RL availability.

## Quality review: PASS (finding fixed and rechecked)

**Resolved P2 — Training metrics had used the validation split.** The implementation now explicitly passes `split="train"` to the metrics `model.log` call; trajectory logging already passed this split. The training-boundary regression asserts the metrics split and completed training step. Independent recheck of the actual production call and `.venv/bin/pytest -q tests/test_rl_dataset.py -k serverless_rollouts` passed (**1 passed**, 5 deselected). The earlier default-to-validation reporting defect is resolved.

No other blocking correctness finding identified in the reviewed scope. In particular, local fixtures do not substitute for the outstanding real continuation/checkpoint probe or real RL training verification; those remain root integration work.

## Independent verification

`.venv/bin/pytest -q tests/test_rl_contract.py tests/test_rl_dataset.py tests/test_learning.py` → **30 passed**, 203 existing Weave/Pydantic deprecation warnings. Checked real installed ART `TrainableModel.register`, `ServerlessBackend.register/train`, `Model.log`, checkpoint lookup and inference naming signatures against the production calls. The test suite includes a real installed ART + OpenAI/httpx transport capture test as well as mocked training orchestration.

The source tree and dependencies were not changed by this review. Only this report was written.
