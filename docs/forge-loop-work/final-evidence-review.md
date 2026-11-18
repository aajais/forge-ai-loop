# Final Forge loop evidence and spec audit

Reviewer: `codex/final_review`, 2026-10-15. Read-only audit of the approved plan, implementation ledger, task/recovery reviews, `live-results.md`, saved Forge/RL state, heldout manifests/scores/publication receipts, assistant-review evidence, and current showcase. Only this review file was written. No live API calls, paid calls, training submissions, runtime edits or broad code re-review were performed.

## Verdict

**PASS for an honest handoff of the implemented and measured loop, with an explicit unsuccessful RL stage.** No new correctness or evidence-integrity blocker was found in the handoff artifacts. This is **not** approval to claim that every requested training stage succeeded, that Gemma matches Sol, or that an RL model was produced.

The implementation and managed SFT path are delivered. Successful RL optimization and evaluation of an RL checkpoint remain unfinished. User ARIA curation and complete Agent Lens clustering verification also remain unverified, as the handoff already states.

## Evidence checked

| Claim | Saved evidence and audit result |
| --- | --- |
| Controlled heldout comparison completed | `heldout-baselines/benchmark.json`, manifest, scored rows and summary validate together through the reviewed report builder: 100 prompts, three arms, 300 attempts. |
| Exact results | Baseline: 95 valid images, 1 provisional strict pass; Sol: 100 images, 71 passes; SFT: 97 images, 1 pass. Denominator is 100 for each arm. Eight generation failures remain included; no judge errors are hidden. |
| Native publication | The 300 feedback receipt IDs exactly match the 300 generation IDs. Each of the three Evaluation receipts reports 100 scored rows, with no publication errors. |
| Rendered comparison artifact | `heldout-baseline-visualization.json` records 300 rows, 292 images and the same frozen comparison fingerprint, linked to run `rqbi08g1`. |
| Real managed SFT | `forge-state.json` is `deployed`, job `100202f8-7034-4bc2-bd2f-2375dd43fbba`, exact Gemma base, immutable adapter `wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v1`. |
| Reviewed data provenance | The actual workflow source contains 14 examples: state records 10 train and four validation. All reviews identify the assistant. Direct checks found no heldout prompt-ID or family overlap. |
| Limited independent image audit | Saved heldout spot-check summary has five images per arm, visual passes 0 baseline / 0 SFT / 3 Sol; its separate agreement receipt has 15/15 agreement. It explicitly excludes training use and does not claim representative human calibration. |
| Observe limitations | Agent Lens receipt records 404 project-wide conversation groups, 390 with intent sentiment, 280 with failure severity. These are project-wide snapshots, not counts unique to this benchmark or proof of completed clustering. |
| Proxy trace behavior | The trace verification preserves actual PNG/media and application ordering. Documentation explicitly discloses additional gateway request turns rather than claiming one merged turn. |
| Showcase consistency | Fresh read-only report construction accepts the pilot at 32/32 and heldout at 300/300, both scored. SFT is deployed; the selected RL state is failed. No fourth model arm or replacement RL artifact is invented. |

## RL status and remaining dependency

**There is no actual new RL checkpoint in the inspected evidence.** The recovered submission state is failed. `rl-recovery-failure.json` records zero completed updates, `new_rl_checkpoint: null`, no automatic retry, and the remote error:

```text
ValueError: Could not locate a history message in the rendered history
```

The saved post-failure remote checkpoint receipt contains only steps **1 and 0**. Step 1 is the verified existing SFT checkpoint, not a completed RL update. All three preserved local RL attempt states are failed. The genuine scored trajectories, their source hashes, and rewards remain available; the failure has not been converted into synthetic success or silently restarted from base.

The local diagnosis receipt inspected at `/private/tmp/bufo-gemma-tokenization-diagnosis.json` reproduces the exact remote error for two untouched captures and a related sampled-boundary error for the other two. Its content-preserving template experiment succeeds for two captures and still fails for two. That supports a specific Gemma/ART history-tokenization incompatibility; it does **not** demonstrate a working general repair.

The remaining technical step is to establish a faithful, supported training representation for those actual sampled student actions, including assessment of exact token-ID/logprob capture where available, before any further optimizer submission. The precise remedy and whether it requires a service/library change are still under investigation. Do not describe this as definitively unfixable locally or exclusively upstream until that investigation concludes. A future successful update must produce a real immutable checkpoint, verify serving, and undergo the same controlled heldout evaluation.

The plan's successful-RL/evaluated-RL item must remain incomplete or explicitly blocked by this recorded failure. The user-facing handoff should lead with the measured SFT result and unsuccessful RL status, rather than an unqualified statement that all requested work succeeded.

## Claims and final housekeeping

- `live-results.md` accurately distinguishes rendering, deployment and quality improvement. It reports that this small SFT pilot did not close the Sol gap and preserves the shared Sol teacher/judge and limited calibration caveats.
- No claim of completed human ARIA approval, population calibration, browser visual testing, RL improvement or Sol parity is supported or needed for handoff.
- Root reports a fresh final verification of **265 Python tests, three Node tests and Ruff**. At audit time, `live-results.md` still says 263 Python tests; update that historical count when recording the final result. This is documentation synchronization, not a blocking defect.
- Package the local tokenization diagnosis into durable project evidence and link it from the final failure handoff. The current `/private/tmp` receipt is a valid inspected observation but should not be the sole long-term support for the diagnosis.

Earlier task approvals remain the code-review basis; this audit independently verifies saved evidence consistency and claim boundaries. It does not claim fresh remote status beyond the timestamps of those receipts.
