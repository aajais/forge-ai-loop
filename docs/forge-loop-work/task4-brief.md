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

## Task 4: Safe RL reward and SFT continuation contract

**Files:** `src/bufogen/train_rl.py`, `src/bufogen/train_common.py`, new `tests/test_rl_contract.py`, relevant `tests/test_learning.py`.

**Interfaces:** Reward consumes current visual scorer outputs, with explicit rubric/calibration metadata. Training config records `base_model` and optional initial checkpoint separately. Only use documented ART continuation support; unsupported SFT continuation raises an actionable error before creating a job.

- [ ] Test missing scorer output, NaN/out-of-range metrics, absent Bufo, generic-label scene, nonstudent captured choices and unsupported initial checkpoints.
- [ ] Reward identity and prompt fulfillment jointly with composition/readability, retaining invalid renders as zero reward. No reward from unchecked cached judge scores.
- [ ] Isolate external search/judge calls from ART student capture; validate captured provider/model and tool trajectory before training.
- [ ] Add verified checkpoint initialization if supported, otherwise keep base-RL and SFT arms explicitly distinct and report continuation blocked. Do not present a new base run as SFT+RL.
- [ ] Run focused tests; independent review.


Verified ART preflight in docs/forge-preflight.md. Pin and install released openpipe-art==0.5.20 optional rl dependency and update lock intentionally (own pyproject.toml/uv.lock allowed). New art.Trajectory() context captures exchanges; no oldAutoTrajectoryContext import. For root pilot controlled RL, construct BufoAgent(web_search_enabled=False) so nestedSolsearchneverruns; judgeoutsidecapture; inspectallcaptured exchange.model IDs tovalidateonlystudent. This avoids changingbenchmarkagent/web sourcehash. Ifyouenableexternalsearch later use no_capture around actual research; do not silentlycaptureteacher.

Same-LoRA SFTcontinuation: validate exact existingserverlesstrainingidentity matches deployedForgeartifact (resolve mutablealias to immutablevN/digest first), originalGemma base, currentstep>0 and matchinginitialcheckpoint. Never auto-create a new base model when claimingcontinuation. Experimental fork upload alone is not evidence of traininginit. Expose explicit inputs and reportunsupported rather than guess. Root willreadrealForgejobstatuswhenavailable. Beforeupdatingvenv, verify rootpilot is no longer running; contactroot ifunclear (long-livedimports mustnotbemixed).

Calibration and RL task data: use SceneQuality output status/requirements and real assistant-review observations in runs/forge-loop/pilot/assistant-reviews.jsonl, plus prior humannegative evidence. Provide a small calibration-report builder that joins exactgenerationIDs/scorerconfig, computes observed agreement/errors and retains samplecounts/reviewerprovenance; never flip rubric status to calibrated just because a flag was passed. For smoke RL, require at least both reviewed positive and negative examples and reportlimitedpilotcalibration explicitly; do not claim populationaccuracy. Reward invalidrenders/missingBufo/errors as0 and combine identity+constraintcoverage+anatomy/composition/readability; nolegacy CLIP/VLM orSVG lengthpenalty.

RL datasets are environmenttasks (prompt_id,prompt,requirements,family,split=train), not teacher targetmessages. Add helper or trainer support for a real versionedartifact of benchmarktrain prompts/requirements; preserve originalcharacterreference fromsharedscorer; neverholdout/validation tasks forRL. Old128reference dataset withoutrequirements must failactionably orbeexplicitlyconverted, not silentlyuseanotherrubric. Root handlespublishingdataandlivejobsubmission. Read Task2report for SceneQualityconfiguration/scores.
