# SDD ledger — plan: docs/superpowers/plans/2026-10-15-forge-ai-loop.md

2026-10-15: User authorized full Forge AI Loop implementation using subagents. Plan created; product preflight and code audit running.

Workspace: existing non-Git directory; reviews use file snapshots and scoped diffs. Preserve this ledger because no Git history exists.

Task 1: complete — authentic tool trajectory persistence/export; 42 focused tests and Ruff pass, independent spec/quality review clean. Reports task1-report.md and task1-review.md.
Task 2: complete — 74 focused/regression tests, Ruff and independent spec/quality review pass; actual offline native Weave cached-score replay verified by reviewer. Reports task2-report.md/task2-review.md.
Task 3: implementation starting — managed Forge integration and state.
Task 4: pending — RL reward/capture/checkpoint contract.
Task 5: pending — CLI, ARIA handoff, showcase and documentation.
Task 6: pending — live execution, review queue, training where prerequisites permit, final verification.

2026-10-15 preflight: Forge authentication works, Bufo alias available for existing W&B project import, training catalog includes exact Gemma. Existing annotation queue: 19 items / 1 completed. No labels fabricated.

2026-10-15 user steering: assistant-reviewed pilot targets explicitly authorized. Root will inspect actual candidate images and record assistant provenance; user ARIA curation remains distinct.

Task 1: implementation in progress, worker /root/trajectory_export. Product preflight /root/forge_preflight; code audit /root/training_audit complete.

Preflight complete: `docs/forge-preflight.md` contains live eligibility and exact API contracts. ART 0.5.20 requires new capture API; search Responses calls require no_capture. Same-training-model SFT continuation must verify storage identity and checkpoint, not assume artifact serving proves initialization.

Human annotation recovered read-only: call `01a1212c-2869-7f4d-8aa8-4e6101efa0b0`, bufo_present=false, similarity=0, prompt_alignment=0, reason="abstract not similar". Saved `existing-completed-feedback.json`; source image manifest `runs/bufo-identity-probe/manifest.jsonl`. This is one negative calibration example, not sufficient calibration by itself.

Task 1 review: found acceptance laundering through cumulative prior conversation turns. Decision: initial SFT export may explicitly refuse external conversation history while preserving it for chat, until every target-bearing ancestor has review/split lineage. Intra-request repair trajectories remain supported. Cost: initial SFT pilot teaches generation/repair, not multi-turn edit histories. Task 5 must expose reviewed-existing-manifest export; old curate CLI's generate-then-review flow is insufficient.

Task 1 live smoke started via /private/tmp/bufo_trajectory_smoke.py: Sol presenter scene with shared web disabled, authentic trajectory structural export before any visual acceptance. Evidence target runs/forge-loop/trajectory-smoke/verification.json. No training submission yet.

Task 1 live smoke PASS: generation 0483e13cd55c450c8624f61b18fc09bf, real Sol + cwsandbox, valid PNG, 5 exported messages and render_svg/search_web schemas, 63.905s. Root visually inspected and recorded explicit assistant acceptance in trajectory-smoke/review-decisions.jsonl (connected character, held mic/pointer, race car and simulator; small-text limitation noted). This is development training data, not heldout evidence.

Forge project created via documented import API: alias bufo-gen / project dd5ddbe2-b31c-43e1-ab39-b5488c78f3d8, v1 route to wandb-inference/google/gemma-4-26B-A4B-it applied. New alias only; existing app configuration unchanged. Evidence runs/forge-loop/project-setup.json. No dataset or job created yet.

Forge proxy connectivity PASS: actual POST to proxy.training.coreweave.com/v1/chat/completions with model=bufo-gen and documented metadata returned HTTP 200, resolved model google/gemma-4-26B-A4B-it and READY. Evidence runs/forge-loop/proxy-preflight.json. This is connectivity evidence, not a scene-quality evaluation.

Scene scorer live smoke: Sol presenter checked 5/5 explicit constraints, identity .85/anatomy .9. Original bad Gemma F1 image identity .75 but anatomy .4, composition .6, readability .5, constraint coverage 1/3; correctly flagged quality failure despite recognizable Bufo. Evidence runs/forge-loop/scorer-smoke/{positive,negative}-result.json. These two examples are uncalibrated development checks, not a statistical comparison.

Task 2 initial review found partial-checkpoint crash safety, prior-image misattribution on interrupted edit, existing queue schema reuse, and silent native-Weave eval coverage loss. Worker fixed with 74 focused/regression tests; independent recheck pending. Suite now 12 train +4 separate validation +100 heldout prompts.

Task 6 live pilot launched after Task2 approval: /private/tmp/bufo_forge_pilot.py, runs/forge-loop/pilot, 12 train +4 validation scenarios on Gemma and Sol, concurrency4. Full logs runs/forge-loop/pilot-live.log. It runs generation, scene scoring/native Weave Evals, then separate review queue. Heldout100 not yet run; freeze relevant agent/runtime changes before final A/B.

Pilot startup correction: shell-redirection invocation failed DNS before creating benchmark state (saved pilot-startup-failed.log). Reissued as standalone approved .venv/bin/python -c runpy command; Weave authenticated and benchmark.json created. Active exec session99335; durable evidence in pilot directory. Do not launch duplicate pilot requests while that process runs.

Task3 complete: independent spec and quality PASS after fixing nested metadata overwriting authoritative lineage. 24 focused tests/Ruff pass. Root actual ForgeClient preflight succeeds; evidence client-preflight.json. Task4 worker active; no venv changes until live pilot ends. W&B MCP Agents list retried and still HTTP403; direct local SDK/Weave publication works.

Live pilot finished exit0:32/32 valid renders; Qwen provisional pass Gemma8/16,Sol14/16 (one judgeerror). NativeWeaveEvals and reviewqueue01a1403d-e1af-7922-aa5f-a553c849a728 published. Assistantteacherreview16 complete:train9accept/3reject,validation4accept; root smokeadds1train. Frozenreviewed-examples14 (train10,val4), Forge dataset058d1158-3c6a-497b-84f4-a786f826b89f dryrun+upload success. ManagedSFTjob100202f8-7034-4bc2-bd2f-2375dd43fbba accepted and training; no modelartifactyet. Scorer audit2falsepasses+1error; cannotusecurrentQwenassoleRLreward. Root probingSoljudge on2positive+2negativeactualimages; no production scorer change yet. Task4maynowinstallART.

ManagedSFT DEPLOYED actualimmutableadapter wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v1; rawjob finetune-latest.json, trainingrun bufo-gemma-sol-pilot-20261015. 10train4validation. Preservev1 beforeRL. Soljudgeprobe17actualimages:14/17agreement,0FP,3FN(ambiguoussitting/title and smallBufoframe detail), noerrors; diagnosticprobeonly, productionscorer integrationpending.

Task4 complete independent specqualityPASS afterfixing trainingmetrics split. 224fulltests+30focused pass, realARTcapture tested. RuntimeART0.5.20 requiredOpenAI2.54/agents0.20; actualtunedagent ranandrenderedvalidPNG undernewruntime. Visualreject pointermiss recordedtuned-smoke/review-decisions.jsonl. FullpilotassistantQA now32images:Gemma1/16,Sol15/16 (developmentpilotonly). Scorerv2 spatialgroundingprobe running; Task4bworkerimplementingconfigurableResponsesjudge. RLdatasetpublish started, noRLtraining yet.

Task4b complete independent specqualityPASS,66focusedtests. Exactv2rubric hash ee0b672fd60f526cdbb293013cecd1966f50d878f8c8a263c436a12a8b00d0b6. LocalSCENE_JUDGE_BACKEND=closed_quality_bar selected; compiledcalibration32images16positive16negative,28agreements0FP0judgeerrors4FN,populationfalse. ActualrawResponsesreplayedwithsourcehashes in calibration/. Calibrationartifactpublishing. RLtaskartifact wandb-smle/bufo-gen/bufo-rl-environments-20261015:v0 published. Rootlaunching2stepSFTcontinuation pilot via /private/tmp/bufo_launch_rl_pilot.py. Task5implementationactive/loop_showcase; wait runtimefreeze beforefullheldout. LocalOPEN_TUNED_MODEL points immutableSFTv1.

RealRLcontinuation VERIFIED registeredsameSFTstorage: model_id7afe082e-fa39-4377-b28b-a093af9b1c0b, run_idbufo-gemma-sol-pilot-20261015, start_step1. FirstactualrolloutPNGvalid, capturedmodelstep1artifact andscenechecklistcoverage.4. Activeexecsession45097; donotduplicate/relaunch. State runs/bufo-gemma-sft-rl-pilot-20261015/rl-training.json. Calibrationartifact wandb-smle/bufo-gen/bufo-scene-v2-pilot-calibration:v0 publishedwith32PNG/rawresponses/frozenrubric.

Task5 implemented, runtime subset independently approved; full245 Python and3 Node checks pass. Final reviewer found CLI frozen-split validation, cross-root state default, and report manifest-coverage blockers; worker fixes pending. Full heldout100x3 run active with immutable SFTv1, frozen runtime, native evals to follow.
RL first attempt failed APITimeoutError before any optimizer submission after3 completed scored rollouts. Read-only remote checkpoints confirm latest step1; evidence rl-interruption-check.json. Worker adding bounded transient inference retry and authentic capture audit; no backend.train retry or ambiguous resume. Failed attempt remains preserved.

Task6 recovery independently approved31focusedtests; fresh RL attempt bufo-gemma-sft-rl-pilot-20261015-retry1 launched after read-only remote checkpoint verification. Original failedattempt preserved. Proxy fullagent e2e passed actualGemma routing+PNG+Weave trace, evidence proxy-agent-smoke/verification.json. Fresh-process UI exposes deployedSFT; existing server requires restart to refresh .env.

Full heldout generation complete300scenarios; scoring/nativeWeavepublication running. Blinded15imageheldout audit:base0/5,SFT0/5,Sol3/5, no training eligibility. RL retry generated4scoredrealcaptures then failed missingpyarrow at model.log beforeoptimizer. Additive dependency/preflight and reviewed saved-group recovery in progress; no capture regeneration or optimizer retry. Firstattempt and retry failures retained.

Full100promptheldout scored300attempts:base1/100,SFT1/100,Sol71/100; generationerrors5/3/0, zerojudgeerrors. Blindedheldout15imagesagree15/15withfrozenjudge; smallsupplementalsample only. W&B300imagecomparison https://wandb.ai/wandb-smle/bufo-gen/runs/rqbi08g1 published; nativefeedback/evalpublicationongoing. Additivepyarrow26.0.0 only,260tests; savedgrouprecoveryapproved33tests+actualParquetroundtrip and launchedsingleoptimizerupdate. State runs/bufo-gemma-sft-rl-recovered-20261015/rl-training.json. Remote17:13UTCcheckstillstep1. AgentLensproject404conversations,390sentiments,280failureseverity; clusteringunverified.

Full heldout publication completed: 300 feedback attachments and three native Weave evaluations, no publication errors. W&B comparison has 300 attempt rows and 292 actual PNGs; eight failed generations remain explicit. SFT pilot strict pass rate did not improve over baseline.

Recovered RL optimizer submission failed remotely: `ValueError: Could not locate a history message in the rendered history`. Checkpoints still [1, 0], zero completed updates and no RL artifact. The failed attempt is preserved; no optimizer retry made. Local read-only reproduction with untouched captures and official Gemma tokenizer reproduced the exact error for two trajectories and a related sampled-boundary error for two. Diagnosis packaging and exact-token capture alternative assessment in progress.

Future job-ID receipts and final showcase evidence links independently reviewed and approved. Root fresh verification: 265 Python tests, three Node chat tests, Ruff all pass. Showcase regenerated with completed heldout/native evaluations, Agent Lens scope, and explicit selected RL failure. Live results documented in `docs/forge-loop-work/live-results.md`. RL training/evaluation remains incomplete.

Token-capture follow-up: the provider returned genuine output IDs when logprobs were requested. However, a complete two-exchange tool/reply probe with ART default token-ID flags and preserve_thinking still failed local alignment; prompt token IDs were absent. This rules out declaring a client-flags-only fix. Original captures/rewards remain untouched, no further optimizer job submitted. Durable evidence `runs/forge-loop/rl-tokenization-diagnosis/`. Final independent evidence review approves an honest handoff, explicitly leaving successful RL and its heldout evaluation incomplete.

Diagnosis finalized: `docs/forge-loop-work/gemma-rl-tokenization-diagnosis.md` includes primary sources, version/file hashes, two verified offline reproduction commands and the exact remaining tokenization contract. No runtime changes from diagnosis; no running training or evaluation processes remain. Final showcase and live-results reflect deployed SFT plus failed RL, without a parity or complete-RL claim.
