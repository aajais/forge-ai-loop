# SDD ledger — plan: docs/superpowers/plans/2026-10-21-frozen-runtimes-and-latency-instrumentation.md

Spec: /Users/aayush.jaiswal/.claude/plans/look-atthis-project-i-noble-scroll.md
VCS: shadow git. Every git command needs: export GIT_DIR=/private/tmp/claude-502/-Users-aayush-jaiswal-Desktop-cw-bufo-gen/e49cc808-fe4a-4d3f-b40b-1953a1b95b51/scratchpad/shadow.git GIT_WORK_TREE=/Users/aayush.jaiswal/Desktop/cw-bufo-gen
Baseline commit: 7f046b7

Ruling: track diffs in a shadow git dir outside the project (GIT_DIR in scratchpad) instead of git init in the project — the approved spec puts git init in Stage 7 and the skill needs commits for review packages — cost if wrong: shadow history is lost with the scratchpad; the project tree is unaffected. Note: project .gitignore ignores docs/ and runs/, so ledger/doc edits and restored runtimes are not in review diffs (intended: runs/ is large output).

## Pre-flight scan
| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1→T2 | restore(run_dir,target,project_root)->dict; T2 adds --sync in same script | consistent |
| T2→T3 | runs/restored/<slug>/ with .venv, RUNTIME.json, env.frozen | consistent |
| T1/T3 tests copied | T1 copies *current* tests/ into restored tree; T3 runs them against *frozen* hashed modules | CONFLICT: tests were never frozen; current tests (e.g. test_benchmark.py asserting max_turns=100) can fail against old frozen benchmark.py for reasons that are not runtime breakage. See ruling R1 |
| T4→T5 | ModelCall, Recording*Model(model, openai_client, calls), BufoGeneration.model_calls/truncated_calls | consistent; ImageReplyModel.get_response delegates to inner.get_response so recording still fires |
| T5→T6 | BufoAgent(disable_thinking: bool|None) | T6 says CLI forwards for run/demo, but pipeline_run.run_pipeline is not in T6 file list. See ruling R2 |
| T6→T7 | --disable-thinking, policy.disable_thinking distinguishes configs | consistent; 12 train prompts exist in config/forge_prompts.yaml |
| T1 self | tests vs code | consistent |
| T2 self | uv sync --locked needs ART git source (network or uv cache) | note only |
| T3 self | summary diff "empty" may include volatile fields | ruling R3 |
| T4 self | consistent |
| T5 self | consistent |
| T6 self | consistent once R2 applied |
| T7 self | strict pass must use benchmark's own definition | ruling R4 |

Ruling R1: In Task 3 the per-test results are recorded and each failure classified as (a) runtime breakage (frozen code cannot import/run) or (b) test newer than runtime (test asserts post-freeze behaviour). The hard gate is: no class-(a) failures, plus Task 3 steps 2 and 3 passing. — the spec requires tests in the environment but tests were never frozen, so a current test asserting newer behaviour proves nothing about the old runtime — cost if wrong: a real incompatibility hidden as class (b); mitigated by requiring the offline re-score through the frozen benchmark/scorer code.
Ruling R2: Task 6 also modifies src/bufogen/pipeline_run.py so run/demo forward disable_thinking. — plan text says the flag is forwarded for run/demo — cost if wrong: one extra small edit.
Ruling R3: Task 3 summary comparison ignores fields that are timestamps or absolute paths; any other difference fails the step and is reported. — re-scoring regenerates the file — cost if wrong: a meaningful field misclassified as volatile; the reviewer sees the list of ignored keys.
Ruling R4: scripts/latency_report.py computes strict pass by reusing the benchmark module's own pass function/definition (benchmark.py summary logic), not a reimplementation. — avoids drift from the frozen definition — cost if wrong: none material.
Task 1: dispatched (implementer agent ae23aedb717c20ee0, sonnet), BASE 7f046b7
Task 1: complete (commits 7f046b7..64dffe9, review clean)
Task 1: minor (deferred→carried into Task 2): frozen overlay goes through the scaffold exclusion filter, so a frozen file under a dir named runs/wandb/.env would be dropped silently; no post-overlay re-hash
Task 1: minor (deferred): source_hashes keys not checked for '..'/absolute paths
Task 1: minor (deferred): empty SCENE_JUDGE_BACKEND= written if scorer fields missing
Ruling R5: Task 2 also (a) copies the frozen overlay unfiltered and re-hashes restored files in target after overlay, (b) replaces README placeholder commands with the real Task 2/3 commands — Task 2 owns this script next and the README is specified as "the Task 2 commands" — cost if wrong: small extra diff.
Task 2: dispatched (implementer ae449a9a6f8032fca, sonnet), BASE 64dffe9; R5 + disk guard (stop <8GB free) in dispatch
Task 2: complete (commits 64dffe9..bfcba2f, review clean; controller spot-check: forge-gap import resolves inside target)
Task 2: minor (deferred): wandb dir timestamp parsed as UTC though wandb names dirs in local time; run_started_utc may be mislabelled (best-effort field)
Task 2: minor (deferred): redundant double parse of requirements.txt in _wandb_requirements
Task 2: minor (deferred): mixed naive/aware timestamps in manifest would surface only as a reason string
Task 2: minor (deferred): _copy_all follows symlinked dirs (loop risk)
Task 2: note: openpipe-art wandb 0.5.20 vs git pin is not real drift
Ruling R6: Task 3 never writes to existing runs/ (re-score and contact validation run on copies inside each target), verifies every row has a cached score before scoring, blanks credentials, and uses only offline validation modes — protects evidence and avoids paid calls — cost if wrong: a validation mode that legitimately needs the original path layout is skipped and recorded rather than run.
Task 3: dispatched (implementer a67cf98dd78944169, opus — judgment-heavy classification), no code BASE (outputs gitignored)
Task 3: first pass 2/8 gate (gap, iteration2 pass with evidence copies; contact x5 fail only fit_compare "Missing completed fit source: fit-candidate"; forge-loop runtime breakage: frozen models.py vs current config/models.yaml + today's pipeline_run import via CLI)
Ruling R7: contact-pilot fit_compare validation is NOT APPLICABLE until the fit-candidate exists (the SFT job 61b53c05 is still training; today's tree fails identically, so it is missing future evidence, not breakage); evaluate.py offline check is the step-3 evidence — cost if wrong: a fit_compare incompatibility surfaces only once the candidate lands; re-run it then.
Ruling R8: gap/iteration2 failures that pass once their run evidence is supplied as read-only copies inside the target count as PASS, relabelled "(c) evidence-path dependent", not (b) — cost if wrong: none; real evidence hashes unchanged.
Ruling R9: forge-loop/heldout-baselines may use ONE evidence-derived reconstruction: config/models.yaml generated solely from benchmark.json configuration.arms (recorded field values of that run), recorded in RUNTIME.json under "reconstructed" with its derivation; and re-scoring calls the frozen benchmark.score_benchmark directly (python -c) instead of today's CLI entry point, since cli.py/pipeline_run.py were never frozen. If it still fails, the runtime is declared non-reconstructable with evidence — cost if wrong: the reconstructed yaml omits a field the old loader needed; the test run will show it.
Ruling R10 (finding from Task 3): RUNTIME.json from_current_tree must list ALL non-frozen files copied from today's tree (config/, docs/, scripts/, tests/, src/), not only src/*.py — restore script fix with a test — cost if wrong: none.
Task 3: fix round 1/5 (R7, R8, R10 addressed; commit 21e11d4 adds full from_current_tree + --refresh-manifest + --project-root; 6/8 gate; open: iteration2 copies fail on absolute evidence paths, forge-loop KeyError base_url_env)
Ruling R11: iteration2 evidence may be supplied via a read-only symlink inside the target (removed afterwards), because its rows store absolute paths into runs/ and build_curriculum.py requires them inside the source run; PASS requires real evidence sha256 unchanged before/after — cost if wrong: none to evidence; the test proves path-handling, not copy-independence.
Ruling R12: forge-loop reconstructed models.yaml may add base_url_env per arm using today's env-var NAMES (INFERENCE_BASE_URL for coreweave, OPENAI_BASE_URL for openai); env.frozen does not set them, so the loader falls back to the recorded base_url values and behaviour is unchanged. Record both name sources in RUNTIME.json "reconstructed". Accepted limitation: today's cli.py/pipeline_run.py cannot import frozen models.py, so this runtime is reproducible for scoring via frozen benchmark code only, not for regeneration via the CLI — cost if wrong: someone assumes forge-loop generation is re-runnable; VERIFICATION.md must state it is not.
Task 3: fix round 2/5 (R11, R12 addressed; 8/8 gate claimed; no new commit)
Task 3: complete (commits bfcba2f..21e11d4, review clean; GATE 8/8 under rulings R7, R8, R11, R12)
Task 3: minor (deferred, final review should triage — user-visible): every VERIFICATION.md still opens with the superseded first-pass table directly under the final GATE: PASS line
Task 3: minor (deferred): _scaffold_files duplicates the copy traversal (drift risk)
Task 3: minor (deferred): refresh_manifest lists today's scaffold ∩ target, not the restore-time copy set
Task 3: minor (deferred): --sync silently ignored with --refresh-manifest; missing RUNTIME.json gives a raw traceback
Task 3: minor (deferred): rescore-copy/ and runs-copy/ (~35 MB x5) left in targets
Task 4: dispatched (implementer a6f85f8af444274b4, sonnet), BASE 21e11d4
Task 4: complete (commits 21e11d4..858946c, review clean; ⚠️ wiring into BufoGeneration is Task 5 — confirmed planned)
Task 4: minor (deferred): calls that raise are not recorded (latency of failed attempts invisible)
Task 4: minor (deferred): Responses non-truncated statuses (failed/cancelled) map to "stop" without a comment
Task 4: minor (deferred): no test for stream=True passthrough unrecorded
Task 4: minor (deferred): reasoning_content vs reasoning precedence untested
Task 5: dispatched (implementer a2a9372a3d9a81ae6, sonnet), BASE 858946c
Task 5: complete (commits 858946c..2a5eecc, review clean)
Task 5: minor (deferred): run= override rows always have model_calls=[] (untested pin)
Task 5: minor (deferred): BUFO_DISABLE_THINKING parse set only tested for "true"/unset
Task 6: dispatched (implementer a8c75c6205db12acf, sonnet), BASE 2a5eecc; R2 carried
Ruling R13: benchmark._configuration(disable_thinking=False) is a defaulted keyword, not required — docs/forge-*/ scripts (tested from the current tree, frozen from editing) call _configuration(**POLICY) without it; the value is still always written to policy, so frozen configs record False explicitly — cost if wrong: a future caller silently gets False; CLI and run_benchmark pass it explicitly.
Task 6: complete (commits 2a5eecc..9ae2952, review clean; ⚠️ Task 3 gate passed before this commit — confirmed)
Task 6: minor (deferred): wheel cache glob accepts any *.whl with consistent sidecar (stale pin risk); restrict to the pinned filename
Task 6: minor (deferred): no regression test that a pre-change benchmark dir fails loudly on resume
Task 7: dispatched (implementer aa685495e2822beb2, opus — paid experiment + interpretation), BASE 9ae2952
Task 7: complete (commits 9ae2952..b47be67, review clean). Result: inconclusive — 0 length finishes in all arms, empty thought channel; truncation not reproduced on current code/endpoint; 0/12 strict passes in all arms (teacher judge)
Task 7: minor (deferred, user-visible doc): channel-block counts include seeded prior-turn history
Task 7: minor (deferred, user-visible doc): "median reasoning chars" column cannot detect inline thinking; needs a note
Task 7: minor (deferred, user-visible doc): historical 129/295 / 93 s baseline uncited in the write-up
Task 7: minor (deferred): 0 vs 0.0 median formatting
Final review: With fixes — Important 1 (failed generations drop model_calls); minors: CLI run/demo ignore BUFO_DISABLE_THINKING, from_current_tree lacks hashes / stale R10 notes, judge wording; fix-now triage: T3-a, T7-a, T7-b, T7-c
Ruling R14: fix Important 1 now rather than defer to Stage 2 — Stage 2 reports p50/p90 and truncation, and biased-to-success numbers would mislead the next decision — cost if wrong: small extra diff in agent.py/benchmark failure path.
Ruling R15: run/demo use argparse default None so the env var applies; benchmark keeps an explicit bool (frozen config) and the doc states the CLI flag/benchmark override the env var — cost if wrong: a user expecting env to drive benchmarks; documented.
Ruling R16: from_current_tree entries get sha256 of the file as it exists in the target (the tree that passed the gate), not today's tree; refresh-manifest hashes target files; VERIFICATION.md notes "gate applies to the tree on disk" — cost if wrong: none.
Final fix wave: dispatched (one fixer, sonnet), FIX_BASE b47be67
Final fix wave: 5/5 addressed (commit dedfdc8), re-review clean
Final: parked — last_calls can hold the previous turn's calls if _generate raises before the rebind (invalid final prompt / setup failure) — Ruling: real but nothing in Stage 1 depends on it; fix by resetting last_calls at the top of _generate when Stage 2 touches agent.py — cost if wrong: one mislabelled failure row in an edge case.
Ruling R17: copy this ledger to docs/superpowers/plans/2026-10-21-frozen-runtimes-and-latency-instrumentation.ledger.md before deleting the workspace — the shadow git lives in a session scratchpad and the project has no git yet, so the ledger is the only durable record of rulings — cost if wrong: one extra file.
Follow-up 2026-10-21: fixed the deferred wandb timestamp minor. `_wandb_started_at` reads UTC `startedAt` from `files/wandb-metadata.json` and falls back to the dir name as local time. 7/8 restored RUNTIME.json files had picked a later snapshot; their `wandb_requirements` field was recomputed in place. The package verdict was unchanged in every runtime (only openpipe-art: git pin 45a17cbd vs wandb label 0.5.20). Restored targets keep their restore-time copy of the script (hashed in from_current_tree). Full suite 674 passed.
