# SDD ledger — plan: docs/superpowers/plans/2026-10-21-astra-judge-and-calibration.md
Spec: docs/superpowers/specs/2026-10-21-svg-distillation-design.md (read). VCS: shadow git (GIT_DIR in plan Global Constraints). Base: 22e6c86.

## Preflight scan
| pair / task | produces vs consumes | finding |
|---|---|---|
| T1↔T3 | benchmark_metadata.reference keys png_path/png_sha256/generation_id/is_self | consistent |
| T1↔T4 | scores["astra_judge"].strict_pass; strict_pass needs status completed ∧ valid | consistent |
| T2↔T4 | scores["efficiency"].serving_path in summary | consistent |
| T3↔T4↔T5 | T3 load_reference_manifest(path to references.json); T5 --references is a directory in T9 commands | ambiguity → Ruling R1 |
| T4↔T5 | kwargs judge/references/self_reference/baseline_arm/limit_prompt_ids | consistent |
| T4↔T7 | T7 reads astra/scored.jsonl rows via Manifest.read | consistent (BufoGeneration dicts) |
| T4↔T9 | self-equal scoring of sol-run1 writes sol-run1/astra/; a later full astra scoring of sol-run1 (data stage) would collide with the config-change guard | → Ruling R2 |
| T6↔T9 | config/distill_prompts.yaml path | consistent |
| T1 self | astra_result lacks strict_pass; score() adds it; judge_failure carries strict_pass False | consistent |
| T2 self | single-file, tests match signature | ok |
| T4 self | scene branch byte-identical; coverage validator generalised must keep 1-scorer behaviour for existing tests | ok, reviewer checks |
| T7 self | packet neutral names vs key; agreement requires completeness | ok |
| T8 self | tests follow existing harness in chat_ui.test.cjs | ok |
| rubric | no empty tests, no mandated duplication (bootstrap copy is intentional: docs version frozen by tests) | ok |

Ruling R1: `references` (score_benchmark kwarg and `--references`) accepts a directory or a references.json path; a directory resolves to `<dir>/references.json` — because T9 commands pass the directory — cost if wrong: none.
Ruling R2: self-reference astra scoring writes under `<run>/astra-self/` instead of `<run>/astra/`, with its own scorer.json — so the self-equal check never blocks a later real-reference scoring of the same run — cost if wrong: one extra directory name to document.
Task 1: dispatched (BASE 22e6c86, implementer sonnet)
Ruling R3: the lint constraint applies to `src scripts` plus the files a task touches; `tests/` has 152 pre-existing ruff errors that Stage 7 cleans up — because the plan's "ruff check src scripts tests" is unreachable today — cost if wrong: none.
Ruling R4: "retried once" (spec) = two attempts total; the brief's "two retries max" was a wording slip — cost if wrong: one fewer judge attempt on flaky responses.
Task 1: implemented ceec04d (DONE_WITH_CONCERNS: retry scope, missing_image extra) → review dispatched
Task 1: minor (deferred): astra_result validation failures are not retried (only JSON-decode failures); spec says "malformed response is retried once" — decide at final review
Task 1: minor (deferred): choices=None → TypeError skips retry; no tests for missing_image / non-object JSON / rescore=True; inline PIL import; cached judge_error sticks until rescore (plan-mandated, mirrors SceneQuality)
Task 1: complete (commits 22e6c86..ceec04d, review clean)
Task 2: dispatched (BASE ceec04d, implementer haiku)
Task 2: implemented da044b0 (trailer names Haiku; accepted) → review dispatched
Task 2: minor (deferred): load_backends() per score call; test fixture totals equal per-call sums (does not pin token source); commit trailer names Haiku (accepted, subagent's own attribution)
Task 2: complete (commits ceec04d..da044b0, review clean)
Task 3: dispatched (BASE da044b0, implementer sonnet)
Task 3: implemented b44336f → review dispatched
Task 3: review → 1 Important (missing PNG file silently null) + minors; fix round 1 dispatched (resume implementer, FIX_BASE b44336f)
Task 3: fix round 1/5 (3 addressed, 0 open; commits b44336f..46a5b4d)
Task 3: minor (deferred): missing-PNG test uses loose match="a"; full suite not re-run after fix commit (confined change)
Task 3: complete (commits da044b0..46a5b4d, review clean)
Task 4: dispatched (BASE 46a5b4d, implementer opus; carries R1, R2)
Task 4: implemented 85b350f, 452b6d9 (concerns: subset-publish state per arm; Task 9 step 5 path → astra-self/) → review dispatched (opus)
Ruling R5: plan Task 9 step 5 corrected to astra-self/summary.json per R2; Task 9 brief regenerated — cost if wrong: none
Task 4: review → 2 Important (stale per-arm publish state after subset publish; duplicated arm-publication block) + minors; fix round 1 dispatched (resume implementer, FIX_BASE 452b6d9)
Task 4: minor (deferred): self_reference publication records reference_run_id = own run_id (undocumented); pydantic model_fields deprecation warnings from weave in test output (pre-existing pattern)
Task 4: fix round 1/5 (5 addressed, 0 open; commits 452b6d9..db4d017)
Task 4: minor (deferred): dynamic weave.Evaluation subclass + scorer-name keying exercised only with fakes — Task 9 step 4 (first real --publish) is the live check
Task 4: complete (commits 46a5b4d..db4d017, review clean); plan R2 fix committed e7d5d36
Task 5: dispatched (BASE e7d5d36, implementer sonnet; carries R1, R2)
Task 5: implemented 4855fcd → review dispatched
Task 5: minor (deferred): new flags lack help= text; --reference-backend has no choices; missing --prompt-ids-file raises raw FileNotFoundError
Task 5: complete (commits e7d5d36..4855fcd, review clean)
Task 6: dispatched (BASE 4855fcd, implementer sonnet)
Task 6: implemented 4880cae → review dispatched
Task 6: review → 3 Important (loose guard tests, vacuous caption check) + prompt-quality minors; fix round 1 dispatched incl. minors (resume implementer, FIX_BASE 4880cae)
Task 6: fix round 1/5 (4 addressed, 0 open; commits 4880cae..1d06240)
Ruling R6: the semicolon-in-requirement and the ice_cream twohands redundancy are treated as Important (not deferred minors) because Task 9's paid runs consume this prompt set before any later review — fix round 2 dispatched — cost if wrong: one small extra round.
Task 6: fix round 2 commit 6335638 → scoped re-review dispatched
Task 6: fix round 2/5 (2 addressed, 0 open; commits 1d06240..6335638)
Task 6: minor (deferred): fix reports lack raw pytest output; "one scoop on top" fact only implied in scene sentence
Task 6: complete (commits 4855fcd..6335638, review clean)
Task 7: dispatched (BASE 6335638, implementer sonnet)
Task 7: implemented 8da65de (concerns: TDD RED demonstrated after the fact; load_scored_rows added) → review dispatched
Task 7: review → 1 Important (form-restore desync loses reviewer state on reload) + minors; fix round 1 dispatched incl. minors (resume implementer, FIX_BASE 8da65de)
Task 7: minor (deferred): no JS execution test for gating/Blob; load_scored_rows duplicates Manifest.read resolution; unblinding key not hash-bound to items.json; PNG metadata not scrubbed; TDD RED shown after the fact
Task 7: fix round 1 commit 0f3e01c → scoped re-review dispatched
Task 7: fix round 1/5 (5 addressed, 0 open; commits 8da65de..0f3e01c)
Task 7: complete (commits 6335638..0f3e01c, review clean)
Task 8: dispatched (BASE 0f3e01c, implementer sonnet)
Task 8: implemented 7ec0ada (concerns: shadowed text var in try; source-order test for hoist) → review dispatched
Ruling R7: the plan's single "Retry loading models" option cannot fire `change` in a native select; replaced by a selected+disabled "Models unavailable" placeholder plus an enabled retry option — because the spec requires a usable picker — cost if wrong: none (strictly more usable).
Task 8: review → 1 Important (plan-mandated retry option unusable) + minors; fix round 1 dispatched (resume implementer, FIX_BASE 7ec0ada)
Task 8: minor (deferred): retry notice clears a history-failure notice; init awaits history then status sequentially; ok-but-non-object JSON reply unguarded; failedThreads test is a source-order assertion
Task 8: fix round 1 commits 09bba9a,756a324 → scoped re-review dispatched
Task 8: fix round 1/5 (2 addressed, 0 open; commits 7ec0ada..756a324)
Task 8: complete (commits 0f3e01c..756a324, review clean)
Ruling R8: Task 9 runs as one implementer dispatch for steps 1–7 (paid runs, scoring, packet, assistant blind pass) and stops before step 8 (user review) — the executor must never open unblinding-key.json — cost if wrong: a biased assistant label set, which the user's labels override anyway.
Task 9: dispatched (BASE 756a324, implementer sonnet; iter2-e3 checkpoint = wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-iter2-e3-20261016:v3)

Task 9: fix round 1 (controller, inline). Executor BLOCKED at step 4a: `score_benchmark` rejected `--baseline-arm` on arms-v1 because validation families hold 12 prompts and train families 8.
Ruling R9: relax the equal-family-count guard in `statistics.paired_family_interval` and the pre-check in `benchmark.score_benchmark`; the bootstrap resamples per-family mean differences, so unequal sizes stay valid and each family keeps equal weight. Result now reports `balanced`, `prompts_per_family`, and the per-prompt `prompt_difference`. — Validation-only would leave 4 families (useless interval); re-scoring a balanced subset would pay Astra twice. — Cost if wrong: Sol-vs-Sol `difference` on the 176 set weights validation prompts at 2/3 of train prompts; heldout (16×5) is balanced so the confirmation statistic is unaffected. Commit recorded below. Tests: test_statistics (uneven accepted + incomplete pairs rejected), test_benchmark_astra (uneven paired and flagged). Executor resumed at step 4a.
f2379ac fix: allow unbalanced families in the paired family bootstrap

Task 9: fix round 2 (controller, inline). Executor BLOCKED at step 4: every Astra call returned HTTP 400; 176/176 + 48/48 + 47/47 judge errors; three all-error Weave evaluations were published (invalid; URLs in task-9-report.md). Root cause: gpt-6-astra rejects `max_tokens` (wants `max_completion_tokens`) and rejects `temperature=0`. My pre-flight probe used the right parameters, the T1 implementation did not, and T1's tests only asserted the parameters the plan named.
Ruling R10: judge sends `max_completion_tokens=3200` and no temperature; configuration records `temperature: "provider_default"` (new `scorer_configuration`, so the stale `astra/` outputs are rejected on reuse). Added: `score_benchmark` raises `RuntimeError` before `--publish` when there is at least one judge_error and no ok verdict; cached judge_error rows are dropped from `scored.jsonl` and retried on re-run; judge_error text carries exception class + HTTP status only. — Cost if wrong: none for the statistic; the Weave project holds three invalid evaluations that the write-up must list as invalid. Live one-row probe through the fixed `_judge` returned a full verdict (two paid calls: one failed 400, one ok). Full suite green. Commit below. Executor resumed at step 4a with instructions to move the stale astra dirs aside.
a0467a0 fix: astra judge request parameters, all-error guard, retry cached judge errors

Task 9: complete (steps 1-7 by executor, commits e44e7da; step 9 write-up and paired intervals by controller, commit 380c965). Step 8 (user's blinded labels and gate) is pending the user and is outside the executor loop. Headline: Sol run 2 vs run 1 strict 174/176; self-equal 24/24 equal; base 1/48, iter2-e3 4/48 on validation (paired lower bounds −1.000 and −0.917); assistant agreement 0.925, gate not passed (one judge-pass/reviewer-fail on a base Gemma item).
Ruling R11: paired student-vs-Sol intervals computed across run directories with `scripts/paired_across_runs.py` (judge config + reference digest must match) rather than re-running Sol inside the students' directory — no paid calls, same rows, same judge. Cost if wrong: none; the heldout confirmation will score all arms in one directory as the plan intends.
Deferred minors (Task 9): packet download gating verified by code reading only; the script's step1 was launched once early by the executor (correct command, kept); three invalid Weave evaluations remain in the project, listed in the write-up.
Final whole-branch review: base e7d5d36, head 380c965.

Final whole-branch review (opus) e7d5d36..380c965: With fixes. Critical 1 / Important 3 / Minor 16. Full text final-review.md.
Ruling R12 (fix-wave scope, brief final-fix-brief.md, base 380c965): fix Critical 1 (write-up + packet folder unblind the user's review; controller's own hand-off message also named the decisive item's arm → packet v1 set aside as `packet-v1-disclosed/`, packet v2 resampled with seed 20261022, key moved to `calibration-v1/key/`, disclosure recorded in the write-up), Important 1 (Weave republish digest + superseding evaluation), Important 2 (heldout guard with explicit `--confirmation`), Important 3 (record requirement checkboxes; gate = judge pass ∧ reviewer marked a requirement missing; v1-style judge-pass/reviewer-fail kept as secondary; PACKET_VERSION bufo-review-v2), minors 1,3,4,5,7,8,9,10,14,15. — The spec gate must be computable from the user's labels and the review must be blind; both are irreversible once labels exist. — Cost if wrong: one more packet build and a repeat assistant blind pass (cheap, no paid calls).
Deferred to the Stage B plan (not dropped): Minor 2 in-call semantic retry (re-run retry documented as the retry), Minor 11 relative paths in published rows, Minor 12 forge_job_id/forge_dataset_id/sft_config in Evaluation metadata (matters for iter2_e3 and every SFT arm), Minor 13 Sol heldout references produced up front and unjudged before any config change, Minor 16 low-confidence human-review queue, Recommendation: anti-anchoring probe (different-but-correct drawings and near-copies minus one requirement) before training.
Reviewer's declined-to-judge list ruled: CRAFT includes similarity (keep, matches bufo-scene-v2 and plan T1); executor blindness unverifiable (assistant labels advisory); 400-before-billing claim unverifiable offline (write-up says so); cost not computed (user decision).

Final fix wave: implementer (opus) commits a1bb6c2..61121e6 (11 commits, trailers rewritten to Fable 5.1), controller commit d1915db (exclusion of disclosed generations + packet v3: seed 20261023, items sha a39850f9…, five disclosed generation ids excluded, verified by PNG hash), style commit above. Scoped re-review (sonnet): all findings ADDRESSED, no new Critical/Important breakage; three cosmetic notes (two fixed in the style commit; pre-digest Weave records "left as published" is intentional and documented). Full suite 802 passed, node 8/8, ruff clean. Browser smoke test of packet v3 (Playwright over a local server): 40 items, 0 broken images, download disabled at 39/40 and enabled at 40/40, decisions.json carries bufo-review-v2, the packet sha and `missing`; no arm/prompt/model strings in the page.
Ruling R13: a fresh blinded agent labelled packet v3 (advisory, as plan step 7 intended) → runs/distill/calibration-v1/decisions-assistant-v3.json; the user's labels remain the gate. Workspace kept until the user's gate result lands (plan step 8 is the only open step); Stage 7 does git init, so finishing-a-development-branch does not apply here.
