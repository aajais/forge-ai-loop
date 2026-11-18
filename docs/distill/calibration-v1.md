# Stage A: Astra judge calibration, v1 (2026-10-21/22)

Everything here is measured on **train + validation** prompts of `config/distill_prompts.yaml`
(prompt hash recorded in each `benchmark.json`). The 80 heldout prompts were not generated,
judged or reviewed. Judge: `gpt-6-astra` over Chat Completions, rubric `bufo-astra-v1`
(`src/bufogen/scorers/astra_judge.py`), strict JSON schema, two images per call (candidate and a
Sol reference drawing), `max_completion_tokens` 3200, provider-default sampling (the model
rejects temperature overrides). The reference for every prompt is Sol run 1's drawing
(`runs/references/distill-v1/references.json`, `bufo-references-v1`, 176 rows).

Status: **`bufo-astra-v5` with three samples is calibrated** (2026-10-22, packet v4: strict
agreement 0.875, no drawing the judge passed that the user marked as missing a requirement;
record `runs/distill/calibration-v1/calibration-record.json`). `bufo-astra-v1` failed the gate
on occluded contact and eye state; revisions v2–v5 were developed against packet v3 and
confirmed on the fresh packet v4. Two user labels were amended after unblinding, both craft
notes typed into the missing field (packet v3 item 14, packet v4 item 29); the amendments and
the user's reasons are in the record.

## Runs

| Run | Dir | Arms | Prompts | Gen errors | Wall |
|---|---|---|---|---|---|
| Sol run 1 (references) | `runs/distill/sol-run1` | `closed_quality_bar` (`gpt-6.1-sol`) | 176 train+val | 0 | 2134 s |
| Sol run 2 | `runs/distill/arms-v1` (run id `fbd48824…`) | `closed_quality_bar` | 176 train+val | 0 | 1910 s |
| Students | `runs/distill/arms-v1-students` (run id `0c363fc5…`) | `open_base` (Gemma 4 26B-A4B base), `iter2_e3` (`wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-iter2-e3-20261016:v3`) | 48 validation | 1 per arm | 1028 s |

Both Sol runs used the current agent configuration (`--max-turns 100 --max-tokens 6000`,
`--concurrency 4`). Exact commands, wall times and exit codes: `runs/distill/stage-a-log.md`,
reproducible through `scripts/distill_stage_a.sh <step>`.

## Sol run 2 judged against Sol run 1 (judge noise floor on a strong model)

| Split | Strict pass | vs_reference better / equal / worse |
|---|---|---|
| train + validation (176) | 174 / 176 = 0.989 | 1 / 173 / 2 |
| train (128) | 127 / 128 = 0.992 | 0 / 127 / 1 |
| validation (48) | 47 / 48 = 0.979 | 1 / 46 / 1 |

Craft means (176 rows): similarity 0.935, anatomy 0.939, composition 0.951, readability 0.955.
Judge errors 0, low-confidence verdicts 0. There are two strict failures (one train, one
validation).

This is not comparable to the earlier 65/80 Sol figure: that used the scene judge (the teacher
itself) on heldout prompts with no reference image.

Efficiency (Sol run 2): latency p50 41.4 s, p90 55.5 s; mean output tokens 2135, of which
reasoning 241; tool calls mean 1.06; truncated calls 0.

## Base Gemma and iter2-e3 on validation (48 prompts, 4 families)

| Arm | Scored | Strict pass | vs_reference better / equal / worse | Latency p50 / p90 | Output tokens |
|---|---|---|---|---|---|
| `closed_quality_bar` (Sol run 2, same prompts) | 48 | 47 (0.979) | 1 / 46 / 1 | 41 / 56 s | 2135 |
| `open_base` | 48 (incl. 1 gen error, see note) | 1 (0.021) | 0 / 0 / 48 | 17.2 / 36.4 s | 1626 |
| `iter2_e3` | 47 (+1 gen error, counted as fail) | 4 (0.083) | 0 / 2 / 45 | 16.1 / 33.2 s | 1705 |

Note on `open_base`: one of its 48 rows ended with a generation error but still left a drawing,
so the judge scored it (it is in the 48); `strict_pass` requires a completed generation, so the
row counts as a fail whatever the drawing shows. `iter2_e3`'s generation-error row left no
drawing (`missing_image`) and is the one unscored row.

Paired family-cluster bootstrap against Sol run 2 (`scripts/paired_across_runs.py`, 20,000
replicates, seed 20261015; the two arms live in different run directories, which the script
accepts only when judge configuration and reference manifest digest match; results in
`runs/distill/calibration-v1/paired-validation-*.json`):

| Student − Sol | Difference | 95% interval | Families | Balanced |
|---|---|---|---|---|
| `open_base` | −0.958 | [−1.000, −0.917] | 4 × 12 | yes |
| `iter2_e3` | −0.896 | [−0.917, −0.854] | 4 × 12 | yes |

Four families give a narrow but weakly supported interval; the number that matters is the one
on heldout (16 families) at confirmation time. The Sol-level margin is a lower bound ≥ −10 pp.
Today's students are about 90 pp short of it on validation.

## Self-equal check (judge consistency)

Sol run 1's 24 lexically-first validation drawings judged against themselves
(`--self-reference`, `runs/distill/sol-run1/astra-self/`, run id `14f3741b…`): 24 / 24 scored,
0 judge errors, `vs_reference` equal 24 / 24, better 0, worse 0. Strict pass 23 / 24: the one
failure is a drawing that misses one of its own requirements, which is the expected behaviour
(self-reference does not make a requirement miss pass).

## Blinded review packets and agreement

**Packet v1 (set aside).** 40 validation items, seed 20261021. Assistant blind pass, aggregate
only:

| | Reviewer pass | Reviewer fail |
|---|---|---|
| Judge pass | 17 | 1 |
| Judge fail | 2 | 20 |

Strict-pass agreement 0.925, `vs_reference` agreement 0.925 (threshold 0.8 met). Gate: **not
passed** (one judge-pass / reviewer-fail item under the v1 gate). Packet v1 is set aside and is
not used for the calibration verdict. Its files are kept whole as evidence at
`runs/distill/calibration-v1/packet-v1-disclosed/`; do not open it before labelling packet v3.

**Disclosure.** The controller's hand-off message and the first version of this write-up named
the arm and subject of packet v1's gate-breaking item, with the judge verdicts and the
assistant's reasons, before the user labelled. Packet v1 is therefore set aside. A first
resample (packet v2, seed 20261022, kept unlabelled at `packet-v2-unused/`) drew two of the
disclosed drawings again, so the user labels packet v3, built with every generation named in
the v1 write-up or hand-off excluded (`runs/distill/calibration-v1/exclude-disclosed.txt`,
five generation ids, verified by PNG hash to be absent from v3). Per-item detail for any packet
(arm, prompt id, judge verdict, reviewer decision) lives only in
`runs/distill/calibration-v1/key/` and is read after labelling.

**Packet v3.** `runs/distill/calibration-v1/packet/packet.html`, 40 validation items, seed
20261023, built by `scripts/distill_stage_a.sh step6`. The unblinding key is in
`runs/distill/calibration-v1/key/`, outside the packet directory. `items.json` sha256
`a39850f9b895fb01ba83c8c7907238d7b2ac7bd3dca171a11859f31aca23a032`. Decisions use
`bufo-review-v2`: for each item the reviewer ticks every requirement the drawing does not meet
(recorded as `missing`), then marks pass/fail and the comparison with the reference. The gate
is strict-pass agreement ≥ 0.8 and no item the judge passes that the reviewer marked as missing
a requirement; a judge-pass / reviewer-fail without a missing requirement is reported as a
secondary count.

**Assistant blind pass on packet v3** (a fresh agent that saw only the packet directory;
`runs/distill/calibration-v1/decisions-assistant-v3.json`, report
`key/agreement-assistant-v3.json`): strict-pass agreement 0.95, `vs_reference` agreement 1.0;
confusion judge-pass/reviewer-pass 16, judge-pass/reviewer-fail 0, judge-fail/reviewer-pass 2,
both-fail 22; no item the judge passed was marked as missing a requirement, so the advisory
gate passes. The two disagreements are both cases where the judge was stricter than the
reviewer. Advisory only; per-item detail stays in `key/`.

**User review (step 8, done 2026-10-22).** The user labelled all 40 items through the Weave
annotation queue `bufo review packet v3` (`scripts/weave_review_queue.py`; record in
`key/weave-queue.json`); `scripts/distill_stage_a.sh pull` wrote
`runs/distill/calibration-v1/decisions-user.json` and `step7 user` wrote
`key/agreement-user.json`.

| | Reviewer pass | Reviewer fail |
|---|---|---|
| Judge pass | 14 | 2 |
| Judge fail | 0 | 24 |

Strict-pass agreement 0.95, `vs_reference` agreement 0.875. By arm: Sol 14/15, iter2-e3 14/15,
base 10/10. **Gate: not passed.** Four drawings the judge passed were marked by the user as
missing a requirement (two of those the user still passed overall):

| Packet item | Arm | Requirement the user marked missing | Judge evidence (abridged) |
|---|---|---|---|
| 16 | Sol | Bufo's eyes are closed | "curved dark closed-lid lines" (the drawing shows open pupils) |
| 36 | Sol | The stem enters the blue vase | "the rim briefly obscures its continuity" (stem is disconnected) |
| 39 | iter2-e3 | One hand grips the cone; lights glow above the cart | "contact is visible, although most of the hand is obscured" (no hand on the cone) |
| 14 | Sol | An airplane is visible behind Bufo | user note "tail looks too big"; the airplane is visible; user passed the item |

The pattern in three of the four is the same: the judge accepts contact or state that is
implied or occluded ("partly occluded", "obscures its continuity", "largely obscured") as
satisfied. Item 14 reads as a craft note recorded in the missing field; even without it the
gate fails on the other three. Three of the four are Sol drawings, so Sol's 174/176 under
`bufo-astra-v1` overstates its strict pass rate.

Secondary: the two judge-pass / reviewer-fail items (16 and 39) are the two the user failed
outright; there is no judge-fail / reviewer-pass item, so the judge is never stricter than the
user on this packet. The five `vs_reference` disagreements are all cases where the user chose
`worse` and the judge `equal`.

**What changes for v2 (`bufo-astra-v2`):** a requirement that names contact, grip, entry,
touching, holding, or a state such as closed eyes is satisfied only when the contact or state
is directly visible in the pixels; occlusion, continuity "behind" another shape, or inference
from pose counts as not satisfied, and the `evidence` field must say what is visible. v2 is
developed against the 40 user-labelled items of packet v3 (now a development set) and
confirmed on a fresh blinded packet drawn from the remaining validation rows, which the user
labels before any per-item detail is reported.

## Rubric revision against the labelled packet (development set)

After the user's labels failed the v1 gate, packet v3 became a development set: each rubric
revision was run on the same 40 drawings only (`scripts/distill_stage_a.sh dev-rubric
<version> [k]`, outputs under `astra-<suffix>/` in each run directory, never published), the
packet was re-keyed with the new verdicts (`scripts/rekey_packet.py`), and the user's unchanged
decisions were compared again. The confirmation is a fresh packet the user has not seen.

| Rubric | k | Strict agreement | vs_ref agreement | Judge pass, user missing | Judge fail, user pass | Gate |
|---|---|---|---|---|---|---|
| v1 | 1 | 0.950 | 0.875 | 14, 16, 36, 39 | none | no |
| v2: visibility rule + "worse not equal" | 1 | 0.925 | 0.750 | 14, 16 | 06, 36 | no |
| v3: visibility rule + eye-state rule | 1 | 0.900 | 0.825 | 14, 39 | 06, 32, 36 | no |
| v3 | 3 | 0.900 | 0.825 | 14, 39 | 06, 32, 36 | no |
| v4: v3 + grip needs visible fingers | 3 | 0.875 | 0.800 | 14 | 02, 06, 32, 36, 38 | no (item 14 only) |
| v5: v3 + grip rule, mitt hands allowed | 3 | 0.900 | 0.825 | none after item 14 corrected | 02, 06, 32, 36 | yes (development) |

Item numbers are the packet labels (index + 1). Reports: `key/agreement-user-<suffix>.json`.

What each step established:

- **v2.** The visibility rule and the per-check `directly_visible` flag turned the disconnected
  stem (36) and the hidden hand on the cone (39) into failures, as the user marked. The judge
  never used the flag to override itself (0 overrides in every run): when it saw occlusion it
  set both fields false. The extra sentence asking for `worse` rather than `equal` when
  requirements are met less clearly moved six Sol drawings from equal to worse against the
  user's equal, so it was dropped.
- **v3.** An explicit definition of closed eyes (no eyeball, pupil, iris or highlight; heavy
  lids are open) fixed item 16, stably: 0 of 5 passes in the probe below. But item 39 passed
  again under the same visibility rule it had failed under in v2.
- **Stability probe** (`runs/distill/calibration-v1/noise/v3-k5.json`, five independent v3
  verdicts per item): item 39 passed 2 of 5 with three different unsatisfied sets, item 06 (key
  tip enters keyhole) 2 of 5; items 14, 16 and 32 were 5 of 5 consistent. Single verdicts on
  borderline contact are unstable, so the judge gained `samples=k`: k independent calls merged
  so a requirement is satisfied only when every sample says so (craft and confidence averaged,
  comparison from the strictest sample). Even k=3 passed item 39 in all three samples once, so
  sampling reduces but does not remove that false pass; the rubric had to name the failure.
- **v4.** A grip definition requiring visible fingers made item 39 fail 0 of 3 and left item 14
  as the only judge-pass / user-missing item (the user passed that item and called it equal; the
  airplane the requirement asks for is visible, and the missing mark carries the note "tail
  looks too big"). The cost was three Sol grips the user accepted (items 02, 32, 38: a rounded
  mitt over a key bow or baton end, drawn without fingers in this style), so Sol-arm agreement
  fell to 0.667.
- **v5** keeps v3 plus a grip rule without the finger requirement: a distinct hand shape, mitt
  or fingers, overlapping the object with no gap counts; a hand mostly behind the object, a bump
  or sliver at its edge, a vanishing arm, or a prop overlapping the body does not. Result: the
  three real false passes (16, 36, 39) stay failed with 0 of 3 votes each, item 38 passes again,
  and items 02 and 32 (mitt over a key bow; batons ending at the fists) still fail against the
  user's pass, by split votes on 02. Item 14 is the only judge-pass / user-missing item left.
  **v5 with k=3 is the candidate for the full re-judge and the fresh confirmation packet.**
- **Item 14 resolved by the user** (2026-10-22): the airplane is visible but badly proportioned,
  a craft complaint rather than a requirement miss, so the user set the missing field to `none`
  in the queue. Re-pulled and re-scored: v1 still fails the gate (items 16, 36, 39); **v5 k=3
  passes the development gate** (agreement 0.90, no judge-pass / user-missing item, four
  judge-fail / user-pass items). This is the development set, so it is not the calibration
  verdict; that comes from packet v4.

Three of the four v1 false passes were Sol drawings, so Sol's 174/176 under v1 is an
overestimate; the revised judge's Sol pass rate on train + validation is reported after the
full re-judge.

## Full re-judge under bufo-astra-v5, three samples (2026-10-22)

`scripts/distill_stage_a.sh rejudge bufo-astra-v5 3`: every row scored three times and merged
conservatively; outputs in `astra-v5-k3/` (Sol run 2, students) and `astra-self-v5-k3/`
(self-equal). 0 judge errors; no HTTP 4xx/5xx across the run. The v1 directories are untouched.

| Arm | Rows | Strict pass v1 | Strict pass v5 k=3 | vs_reference v5 better / equal / worse |
|---|---|---|---|---|
| Sol run 2, train + validation | 176 | 174 (0.989) | 154 (0.875) | 2 / 153 / 21 |
| Sol run 2, train | 128 | 127 | 115 (0.898) | |
| Sol run 2, validation | 48 | 47 | 39 (0.813) | |
| `open_base`, validation | 48 | 1 | 0 | 0 / 0 / 48 |
| `iter2_e3`, validation | 47 (+1 gen error) | 4 | 0 | 0 / 0 / 47 |
| Sol run 1 self-equal | 24 | 23 | 19 (0.792); equal 24 / 24 | |

Ten Sol rows had split samples and resolve to fail. Sol's most-missed requirements under v5:
"Bufo's head, torso and limbs form one connected figure" (3), "An arrow lies on the bow aimed
toward a visible target" (2), then one each for a baton grip, a key tip entering a keyhole, a
stem entering a vase and closed eyes. The self-equal check still has no comparison noise (24
of 24 equal); its strict rate fell because five of Sol run 1's own drawings miss a requirement
under the stricter reading, which is what self-reference should show.

Paired family-cluster bootstrap on validation, v5 k=3 (`paired-validation-v5-k3-*.json`):

| Student − Sol | Difference | 95% interval | Families |
|---|---|---|---|
| `open_base` | −0.813 | [−0.875, −0.750] | 4 × 12, balanced |
| `iter2_e3` | −0.813 | [−0.875, −0.750] | 4 × 12, balanced |

The Sol-level bar (lower bound ≥ −10 pp) is now measured against a Sol that passes 81% of
validation prompts rather than 98%; the students are at zero.

Weave evaluations (project `wandb-smle/bufo-gen`): Sol run 2
https://wandb.ai/wandb-smle/bufo-gen/r/call/01a16490-383e-78ad-999e-b0ae8a4249a6, base
https://wandb.ai/wandb-smle/bufo-gen/r/call/01a1649e-bda8-7125-a6c0-464598d06cbc, iter2-e3
https://wandb.ai/wandb-smle/bufo-gen/r/call/01a1649f-03e4-72b3-a41a-6274a340073b, self-equal
https://wandb.ai/wandb-smle/bufo-gen/r/call/01a164a3-36b5-7b1e-8458-e1d4bac3be6c. Their names
are `bufo-astra-<model>-v5` without the intended `-k3`: the publish step missed the sample
count (fixed afterwards, commit e3e2302). Each evaluation's scorer object records `samples: 3`
and `rubric_version: bufo-astra-v5`, and no single-sample v5 evaluation was ever published, so
the `-v5` names denote these k=3 runs.

## Confirmation packet v4 (user labels in, 2026-10-22)

`scripts/distill_stage_a.sh packet-v4 bufo-astra-v5 3 20261024`: 40 validation drawings
drawn from rows the user had not labelled (the 40 of packet v3) or seen named (the 5 disclosed
generations), keyed with the v5 k=3 verdicts in `runs/distill/calibration-v1/key-v4/`, Weave
queue `bufo review packet v4` (record `key-v4/weave-queue.json`), `items.json` sha256
`a0425b490eb8c40e7daff5e0b542b80d31b8d55111b6184cb63dc52d4c4e8ce0`. The user labelled all 40
in the queue; `decisions-user-v4.json`, report `key-v4/agreement-user.json`.

| | Reviewer pass | Reviewer fail |
|---|---|---|
| Judge pass | 15 | 1 |
| Judge fail | 4 | 20 |

Strict-pass agreement 0.875, `vs_reference` agreement 0.85. By arm: Sol 15/19, iter2-e3 8/9,
base 12/12. First pass of the gate: **not passed** on one item, packet item 29 (a Sol
ice-cream-cart drawing), which the judge passed 3 of 3 samples on every requirement and the
user failed, marking "the other hand holds a scoop utensil" and "a pink scoop touches the top
of the cone" as missing with the note "hand and cone are overlapping". Unblinded, the judge's
evidence names the utensil handle ending in the raised fist and a pink-to-orange boundary where
the scoop meets the cone; the utensil arm crosses in front of the scoop. Asked whether that was
a requirement miss or a craft fault, the user ruled craft fault, cleared the missing marks in
the queue and kept the item as fail / worse. Re-pulled: judge-pass / reviewer-missing none,
judge-pass / reviewer-fail 1 (item 29), judge-fail / reviewer-pass 4 (items 01, 24, 28 Sol;
21 iter2-e3, all grip or contact read more strictly by the judge). **Gate: passed.** The judge
is never more lenient than the user on a requirement in this packet.

Caveat stated plainly: both gate failures (packet v3 item 14, packet v4 item 29) were resolved
by the user re-reading a label after unblinding, in both cases a craft complaint recorded in
the missing field. The queue guidelines now say the missing field is for requirements that are
literally unmet. A reader who wants the pre-amendment numbers has them above and in
`stage-a-log.md`.

## Weave evaluations

Project `wandb-smle/bufo-gen`. Scorer objects: Sol run 2
`bufo-astra-gpt-6.1-sol:a82T4HQe…`, base `bufo-astra-google-gemma-4-26B-A4B-it:vUXXPI3N…`,
iter2-e3 `bufo-astra-ft-google-gemma-4-26B-A4B-it:csdCaBGt…`, self-equal
`bufo-astra-gpt-6.1-sol:U3TI1aGb…`. Evaluation calls:

- Sol run 2: https://wandb.ai/wandb-smle/bufo-gen/r/call/01a1612f-5e82-7b40-902a-dbd632f054fa
- base: https://wandb.ai/wandb-smle/bufo-gen/r/call/01a1613b-3054-7e1c-9aed-6d4946df96db
- iter2-e3: https://wandb.ai/wandb-smle/bufo-gen/r/call/01a1613b-7171-7504-82e3-9cb60bfdd7f9
- self-equal: https://wandb.ai/wandb-smle/bufo-gen/r/call/01a1613d-d7f2-7627-8ca0-556390837069

Judge cost is not in the local publication record; Weave's cost table has `gpt-6-astra`, so
the per-call cost shows in the UI. `gpt-6.1-sol` is not in the table, so Sol generation cost
is not shown.

**Invalid publications.** The first scoring pass sent `max_tokens` and `temperature=0`, both
rejected by `gpt-6-astra` with HTTP 400; every row was a judge error and three evaluations were
published before the no-verdicts guard existed. Treat them as invalid:
`…/call/01a16117-50c2-7082-8324-84b51bfacc5c` (Sol), `…/call/01a16119-3289-7ca6-82b5-ffef0072e6ee`
(base), `…/call/01a16119-8411-76b6-8370-8a42aea2b62c` (iter2-e3). Their local outputs are in
`runs/distill/arms-v1/astra-invalid-400` and `runs/distill/arms-v1-students/astra-invalid-400`.
Those requests were rejected before billing.

## Fixes made during the runs

- `paired_family_interval` and `score_benchmark` accepted only equal prompt counts per family;
  the enlarged validation families (12) against train (8) tripped it. Relaxed: families keep
  equal weight, the result reports `balanced`, `prompts_per_family` and a per-prompt
  `prompt_difference` (commit f2379ac).
- Judge request parameters (`max_completion_tokens`, no temperature); `score_benchmark` raises
  before publishing when the judge produced no verdicts; cached judge-error rows are retried on
  re-run; judge-error text carries exception class and HTTP status only (commit a0467a0).

## What this does and does not establish

- The judge reads two images and a strict schema reliably: 0 judge errors over 295 calls, a
  zero-noise self comparison, and 98.9% strict pass for Sol against a Sol reference.
- The judge is independent of the teacher and of every student.
- Whether the judge agrees with a human closely enough to stand in for one is decided by the
  user's labels on packet v3, not yet in hand.
- The student gap on validation is large (about 90 pp). Nothing here is a heldout claim.
