# Contact renderer experiment progress

The goal remains measured improvement of `ft-google/gemma-4-26B-A4B-it`
toward `gpt-6.1-sol`, using Forge and the AI Loop. A tool improvement alone
is not a fine-tuning improvement.

## 2026-10-20 continuation

- The twelve final engineering fixtures passed independent assistant inspection.
  All scene/SVG/PNG hashes match the frozen compiler. Crossed wrists are now
  connected. The deliberately occluded fixture is a negative control;
  acceptance does not claim that fine contacts are legible at 64px.
- The 64-attempt comparison ran under
  `runs/forge-contact-pilot/comparison/pilot-protocol.json`. Source files for
  that run remain frozen.
- The old SVG interface's provisional image judge passed 0/16 for
  `google/gemma-4-26B-A4B-it` and 16/16 for `gpt-6.1-sol`. These are
  development prompts; all attempts are included.
- A real Gemma call in the contact arm supplied a valid JSON document with
  `objects[0].attach.to="none"`. Local validation reproduces the specific
  error, “Anchor reference must be object.anchor.” The SDK wrapper exposed
  only “Invalid JSON input for tool render_scene” to the model, hiding the
  actionable cause. Saved evidence:
  `comparison/contact/sdk-failure-trace-sample.json`.
- Ruling: preserve the current run and introduce a separate v2 agent with
  explicit `attach=null` guidance and field-level validation feedback for
  both models. Do not change the live experiment or describe the change as
  training. This costs an additional matched development comparison.
- The publisher requires all 64 model-blinded assistant decisions, including
  missing-image failures, before publishing. It binds reviews to exact PNG
  hashes and includes both assistant and automated judgments.
- The separately implemented training-only validation adapter preserves the
  actual `render_scene` contract and captured history. It does not rename tools
  or fabricate successful trajectories.

No checkpoint has been promoted. The 48 reserved confirmation outputs remain
ungenerated. The teacher-reliability and remaining-gap checks qualified the
single training-fit trial described below.

## Completed comparisons and qualified targets

The original 64-attempt comparison is published with paired PNGs, every failed
attempt, automated scores and all assistant reviews:
https://wandb.ai/wandb-smle/bufo-gen/runs/876b9b82.
The original review packets hid model IDs but disclosed the interface in file
paths; that limitation is recorded in the publication. Subsequent packets copy
PNGs to neutral filenames and verify the copied bytes against capture hashes.

The v2 development comparison completed all 32 attempts. Provisional image-judge
passes are 2/16 for `google/gemma-4-26B-A4B-it` and 16/16 for `gpt-6.1-sol`;
Gemma had ten generation errors. These are tool/instruction diagnostics, not a
fine-tuning result. All three assistant review packets have been frozen.
Assistant passes were 2/16 and 15/16 respectively; requiring both assessments
to pass the same image gives 1/16 and 15/16. The complete paired comparison is
published at https://wandb.ai/wandb-smle/bufo-gen/runs/57fc0b12.

The new 24 teacher targets all passed explicit visual review and each contains
exactly one successful `render_scene` call. Offline training qualification
passed with 24 train entries and zero validation uploads. The eight new
diagnostics and 48 reserved confirmation IDs are excluded.

- Source SHA256:
  `431640b678b2757cd6f76bd2866ff69340d1b3793b7629736e49d989c9bc245f`
- Forge upload SHA256:
  `365d75eda892534ce5c04c9451709b0bf9ef0d94fd182431e94814718c969a52`
- Contact contract SHA256:
  `8acf296477d900c1fb04e95ad1ac0dda869ff795370b4d9d1c3dad9e31df3379`
- Prompt version: `bufo-contact-v2-a6e61653874f`

The frozen single-candidate recipe is six epochs, batch size two, peak learning
rate 0.0001 and all assistant turns, independently initialized from the base
model. The 72 updates are nominal until verified from actual training history.
The independent lifecycle review was required before submission.

The lifecycle review passed (18 tests independently verified), and Forge accepted
job `61b53c05-d136-4e13-9375-56517ebd647b`, dataset
`f3831043-1af1-4b04-8d23-dd98c0ccfdf0`, revision 2. Its current observed status is
training; no checkpoint is yet available. The unchanged Gemma baseline has
completed and published all 24 fit prompts, including eight generation
failures and seven missing PNGs. One failed attempt retains a partial PNG.
Its provisional image-judge pass count is 7/24, with zero judge errors.
Partial images are retained for review but cannot pass the completed-agent gate.

Ruling before candidate inference: interpret the 20/24 training-fit gate
conservatively as agreement between the image judge and explicit assistant
visual review. Report each assessment separately and retain disagreements.
This can reject some genuinely acceptable images, but avoids scaling on the
known false-positive judgments seen in the development pilot. Reviews will mix
all 24 baseline and 24 candidate attempts using neutral image filenames.
The gate remains an in-sample learnability test, not a generalization claim.

The conditional RL adapter and reward-calibration requirements are mapped in
[rl-next-step.md](rl-next-step.md). No contact-agent RL job has been submitted.
Neither completing SFT nor reducing training loss is sufficient to clear the
fit gate or reward-calibration gate.

### Service observation, 16:36 UTC

Forge still reports `training`, 0% progress and no returned checkpoint; its
`training_started_at` is null. The exact model-name W&B run has no training
history rows. Its logged `lora` artifact
`bufo-gemma-contact-v2-fit-20261020:v0` has aliases `step0` and `latest`;
this is initialization evidence, not a trained candidate. The artifact identity
and metadata observation are saved under `training/artifact-observations`.

The public Forge response does not expose the serverless training-job or model
ID needed for ART's event/checkpoint APIs. Its `task_id` is the project UUID and
must not be substituted for either. An internal operator lookup was requested
using the actual Forge job UUID. The last status check at 16:40 UTC still
reported training at 0%, with no returned checkpoint.
No replacement or retry job was submitted.

### Dataset fidelity and evaluation readiness

The actual Forge dataset artifact was downloaded and compared to the frozen
upload, joining its reordered rows by unique exact user messages. All 24
system instructions, tool schemas, argument strings, tool result strings and
assistant replies match. Only consistently linked tool-call IDs changed.
See [the service dataset audit](service-dataset-fidelity-audit.md) for artifact
and file hashes and the remaining limitation: this does not establish hosted
token masks, sequence retention or completed optimizer updates.

The mixed fit-review helper and W&B publisher passed 35 focused tests and
independent read-only review with no blockers. They preserve all 48 planned
attempts, require the fixed candidate's deployed checkpoint, use neutral image
paths, and distinguish image-only verdicts from completed-agent successes.
No candidate generations, fit-review packets or fit publication exist yet.
The next execution remains the exact returned checkpoint on the 24 training
prompts, followed by three mixed 16-image review packets and the frozen fit gate.
Root verification: `pytest -q tests/test_contact_fit_compare.py
tests/test_contact_fit_publisher.py` completed with 35 passed in 9.53 seconds.

### User-requested runtime budget change

After the user reported `MaxTurnsExceeded`, production chat, pipeline and new
benchmark defaults were raised from 8 to 100 turns. The chat request timeout
was raised from five to thirty minutes; CLI `--max-turns` now also reaches
`run` and `demo`. A real SDK loop with mocked model transport reproduced the
eight-turn failure and subsequently completed twelve tool rounds plus a final
reply, including a conversational follow-up. All 47 targeted CLI, pipeline,
UI and benchmark tests passed.

The historical contact scripts, saved configurations, training uploads and
frozen runtime copies remain unchanged. Their explicit eight-turn policy is
historical evidence. Because production source hashes have now changed,
current-source replay and publication guards intentionally reject the old
configuration. Continue that experiment only from its exact frozen runtime,
or create a new matched comparison with the increased budget for every arm.
Do not compare a new 100-turn candidate against the old eight-turn baseline
as a fine-tuning-only gain.
