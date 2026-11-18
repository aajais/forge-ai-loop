# Second data-improvement experiment

The target is `ft-google/gemma-4-26B-A4B-it` approaching `gpt-6.1-sol` on
recognizable Bufo scenes with visible, correct interactions. The completed
development comparison found no quality gain from this iteration. Neither
candidate is approved for promotion.

## Completed quality comparison: 2026-10-20

The [120-generation W&B comparison](https://wandb.ai/wandb-smle/bufo-gen/runs/01c8681c)
contains actual images, metrics, and immutable checkpoint identities.
All five arms used the same updated instructions, tools, rendering budget,
24 development prompts, and `bufo-scene-v2` image scorer.

| Model | Checkpoint | Strict passes / all attempts | Mean requirement coverage |
|---|---|---:|---:|
| `google/gemma-4-26B-A4B-it` | Base | 1/24 | 0.540 |
| `gpt-6.1-sol` | Reference | 21/24 | 0.978 |
| `ft-google/gemma-4-26B-A4B-it` | Previous selected, one epoch | 0/24 | 0.392 |
| `ft-google/gemma-4-26B-A4B-it` | Iteration 2, one epoch | 0/24 | 0.441 |
| `ft-google/gemma-4-26B-A4B-it` | Iteration 2, three epochs | 0/24 | 0.520 |

All 120 attempts produced a valid PNG; one base-model attempt nevertheless
ended with a generation failure and stays in the denominator. There were no
judge errors. Weave publication completed with 120 feedback records, five
native evaluations, and no publication errors.

The frozen selection rule chose the three-epoch candidate by coverage after
a strict-pass tie. This is a diagnostic selection, not evidence of improvement
or authorization to promote the model. The selected adapter subsequently
failed all 16 exact contact-training prompts despite producing 16 valid PNGs.
Each fit prompt has one explicit contact requirement; the zero coverage is
not caused by an absent requirements field.

A separate assistant reviewed 20 images without seeing model identities or
scores: four prespecified prompts across all five arms. The reference passed
4/4 and every Gemma arm passed 0/4. These are assistant reviews of a small
development subset, not human annotations or a held-out quality estimate.
Original decisions remain in `runs/forge-iteration2/validation-blind-review`.

## Failure diagnosis and next controlled test

All 101 accepted training targets match the renderer's assembly of their
authentic final SVG tool arguments. System instructions and tool schemas
match at training and evaluation. No upload or capture mismatch was established.
The exported adapters both report rank 1, which is a constraint, not a proven
cause of the observed failure.

The concrete failure is inconsistent contacts: the model repeatedly labels
shoulder positions as hands, draws the arm tips elsewhere, and places props at
the shoulder coordinates. Clothing then hides some props. The shared planner
converts coordinates but does not bind a hand endpoint to a prop handle.
The selected adapter also copies the worked-example shirt path in 11/24
development images, versus 0/24 for the reference; this suggests template
anchoring but does not prove its cause.

The completed teacher-plan diagnostic compared ordinary generation against
continuation from an authentic teacher plan on the fixed 16 training-contact
prompts. Both arms used the same deployed adapter, temperature zero,
instructions, tools, and rendering budget. All 32 attempts rendered, with no
generation or judge errors. Teacher-prefix trajectories are explicitly marked
mixed-source; both diagnostic arms are excluded from training eligibility.

| Review | Ordinary generation | Supplied teacher plan |
|---|---:|---:|
| Image judge | 1/16 | 6/16 |
| Blinded assistant | 0/16 | 7/16 |

The two disagreements concerned a simplified umbrella grip and whether a
skateboard touched the foot or merely appeared behind the ankles. They remain
recorded rather than being reconciled in favor of a better score.
The [paired diagnostic in W&B](https://wandb.ai/wandb-smle/bufo-gen/runs/faaaac62)
shows all 32 outputs alongside the original 16 reviewed training targets.
An initialization failure was traced to W&B interpreting a top-level serving
URI as an artifact instruction. Read-only checks verified that the failed run
contained no history or artifacts; publication then reused the same run ID
after storing ordinary checkpoint-path metadata. No generations were repeated.
All supplied-plan generations called the planner again: twelve copied the full
teacher arguments exactly, and the remaining four mostly preserved their
contact coordinates. Assistance helped, but correct plans still did not
reliably become correct contacts. This is an in-sample diagnostic, not a new
model-quality claim.

The next experiment is an isolated shared contact renderer with arbitrary
model-authored SVG props and explicit attachment anchors. Its first gate is
twelve rendered engineering fixtures; then the base and reference models
receive both old and new interfaces on a fixed 16-prompt development set.
See `docs/superpowers/plans/2026-10-20-contact-renderer-pilot.md`.
Any later fine-tuning benefit must be measured against the base model using
the same new tool.

The 48 fresh confirmation prompts remain ungenerated. Confirmation is deferred
because neither candidate passed the development gate; preserved selection
and evaluation receipts are unchanged. Any new renderer or training method
requires a separately frozen comparison, with identical capabilities for base,
fine-tuned, and reference models.

Collection and review are complete: 104/104 teacher generations produced valid
PNGs; 101 passed explicit assistant image review (78 train, 23 validation).
Three were rejected for an occluded GPU seedling, an unclear paddle-handle grip,
and a flower stem drawn down the front of an opaque vase. All 16 fixed contact
training targets passed review. Accepted source SHA256:
`1e9d39fb4f79c9b4ec3193d7433d6fa69d6335b7d6dbfc1ec66a33c7ea816841`.

The [eight-image pilot comparison](https://wandb.ai/wandb-smle/bufo-gen/runs/fe11f7a5)
contains actual paired PNGs and assistant observations, not a generalization claim.

The previous selected checkpoint passed 7/80 confirmation cases, versus 65/80
for the teacher and 2/80 for the base model. These are now historical diagnostic
results. The next comparison uses the same updated agent for every model and
48 previously unused confirmation prompts, selected before generating outputs.

The new curriculum contains 80 training requests, 24 development-validation
requests and 48 confirmation requests. Training includes single-contact
examples, complete interactions, scene context, clothing/layer order, and 16
repairs of actual student failures. Automatic critiques in repair inputs are
explicitly labeled unreviewed. Every accepted target requires inspection of its
actual PNG; failed and rejected targets remain recorded.

An observable `plan_scene` tool records contact points, character transforms
and paint order. It checks proposed geometry only, not image quality. Its
calls and responses are actual teacher trajectory data, not fabricated hidden
reasoning. All models receive the same tool and instructions.

## Serving compatibility check

The initial eight-attempt pilot generated four teacher PNGs and four Gemma
HTTP 400 failures. Direct probes isolated a tool-schema regex unsupported by
the server's xgrammar engine. Removing only the regex allowed a valid Gemma
tool call; numeric, length and array constraints remained supported. The fix
retains nonblank-label validation in Python. The original failed pilot remains
in `runs/forge-iteration2/pilot`; the fixed pilot has a separate directory.

The user confirmed higher adapter rank is not supported on the serverless
route. This experiment uses supported training settings: independent one- and
three-epoch jobs, batch size two, all assistant turns and peak learning rate
`1e-4`. Actual exported adapter settings must still be verified. The first
experiment's checkpoints remain unchanged.

## Upload transport diagnosis

The first dataset was created and dry-run validated, but its committed upload
lost the connection before acknowledgment. Three dataset GETs showed unchanged
revision 1 and zero rows; two entries GETs independently returned zero rows.
The ambiguous state and these observations were archived before an explicit
retry. That retry failed during dry-run, before any further mutation.

Controlled dry-run probes isolated a size-sensitive TLS error: one- and ten-row
requests passed, the full 2,150,430-byte payload failed with
`SSLV3_ALERT_BAD_RECORD_MAC`, and the identical full payload passed with a
TLS-1.2-only context. Certificate verification and hostname checks remained
enabled. The diagnostic evidence is in
`runs/forge-iteration2/diagnostics/upload-transport-network.json`. This does not
by itself identify which network component caused the error.
