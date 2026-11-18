# Pilot image-judge calibration audit

The current Qwen judge is **not ready to act as the sole RL reward or automatic quality gate**. It passed both assistant-reviewed visual negatives, including a clean-looking image that violates a specific spatial constraint. It also has a requirement-serialization failure and an unresolved exact-text policy difference. Its persisted `calibration_status: uncalibrated` should remain unchanged.

This audit compares the 16 individually inspected Sol final PNGs in `runs/forge-loop/pilot/assistant-reviews.jsonl`, plus the root assistant's Gemma concert negative in `assistant-negative-reviews.jsonl`, against actual `scores.scene_quality` records in `scored.jsonl`. The pilot run ID is `05266853b35d40bcb0d77531e6d89aab`. Judge configuration: `Qwen/Qwen3.8-27B`, CoreWeave, temperature 0, `bufo-scene-v1`, threshold 0.65, rubric hash `9a567f46f96f7a0f67ae724d955bf5475ae93334ed3e013029af04a896866b89`.

All labels are assistant reviews, not human annotations. The 16 Sol PNGs were inspected before this comparison; the three contested images were viewed again during the audit. Existing review files were not changed. Review joins use generation IDs and verbatim requirement strings. Visual truth is `visual_quality_pass`, **not** SFT `decision`: the two visually passing edit trajectories were rejected for training eligibility and must not count as visual negatives. Validation examples retain their validation partition.

## Observed agreement

A usable judge result requires `status == "ok"` and a non-null `quality_failure`. A judge pass means `quality_failure == false`. Null/error results are abstentions, not passes or negative image labels.

| Comparison | Count |
|---|---:|
| Assistant-reviewed images | 17 |
| Valid judge results | 16 |
| Judge error / abstention | 1 |
| True visual positives | 13 |
| True visual negatives | 0 |
| False visual positives | 2 |
| Nominal false visual negatives | 1 |
| Image agreement among valid results | 13/16 (81.25%) |
| Matched checklist items | 58 |
| Checklist agreement | 53/58 (91.38%) |
| Checklist false positives | 4 |
| Checklist nominal false negatives | 1 |
| Unavailable checklist items from judge error | 4 |

The nominal false negative is an exact-text interpretation disagreement, described below, rather than a visual perception error. Excluding that ambiguous image leaves 13 agreements and 2 false positives across 15 valid cases. Both reviewed negatives passed the judge: empirical negative detection is 0/2. That is a concrete failure on these examples, not a reliable population estimate. This small, teacher-heavy sample and one deliberately selected Gemma negative cannot establish calibration, generalization, a safe threshold, or a useful RL reward ranking. All four validation images passed both reviewers, so validation supplies no negative examples here.

| Prompt | Backend | Assistant visual pass | Judge result |
|---|---|---|---|
| train-f1_presentation-01 | Sol | yes | pass |
| train-concert_stage-01 | Sol | yes | pass |
| train-concert_stage-02 | Sol | yes | judge_error |
| train-f1_presentation-02 | Sol | yes | fail: title policy disagreement |
| train-gpu_garden-01 | Sol | yes | pass |
| train-office_coffee-01 | Sol | yes | pass |
| train-skate_park-01 | Sol | yes | pass |
| train-pizza_kitchen-01 | Sol | yes | pass |
| train-basketball_court-01 | Sol | no | pass: false positive |
| train-workshop_repair-01 | Sol | yes | pass |
| train-video_call-01 | Sol | yes | pass |
| train-rainy_commute-01 | Sol | yes | pass |
| validation-airport_marshaller-01 | Sol | yes | pass |
| validation-ice_cream_cart-01 | Sol | yes | pass |
| validation-florist_arranging-01 | Sol | yes | pass |
| validation-locksmith-01 | Sol | yes | pass |
| train-concert_stage-01 | Gemma | no | pass: false positive |

## Exact disagreements and reinspection

### Basketball: missed below-waist relation

Generation `99dffb4a2b624d5289cdf525c0e1a0da`, [original conversation](https://wandb.ai/wandb-smle/bufo-gen/weave/agents/conversations/05266853b35d40bcb0d77531e6d89aab%3Aclosed_quality_bar%3Atrain-basketball_court-01).

Requirement, verbatim: **“One hand contacts an orange basketball below the waist”**.

The assistant marked false. Qwen marked true and described the ball as at hip level, below the waist. Re-viewing the PNG confirms that contact occurs near sleeve/upper-torso height, around y=350; the ball center is around y=396, while the jersey hem is near y=420. The hand touches the ball, but the required below-waist relationship is missing. Court lines and attached blue jersey pass both reviews.

Qwen assigns similarity 0.85, anatomy 0.90, composition 0.90, readability 0.95, and constraint coverage 1.0. Thus the error is not a low score narrowly crossing 0.65: the spatial violation receives strong scores and full coverage. Raising a scalar threshold alone is not an evidenced solution.

### Gemma concert: contact, clothing, and foreground placement

Generation `a4c47892125443f4a70723b3e7ea586d`, [original conversation](https://wandb.ai/wandb-smle/bufo-gen/weave/agents/conversations/05266853b35d40bcb0d77531e6d89aab%3Aopen_base%3Atrain-concert_stage-01).

| Requirement, verbatim | Frozen assistant label | Qwen label | Reinspection |
|---|---|---|---|
| One hand visibly grips a microphone at his mouth | false | true | Microphone head sits below the mouth; the tiny hand is ambiguous and the handle extends down through the clothing. |
| Blue jacket is attached to torso and sleeves align with arms | false | true | Clothing is formed from overlapping boxlike panels; a dark horizontal foreground band cuts across the chest/sleeves. Clear arm-to-sleeve alignment is absent. |
| An audience sits below the stage | false | true | Dark audience-like circles occur at the torso-level horizontal band rather than forming a legible audience below the stage. |

Both reviews agree that feet and two spotlights are visible. Qwen gives anatomy 0.75, composition 0.80, readability 0.85 and coverage 1.0, treating ambiguous relationships as satisfied. The assistant gave anatomy 2/4, composition 2/4 and readability 3/4. Reinspection supports preserving the existing visual rejection; individual boundary judgments remain assistant evidence rather than human ground truth.

### F1 edit: exact-text policy is underspecified

Generation `9a76f40d37a542e6bb4f720da9c0d25a`, [original conversation](https://wandb.ai/wandb-smle/bufo-gen/weave/agents/conversations/05266853b35d40bcb0d77531e6d89aab%3Aclosed_quality_bar%3Atrain-f1_presentation-02).

Requirement, verbatim: **“Screen title is F1”**.

The rendered title is visibly **F1 DESIGN**. The assistant allowed the extra descriptive word as a minor variation preserving F1; Qwen required the title to be only F1 and marked false. Both correctly read the image. This is a nominal false negative relative to the frozen label, but should be adjudicated as a rubric-policy disagreement before using it to tune rewards. The other three requirements agree. The long arm is connected and the finger reaches the wheel; the assistant scored anatomy/composition 3/4 rather than treating those relationships as broken.

If exact text is mandatory, specify “reads exactly F1, with no other words” and apply the policy consistently to future labels. If including F1 is sufficient, say so explicitly. Do not silently flip a frozen label to match the judge. This image remains SFT-ineligible due to its cumulative trajectory regardless of the visual-policy outcome.

## Judge error: curly apostrophe changed in requirement

Sol concert edit generation `4883f02428c9475bb4589c0dadf7076a` has persisted `status: judge_error`, null scores, null `quality_failure`, empty `prompt_checks`, and `ValueError: image judgment failed`.

The retained raw response supplies four checks, but changes **“Microphone is held in Bufo’s left hand at his mouth”** to **“Microphone is held in Bufo's left hand at his mouth”**. The curly apostrophe U+2019 became ASCII U+0027. This mismatch is consistent with the failure of exact requirement validation; no source-code change or retry was performed in this audit. The raw response must not be substituted for the failed validated result. It also describes the non-microphone hand as on the guitar neck, whereas the PNG places it on the guitar body near the strings, showing that plausible explanatory text is not itself reliable evidence.

For a future implementation, stable requirement IDs can separate matching from verbatim display text. Preserve original requirement strings in records, validate completeness and order/identity, and explicitly handle any text normalization rather than silently accepting arbitrary paraphrases. Error and missing-score paths must continue to abstain.

## Consequences for RL use

The judge can remain a diagnostic scorer with explicit uncalibrated status. It should not currently qualify examples, select winners, or supply an unguarded optimization reward: recognizable Bufo and attractive scenes can receive near-maximal reward while violating contact/spatial requirements. Training against that signal risks reinforcing precisely those violations.

Before considering RL use, resolve the exact-text policy, fix the requirement-identity failure, and build an independently reviewed set containing both passes and failures for contact, relative position, count, clipping, hand assignment, clothing attachment and occlusion. Include difficult negative and near-boundary cases across both generators, and reserve untouched families for evaluation. Measure negative detection and checklist errors by failure mode; verify ranking on matched alternatives and repeated scoring reliability. Sample count alone is not a calibration criterion, and this audit does not designate a replacement threshold or certify the judge.

The nine accepted train targets remain justified by explicit assistant image review and separate trajectory checks, not by this judge's scores. The four validation examples remain excluded from training.
