# Curriculum review before generation

Final verdict: **approved after pre-generation corrections**. The initial findings are retained below as review history.

## Re-review of corrected curriculum

The corrected YAML SHA-256 is `6f31c0774cf37f8ff901172e74a28856b0a34cf75156dc31e37f606844ac9bfc`. Rechecked 128/24/80 counts, 232 unique IDs/prompts, pairwise split-family disjointness, disjointness from original heldout families, generator parity, and manifest hash: all pass.

Every requirement now appears verbatim in a `Visible details` clause in its prompt. This resolves the unspecified-prop scoring issue and makes the blue bicycle-frame binding explicit. The clothing requirement now asks for head and hands inside the frame rather than an unobscured face, so binocular eye occlusion is allowed. The minimal variant retains every requested scene element against a pale backdrop and scores flat colors rather than an empty background. No blocking contradictions remain.

The optional carrot-location, chalk-outline, bounding-height and stale-comment clarifications below remain nonblocking. Family-cluster uncertainty remains necessary for the eventual paired report. No confirmation outputs were generated or inspected during either review.

Reviewed `config/forge_gap_prompts.yaml`, its generator `docs/forge-gap-work/build_curriculum.py`, and `docs/superpowers/plans/2026-10-15-gemma-quality-gap.md`. This was a prompt/requirements audit only: no confirmation outputs were generated or inspected. Product/config files were not edited by this reviewer.

## Verified structure

- 128 training prompts across 16 families, 24 validation prompts across 4 families, and 80 confirmation prompts across 16 families.
- All 232 IDs and prompt strings are unique.
- Train, validation, and confirmation family IDs are pairwise disjoint and all are disjoint from the original 20 heldout family IDs.
- The YAML matches the generator's returned rows exactly. YAML SHA-256 at review: `9534cdeb054f63eba83113cd9be1cc1f7acd483c0900d076107a4664d827931f`; the curriculum manifest records that same hash.
- The original heldout scenes have distinct activities from the new scenes on semantic inspection. Shared general skills such as tool contact, two-handed grips, or a prop on a work surface are intentional transfer rather than duplicate scenarios.
- Validation selection before opening confirmation is explicit in both plan and manifest. No input here establishes trajectory/image leakage; perform those checks again when importing actual generated examples.

## Blocking findings

### 1. Scored constraints are absent from several prompts

A scene name does not entitle the judge to demand an arbitrary background object. Under strict all-requirements scoring, a perfectly reasonable response could fail for never-requested content. Add the scored facts explicitly to each family action/scene text, which fixes every variant of that family. Keep the requirement lists; make the model request match them.

| Family | Detail to state explicitly |
|---|---|
| concert_stage | Visible stage spotlights |
| pizza_kitchen | A visible counter or oven |
| basketball_court | Visible court lines |
| rainy_commute | Visible rain and city buildings |
| film_set | A visible film camera on a tripod |
| ping_pong | A visible central table net |
| laundry_room | A visible washing machine |
| bakery | Dough rests on a work surface |
| tennis_court | Visible tennis net or court lines |
| airport_marshaller | A visible airplane behind Bufo |
| ice_cream_cart | The other hand uses a scoop utensil, rather than placing the scoop with bare fingers |
| birdwatching | A visible bird on a nearby branch |
| car_wash | Visible soap bubbles around the sponge contact area |
| bowling_alley | Visible pins at the far end of a lane |
| spacewalk | Visible stars and a planet |
| golf_green | A visible flagged hole |
| drum_lesson | The second drumstick is raised |
| soccer_pitch | The other foot is planted, with visible pitch markings |

Some scene props are plausible implications, but the extra words remove the avoidable ambiguity at essentially no cost. Also clarify the science-fair funnel is supported by the other hand if that is intended; current requirements only enforce the pour and flask connection.

### 2. Birdwatching clothing variant contradicts its action

`gap-heldout-birdwatching-clothing` requires binocular eyepieces against the eyes and also requires the face to remain unobscured. Binoculars necessarily cover part of the face. Override this variant with wording and checks such as: “Add a fitted blue vest. Keep the mouth, cheeks, and gripping hands visible; the binoculars may cover the eyes.” Keep the eye-contact requirement unchanged.

### 3. Minimal background wording conflicts with scene checks

The generic `minimal` variant requires a “plain” background, while `rainy_commute` explicitly scores city buildings. Other variants require stage spotlights, a camera/tripod, or an appliance. A foreground silhouette against a pale field can satisfy these, but a literal plain-background judge can reject it. Replace with “Use a pale, otherwise uncluttered background, keeping every requested scene prop visible” and score that definition rather than a blank/plain background. This preserves the intended simplification without rewarding omission of scene cues.

### 4. Bicycle color binding is ambiguous

`bicycle_tuneup` says “a blue bicycle tire valve,” while its check demands a blue bicycle *frame*. Change the prompt to “the tire valve of a bicycle with a blue frame.” This fixes all five confirmation variants without changing difficulty.

## Nonblocking clarifications worth making before freeze

- `snowman_building` intentionally puts the carrot in the middle snowball rather than the conventional head/top ball. This is physically possible, but the visual result can look like a failed snowman. Prefer the top/head ball unless unusual-location compliance is a deliberate benchmark objective; if deliberate, say “middle body snowball, not the head.”
- `chalk_artist` combines blue chalk with a yellow sun. This can work as adding a blue outline, but explicitly requesting “tracing a blue outline around a partly drawn yellow sun” avoids asking the model to infer why the drawn mark has another color.
- In `small` variants, specify the main Bufo's head-to-foot bounding height, excluding held props, is less than one third of canvas height. Current “Bufo” is generally interpretable, but an umbrella or bow changes the apparent extent.
- The existing `skate_park-rail` exception correctly keeps feet and board visible. Its generator comment inaccurately says feet-action examples use a sign; update the comment to describe the actual background-rail exception.
- Full-body variants around tables, desks, or a dinghy are achievable with an oblique viewpoint and need not be removed. Clothing plus a two-hand prop is also achievable; the binocular eye occlusion is the distinct genuine contradiction.
- Generic observer variants are achievable and meaningful: exactly two characters, one passive and smaller on viewer-right. Their additional character-count challenge is explicit.

## Statistical limitation to preserve in reporting

The 80 confirmation prompts are 16 scene families with five correlated variants, not 80 independent scene families. Report paired uncertainty using a family-cluster bootstrap or otherwise account for family dependence. Likewise, the 24-row validation set contains only four families; a one-example winner is a fragile selection result. The five-point proximity goal is an experimental target, not something this sample size guarantees can be established precisely.

## Re-freeze procedure

Because no generation has occurred, apply the above prompt/requirement corrections now, regenerate once, and record the new hash and this review in the manifest or work ledger. Do not silently overwrite the reviewed hash after outputs exist. Recheck counts, uniqueness, disjointness, generator parity, and exact new hash before starting paid generation.
