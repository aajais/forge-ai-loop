# Augmented train prompts (Stage B)

- Generator: `scripts/augment_prompts.py --seed 20261025` (templates per family, no model calls)
- Provenance tag: `stage-b-augment-2026-10`
- Prompts added: 128 (8 per train family, ids `aug-train-<family>-<nn>`)
- sha256 of `config/distill_prompts.yaml` with the augmented block: `61b964bf06c0a58839600b860c59b4259099967d0d491e2710e3042377fa0264`
- Every prompt changes exactly one element of the family's base prompt.

| id | derived from | changed element | requirements |
|---|---|---|---|
| aug-train-concert_stage-01 | gap-train-concert_stage-base | colour | 4 |
| aug-train-concert_stage-02 | gap-train-concert_stage-base | colour | 4 |
| aug-train-concert_stage-03 | gap-train-concert_stage-base | count | 4 |
| aug-train-concert_stage-04 | gap-train-concert_stage-base | sign | 5 |
| aug-train-concert_stage-05 | gap-train-concert_stage-base | pose | 5 |
| aug-train-concert_stage-06 | gap-train-concert_stage-base | setting | 5 |
| aug-train-concert_stage-07 | gap-train-concert_stage-base | setting | 5 |
| aug-train-concert_stage-08 | gap-train-concert_stage-base | sign | 5 |
| aug-train-f1_presentation-01 | gap-train-f1_presentation-base | colour | 3 |
| aug-train-f1_presentation-02 | gap-train-f1_presentation-base | colour | 3 |
| aug-train-f1_presentation-03 | gap-train-f1_presentation-base | count | 3 |
| aug-train-f1_presentation-04 | gap-train-f1_presentation-base | sign | 4 |
| aug-train-f1_presentation-05 | gap-train-f1_presentation-base | pose | 4 |
| aug-train-f1_presentation-06 | gap-train-f1_presentation-base | setting | 4 |
| aug-train-f1_presentation-07 | gap-train-f1_presentation-base | setting | 4 |
| aug-train-f1_presentation-08 | gap-train-f1_presentation-base | sign | 4 |
| aug-train-gpu_garden-01 | gap-train-gpu_garden-base | colour | 3 |
| aug-train-gpu_garden-02 | gap-train-gpu_garden-base | colour | 3 |
| aug-train-gpu_garden-03 | gap-train-gpu_garden-base | count | 3 |
| aug-train-gpu_garden-04 | gap-train-gpu_garden-base | sign | 4 |
| aug-train-gpu_garden-05 | gap-train-gpu_garden-base | pose | 4 |
| aug-train-gpu_garden-06 | gap-train-gpu_garden-base | setting | 4 |
| aug-train-gpu_garden-07 | gap-train-gpu_garden-base | setting | 4 |
| aug-train-gpu_garden-08 | gap-train-gpu_garden-base | sign | 4 |
| aug-train-office_coffee-01 | gap-train-office_coffee-base | colour | 3 |
| aug-train-office_coffee-02 | gap-train-office_coffee-base | colour | 3 |
| aug-train-office_coffee-03 | gap-train-office_coffee-base | count | 4 |
| aug-train-office_coffee-04 | gap-train-office_coffee-base | sign | 4 |
| aug-train-office_coffee-05 | gap-train-office_coffee-base | pose | 4 |
| aug-train-office_coffee-06 | gap-train-office_coffee-base | setting | 4 |
| aug-train-office_coffee-07 | gap-train-office_coffee-base | setting | 4 |
| aug-train-office_coffee-08 | gap-train-office_coffee-base | sign | 4 |
| aug-train-skate_park-01 | gap-train-skate_park-base | colour | 3 |
| aug-train-skate_park-02 | gap-train-skate_park-base | colour | 3 |
| aug-train-skate_park-03 | gap-train-skate_park-base | count | 4 |
| aug-train-skate_park-04 | gap-train-skate_park-base | sign | 4 |
| aug-train-skate_park-05 | gap-train-skate_park-base | pose | 4 |
| aug-train-skate_park-06 | gap-train-skate_park-base | setting | 4 |
| aug-train-skate_park-07 | gap-train-skate_park-base | setting | 4 |
| aug-train-skate_park-08 | gap-train-skate_park-base | sign | 4 |
| aug-train-pizza_kitchen-01 | gap-train-pizza_kitchen-base | colour | 4 |
| aug-train-pizza_kitchen-02 | gap-train-pizza_kitchen-base | colour | 4 |
| aug-train-pizza_kitchen-03 | gap-train-pizza_kitchen-base | count | 4 |
| aug-train-pizza_kitchen-04 | gap-train-pizza_kitchen-base | sign | 4 |
| aug-train-pizza_kitchen-05 | gap-train-pizza_kitchen-base | pose | 4 |
| aug-train-pizza_kitchen-06 | gap-train-pizza_kitchen-base | setting | 4 |
| aug-train-pizza_kitchen-07 | gap-train-pizza_kitchen-base | setting | 4 |
| aug-train-pizza_kitchen-08 | gap-train-pizza_kitchen-base | sign | 4 |
| aug-train-basketball_court-01 | gap-train-basketball_court-base | colour | 3 |
| aug-train-basketball_court-02 | gap-train-basketball_court-base | colour | 3 |
| aug-train-basketball_court-03 | gap-train-basketball_court-base | count | 4 |
| aug-train-basketball_court-04 | gap-train-basketball_court-base | sign | 4 |
| aug-train-basketball_court-05 | gap-train-basketball_court-base | pose | 4 |
| aug-train-basketball_court-06 | gap-train-basketball_court-base | setting | 4 |
| aug-train-basketball_court-07 | gap-train-basketball_court-base | setting | 4 |
| aug-train-basketball_court-08 | gap-train-basketball_court-base | sign | 4 |
| aug-train-workshop_repair-01 | gap-train-workshop_repair-base | colour | 3 |
| aug-train-workshop_repair-02 | gap-train-workshop_repair-base | colour | 3 |
| aug-train-workshop_repair-03 | gap-train-workshop_repair-base | count | 4 |
| aug-train-workshop_repair-04 | gap-train-workshop_repair-base | sign | 4 |
| aug-train-workshop_repair-05 | gap-train-workshop_repair-base | pose | 4 |
| aug-train-workshop_repair-06 | gap-train-workshop_repair-base | setting | 4 |
| aug-train-workshop_repair-07 | gap-train-workshop_repair-base | setting | 4 |
| aug-train-workshop_repair-08 | gap-train-workshop_repair-base | sign | 4 |
| aug-train-rainy_commute-01 | gap-train-rainy_commute-base | colour | 3 |
| aug-train-rainy_commute-02 | gap-train-rainy_commute-base | colour | 3 |
| aug-train-rainy_commute-03 | gap-train-rainy_commute-base | count | 4 |
| aug-train-rainy_commute-04 | gap-train-rainy_commute-base | sign | 4 |
| aug-train-rainy_commute-05 | gap-train-rainy_commute-base | pose | 4 |
| aug-train-rainy_commute-06 | gap-train-rainy_commute-base | setting | 4 |
| aug-train-rainy_commute-07 | gap-train-rainy_commute-base | setting | 4 |
| aug-train-rainy_commute-08 | gap-train-rainy_commute-base | sign | 4 |
| aug-train-video_call-01 | gap-train-video_call-base | sign | 3 |
| aug-train-video_call-02 | gap-train-video_call-base | sign | 3 |
| aug-train-video_call-03 | gap-train-video_call-base | colour | 4 |
| aug-train-video_call-04 | gap-train-video_call-base | count | 4 |
| aug-train-video_call-05 | gap-train-video_call-base | pose | 3 |
| aug-train-video_call-06 | gap-train-video_call-base | setting | 4 |
| aug-train-video_call-07 | gap-train-video_call-base | setting | 4 |
| aug-train-video_call-08 | gap-train-video_call-base | colour | 3 |
| aug-train-film_set-01 | gap-train-film_set-base | colour | 3 |
| aug-train-film_set-02 | gap-train-film_set-base | colour | 3 |
| aug-train-film_set-03 | gap-train-film_set-base | count | 4 |
| aug-train-film_set-04 | gap-train-film_set-base | sign | 4 |
| aug-train-film_set-05 | gap-train-film_set-base | pose | 4 |
| aug-train-film_set-06 | gap-train-film_set-base | setting | 4 |
| aug-train-film_set-07 | gap-train-film_set-base | setting | 4 |
| aug-train-film_set-08 | gap-train-film_set-base | sign | 4 |
| aug-train-ping_pong-01 | gap-train-ping_pong-base | colour | 3 |
| aug-train-ping_pong-02 | gap-train-ping_pong-base | colour | 3 |
| aug-train-ping_pong-03 | gap-train-ping_pong-base | count | 4 |
| aug-train-ping_pong-04 | gap-train-ping_pong-base | sign | 4 |
| aug-train-ping_pong-05 | gap-train-ping_pong-base | pose | 4 |
| aug-train-ping_pong-06 | gap-train-ping_pong-base | setting | 4 |
| aug-train-ping_pong-07 | gap-train-ping_pong-base | setting | 4 |
| aug-train-ping_pong-08 | gap-train-ping_pong-base | sign | 4 |
| aug-train-archery_range-01 | gap-train-archery_range-base | colour | 4 |
| aug-train-archery_range-02 | gap-train-archery_range-base | colour | 4 |
| aug-train-archery_range-03 | gap-train-archery_range-base | count | 4 |
| aug-train-archery_range-04 | gap-train-archery_range-base | sign | 4 |
| aug-train-archery_range-05 | gap-train-archery_range-base | pose | 4 |
| aug-train-archery_range-06 | gap-train-archery_range-base | setting | 4 |
| aug-train-archery_range-07 | gap-train-archery_range-base | setting | 4 |
| aug-train-archery_range-08 | gap-train-archery_range-base | sign | 4 |
| aug-train-laundry_room-01 | gap-train-laundry_room-base | colour | 3 |
| aug-train-laundry_room-02 | gap-train-laundry_room-base | colour | 3 |
| aug-train-laundry_room-03 | gap-train-laundry_room-base | count | 4 |
| aug-train-laundry_room-04 | gap-train-laundry_room-base | sign | 4 |
| aug-train-laundry_room-05 | gap-train-laundry_room-base | pose | 4 |
| aug-train-laundry_room-06 | gap-train-laundry_room-base | setting | 4 |
| aug-train-laundry_room-07 | gap-train-laundry_room-base | setting | 4 |
| aug-train-laundry_room-08 | gap-train-laundry_room-base | sign | 4 |
| aug-train-bakery-01 | gap-train-bakery-base | colour | 4 |
| aug-train-bakery-02 | gap-train-bakery-base | colour | 4 |
| aug-train-bakery-03 | gap-train-bakery-base | count | 4 |
| aug-train-bakery-04 | gap-train-bakery-base | sign | 4 |
| aug-train-bakery-05 | gap-train-bakery-base | pose | 4 |
| aug-train-bakery-06 | gap-train-bakery-base | setting | 4 |
| aug-train-bakery-07 | gap-train-bakery-base | setting | 4 |
| aug-train-bakery-08 | gap-train-bakery-base | sign | 4 |
| aug-train-tennis_court-01 | gap-train-tennis_court-base | colour | 4 |
| aug-train-tennis_court-02 | gap-train-tennis_court-base | colour | 4 |
| aug-train-tennis_court-03 | gap-train-tennis_court-base | count | 4 |
| aug-train-tennis_court-04 | gap-train-tennis_court-base | sign | 4 |
| aug-train-tennis_court-05 | gap-train-tennis_court-base | pose | 4 |
| aug-train-tennis_court-06 | gap-train-tennis_court-base | setting | 4 |
| aug-train-tennis_court-07 | gap-train-tennis_court-base | setting | 4 |
| aug-train-tennis_court-08 | gap-train-tennis_court-base | sign | 4 |

## Fix 1: validation overlap in aug-train-gpu_garden-07

- The first version of `aug-train-gpu_garden-07` carried the requirement "A yellow sun is visible in the sky", which appears verbatim in two validation prompts (`gap-validation-florist_arranging-weather`, `gap-validation-ice_cream_cart-weather`). Validation is the selection set, so no augmented requirement may copy a validation or heldout requirement string.
- Reworded in the generator to "The sun shines above the garden" (scene: "under a bright sun"). The augmented block was removed from the YAML and regenerated with the same seed (20261025); the original 256 prompts are byte-identical and only this one prompt differs. The sha256 above is the post-fix file.
- Its Sol reference was regenerated alone: `runs/distill/stage-b/sol-aug-ref-fix1` (1 generation, 0 errors), manifest `runs/references/distill-aug-fix1`.
- `runs/references/distill-v2` was rebuilt (the first, unconsumed build was removed) with `merge_reference_manifests([distill-v1, distill-aug-v1, distill-aug-fix1], ..., replace=True)`: 304 rows, the `aug-train-gpu_garden-07` row taken from the fix run (recorded as `replaced_prompt_ids` on that source). `references.json` sha256: `9d0359ba9bcdaeaf232041479e05c9f94fba101d94c0ce1f49c11d7acd1507c9` (the removed first build was `50d4bd7e...`).
- Guards: `tests/test_augment_prompts.py` now checks augmented requirements against both validation and heldout requirements; `load_benchmark` validates every row carrying `provenance` or `augmented_from` (id must be `aug-train-<family>-NN` with its own family, provenance pinned to `stage-b-augment-2026-10`).
