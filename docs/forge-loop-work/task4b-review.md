# Task 4b independent review

## Spec: PASS

Reviewed the task brief and its grounded-rubric amendment, implementation report, exact diff, current scorer, shared vision helper, backend configuration, benchmark fingerprint/cache consumers, and the real probe script and persisted responses. No blocking spec findings.

- The production `RUBRIC` bytes exactly equal the preserved grounded probe rubric. Independently computed SHA256: `ee0b672fd60f526cdbb293013cecd1966f50d878f8c8a263c436a12a8b00d0b6`. Runtime labels the intentional change `bufo-scene-v2` and remains `uncalibrated`.
- Backend selection defaults to `open_base`; scene-specific/explicit model overrides have the documented precedence. Backend/provider/URL/model validation occurs before requests. Provider availability and arbitrary remote model compatibility remain remote concerns and correctly produce abstentions.
- The Responses request matches the actual successful probe: selected OpenAI model/client, exact instructions, reference then candidate image, prompt plus exact requirements, JSON-object output, 4000-token limit, and omitted temperature. Omitting prior requests matches the reviewed probe; requirements remain the full supplied checklist.
- The default CoreWeave path retains its existing content, temperature 0, 3200/6400 retry budgets, and Qwen thinking-disable option. Configuration reports the corresponding actual API mode/provider/URL/model/budgets/options.
- Backend/model/rubric changes alter scorer configuration and comparison fingerprints. Old embedded scores are recomputed; explicitly supplied incompatible scores fail validation. Exact checklist validation, raw valid JSON retention, sanitized errors, null unavailable metrics, and prior-turn/missing-image behavior are preserved.

Independently joined the 32 stored grounded responses to generation IDs and the frozen assistant reviews and revalidated each raw judgment against its exact requirements. Result: **28/32 agreement, 0 false positives, 4 false negatives**. Recomputed decisions agree with the stored decisions. Disagreements are Sol rainy commute, Sol concert, and the F1 edit from both models. These are observations on an assistant-reviewed development sample, not population calibration or an independent test of teacher superiority. Old v1 result labels must not be silently represented as v2 runtime results; the root's planned derived evidence should retain original probe provenance and explicitly apply the now-versioned parser/configuration to the preserved raw response.

## Quality: PASS

No blocking correctness or maintainability findings in the owned production diff. The backend branch is small, shared validation/error behavior remains centralized, and tests cover both request shapes, override resolution, malformed local configuration, cache invalidation, and transport/shape failures.

Independent validation: `.venv/bin/pytest -q tests/test_scene_quality.py tests/test_benchmark.py tests/test_rl_contract.py tests/test_rl_dataset.py tests/test_learning.py` → **66 passed**, 2364 existing Weave/Pydantic deprecation warnings, 3.92 seconds. No remote calls or source/dependency edits were made during review.

Operational note outside the production diff: the original temporary probe imported `RUBRIC` and appended the grounding paragraph, which would duplicate it on a later rerun. Root corrected `/private/tmp/bufo_judge_grounded_probe.py` to read the frozen rubric instead; that correction was verified during review. The existing preserved rubric and response files are consistent with the implementation reviewed here.

Root's subsequent `runs/forge-loop/calibration/rl-calibration.json` records the matching v2/Sol configuration, agreement 0.875, `rl_ready: true` under the explicit limited-pilot readiness policy, and `population_calibrated: false`. Its agreement matches this independent reconstruction. This does not erase remaining false negatives or the teacher/judge correlation.

Approval covers Task 4b implementation and selection of this explicitly disclosed scorer for subsequent evidence generation. It does not assert population calibration, Sol parity, or completion of the broader Forge loop.
