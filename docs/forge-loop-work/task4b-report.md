# Task 4b report: explicit scene judge backend and grounded rubric

Implemented only in `src/bufogen/scorers/scene_quality.py`, `tests/test_scene_quality.py`, and `.env.example`.

- `SCENE_JUDGE_BACKEND` selects `open_base` (default, preserved Qwen/CoreWeave behavior) or `closed_quality_bar` (configured Sol/OpenAI backend). `SCENE_JUDGE_MODEL` optionally overrides the scene-only model. Explicit constructor model takes precedence; otherwise open_base uses JUDGE_MODEL, closed_quality_bar uses its configured SOL_MODEL.
- Local invalid backend, provider, URL, and empty/whitespace model configuration fail before remote requests. Arbitrary served-model availability still requires the provider; transport/model errors remain abstentions, not zero scores or passes.
- Responses request matches the actual probe: exact instructions, `Return JSON. REFERENCE Bufo identity`, reference and candidate `input_image`, prompt plus exact requirements in candidate text, JSON object format, 4000 output tokens, omitted temperature. It intentionally omits prior_requests just as the probe does. Existing CoreWeave content and 3200/6400 bounded JSON retry remain unchanged.
- Configurations include selected backend, actual provider/base URL/model/API mode, response format, thinking-disable behavior, temperature and explicit omission flag, applicable token budget, rubric hash/version and uncalibrated status. Backend/model/rubric changes invalidate saved score reuse and comparison fingerprints.
- Root approved exact grounded `bufo-scene-v2` rubric after all 32 grounded probes. Production string is byte-identical to `runs/forge-loop/judge-probe-grounded/rubric.txt` (SHA256 `ee0b672fd60f526cdbb293013cecd1966f50d878f8c8a263c436a12a8b00d0b6`). General contact/endpoint/relative-position grounding is added without image-specific coordinates. Existing v1 evidence is not relabeled.
- Exact requirement matching, raw valid JSON judgment retention, sanitized errors, null unavailable scores, and missing/prior-turn handling remain unchanged. No population calibration claim; new runtime still says uncalibrated. Root owns new calibration evidence from the completed probe.

Validation:

- New tests demonstrated four failures before implementation (missing selection/configuration/request paths), then passed.
- `pytest -q tests/test_scene_quality.py tests/test_benchmark.py tests/test_rl_contract.py tests/test_rl_dataset.py tests/test_learning.py`: **66 passed**.
- Ruff check and formatting passed for both modified Python files.
- Separate runtime byte-equality assertion verified exact grounded probe rubric.
- Tests cover both fake API request shapes, explicit/environment model precedence, invalid local configuration before requests, backend cache/fingerprint invalidation, and malformed JSON/transport abstentions with secret sanitization. Backend default tests clear relevant local environment to remain stable after root opts into Sol.

No live calls, provider writes, dependency edits, environment changes, or subagents were performed. Independent review is next before root changes local backend selection or uses this for live RL.
