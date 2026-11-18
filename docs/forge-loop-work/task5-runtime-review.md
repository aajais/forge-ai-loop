# Task 5 runtime freeze review

Reviewer: `codex/review_loop_runtime`. Read-only production-code review, 2026-10-15. This approval covers only the frozen runtime subset in `task5-diff.txt`, not the remaining CLI, report, documentation, or live service behavior.

## Spec: PASS

- `forge_proxy` is an explicit opt-in backend. Existing direct Gemma and Sol settings and the UI's `open_base` default remain unchanged.
- Forge takes the `OpenAIChatCompletionsModel` path. Only provider `openai` selects Responses. SDK `ModelSettings.extra_body` forwards the literal `wandb.entity` and `wandb.thread_id` metadata keys. The application retains the same group ID through follow-ups and within-request repairs.
- Requested alias remains the configured model value. Successful HTTP responses supply the separately recorded `resolved_serving_models` list; the implementation does not infer an actual model from the alias.
- The final response wrapper passes the existing model settings through, so metadata remains attached to the real final model call. Tool-before-text ordering and the PNG appended to the final assistant response remain intact.
- The only UI runtime change is checking `WANDB_ENTITY` for Forge availability. Input image handling, per-conversation isolation, background completion, and existing model selection paths are preserved.
- `open_rl` is a separate, initially unconfigured backend. It does not overwrite the SFT backend or silently enable a trained model.
- `models.py` now participates in source hashing and therefore in the comparison fingerprint. Resolved backend settings remain part of each recorded benchmark configuration. The comparison fingerprint intentionally excludes arm identity and concurrency, while retaining the shared model transport implementation.

## Quality: PASS

No blocking correctness, maintainability, or regression findings in this runtime diff. The HTTP hook observes actual non-streaming responses without replacing model responses, tolerates non-JSON or unexpected response content, and uses the existing client lifecycle. Direct providers retain their previous transport headers and behavior.

Relevant verification completed locally without remote calls:

```text
.venv/bin/pytest -q tests/test_forge_cli.py::test_proxy_http_metadata_resolved_model_and_thread tests/test_forge_cli.py::test_fingerprint_covers_model_transport_source tests/test_ui.py tests/test_reply_model.py tests/test_image_input.py tests/test_telemetry.py tests/test_benchmark.py
34 passed in 3.74s

node --test tests/chat_ui.test.cjs
3 passed
```

Python emitted existing Weave/Pydantic deprecation warnings. The proxy test exercises the real SDK's Chat Completions request construction through an HTTP mock, verifies metadata across two requests, and observes the actual response model. This is not evidence of a live Forge agent run or a rendered browser check; those remain root integration work.

## Approved file hashes (SHA-256)

```json
{
  "src/bufogen/agent.py": "905b130146348a64d4cb0923485dc71ab53b0d28883c821e8c9a653d7ac65e16",
  "src/bufogen/models.py": "8a4b61aecc2135c4d4669e559846c0157bd536e14fc8ab68a0df98f10d1a4978",
  "src/bufogen/ui.py": "76fc5501c05cbf7821929bce695334d393a36dc19d364f405d7cbd96b0c11a28",
  "src/bufogen/benchmark.py": "5165fea78d2e037045a395a871940cfa5b239ac6b87e34ec94a9ccde9e3f0fb8",
  "config/models.yaml": "3815ec813d32862b5928e4fe5f77b357f3bf9ed229dd421462cdaa03931f4a1b"
}
```

The runtime source freeze is approved for the heldout benchmark. Remaining Task 5 work must preserve these bytes or return relevant changes for review before the benchmark starts.
