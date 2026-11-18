# Gemma Serverless RL compatibility: verified remedy

Research and live capability verification: October 15, 2026.

## Finding

The previous failure is **not evidence that Gemma cannot use Serverless RL**. The installed PyPI `openpipe-art==0.5.20` predates the necessary Dynamo response support. Its version remains the latest on PyPI, so another `uv sync --upgrade` alone would not fix this. The official ART notebooks now explicitly describe this limitation and pin a Git revision instead. See the [official notebook README](https://github.com/OpenPipe/art-notebooks) and [dependency source pin](https://github.com/OpenPipe/art-notebooks/blob/main/pyproject.toml).

The concrete fix is [merged ART PR #934](https://github.com/OpenPipe/ART/pull/934), commit **`45a17cbd2a85cc0333172e652fc1211c6ae3f3da`**. Dynamo supplies exact training evidence through `nvext.engine_data`. Request it with:

```python
logprobs=True,
extra_body={"nvext": {"extra_fields": ["engine_data"]}},
```

The upstream implementation attaches actual engine prompt IDs, completion IDs, and completion log probabilities to ART's training representation. It retains the original visible response and its text logprobs. The updated managed model client requests this extension automatically; a directly constructed OpenAI client must request it. This is an inference metadata fix, not a change to rewards or sampled actions.

## Actual same-model proof

Root authorized exactly two paid inference calls, with no model registration, optimizer update, or runtime dependency modification. Both used the existing immutable SFT artifact:

`wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v1`

The response identifies base `google/gemma-4-26B-A4B-it`. The first call generated a tiny valid SVG validation tool call. Python actually parsed that SVG, and its successful result became the second call's tool result. The second call produced the final reply. Each request was capped at 128 output tokens. The existing request flags `return_token_ids`, `return_tokens_as_token_ids`, and `preserve_thinking` were retained; `nvext` was the material addition.

| Exchange | Actual prompt IDs | Actual completion IDs | Matching logprobs | Finished |
| --- | ---: | ---: | ---: | --- |
| Tool call | 103 | 39 | 39 | true |
| Final reply | 107 | 6 | 6 | true |

The untouched capture **tokenizes successfully** using the isolated official commit above, the same downloaded canonical Gemma tokenizer, and `multi_history=True`. It produces two histories, honestly preserving the provider's rerendered prompts; the second prompt is not assumed to extend the first. No token IDs, logprobs, text, arguments, or tool outcomes were synthesized or transplanted.

Evidence: `runs/forge-gap/rl-probe/metadata.json`, `two-exchange-probe.json`, `exchange-{0,1}-{request,response}.json`, and `two-exchange-probe-tokenization.json`. Raw captures have mode `0600`; metadata records SHA-256 hashes and exact model identities. The previous failed captures remain unchanged.

## Implementation recommendation

1. Pin `openpipe-art` to the verified commit above through `tool.uv.sources`. Its project version is still `0.5.20`; version checks alone cannot distinguish the working code from PyPI. Record and verify the source revision as well as package version. Do not install GPU/backend extras.
2. Use the registered model's `model.openai_client().with_options(default_headers={"OpenAI-Project": project_path()}, max_retries=0, timeout=180)` and inject it into `BufoAgent(client=...)`. ART's proxy preserves its default metadata options through `with_options`.
3. Add an RL-only agent request option mapping to `ModelSettings.extra_args={"logprobs": True}`. The installed Agents SDK explicitly supports this field. The managed ART client requests engine metadata but does not itself enable logprobs.
4. Before expensive scoring or optimizer submission, validate genuine prompt/completion token evidence and finite aligned completion logprobs for every captured call. Preserve raw captures and their exact serving/checkpoint identity.
5. Collect a **new** genuine scored rollout group. Old captures lacking engine prompt IDs cannot be repaired by a later inference call. Submit only one coordinated update after normal continuation/provenance checks and persist the actual remote training job ID.
6. A real remote optimizer update and immutable checkpoint are still required to establish complete service compatibility. The capability probe proves inference and local preprocessing, not remote gradient training or quality improvement.

## Compatibility checks

Compared installed PyPI metadata against the selected commit: **no required dependency additions or removals**. `src/art/serverless/backend.py` is byte-identical, retaining the exact job submission/event interface used by the existing durable receipt guard. Receipt: `runs/forge-gap/rl-probe/dependency-compatibility.json`.

Against the isolated upstream source, the project's RL contract, recovery, job-receipt, and dataset suites passed: **36 tests**. The upstream `test_dynamo_tokens.py` could not collect because this CPU project environment lacks PyTorch; no dependency was installed to hide that limitation. The genuine two-exchange tokenizer check itself ran successfully without PyTorch.

Source was downloaded beneath `/private/tmp/bufo-rl-research/art-dynamo/`, not installed into `.venv`. Local-only reproduction:

```sh
DIAG_ART_SOURCE_PATH=/private/tmp/bufo-rl-research/art-dynamo/OpenPipe-ART-45a17cb/src \
DIAG_TRANSFORMERS_PATH=/private/tmp/bufo-transformers-diagnosis \
DIAG_TOKENIZER_PATH=/private/tmp/bufo-gemma4-tokenizer \
.venv/bin/python docs/forge-gap-work/verify-gemma-dynamo.py
```

`probe-gemma-dynamo.py` makes paid calls and already has a saved result; do not rerun it as a test.

## Alternatives considered

[CoreWeave's October 5 engineering blog](https://coreweave.com/blog/closing-the-loop-between-inference-and-post-training) distinguishes managed Model Distillation, ART-based Serverless RL, a programmable training API in limited preview, and dedicated RL Rollouts with training metadata. The programmable API and separate GPU rollout deployment are not necessary to resolve this demonstrated client metadata gap. No evidence gathered here establishes that a Tinker switch could import this precise existing CoreWeave SFT storage identity. Retaining ART and the verified SFT continuation is the narrower supported path.
