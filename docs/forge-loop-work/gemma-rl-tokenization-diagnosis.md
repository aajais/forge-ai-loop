# Gemma 4 / ART tokenization diagnosis — 2026-10-15

The saved four-rollout group cannot currently be trained through this ART/serverless path without resolving a tokenizer/capture compatibility dependency. Local ART 0.5.20 reproduces the exact remote error on two original captures and related assistant-boundary errors on the other two. A complete, genuine two-exchange probe with ART's inference defaults and provider-returned exact output IDs still fails. No production fix or further training retry is justified by these results.

## Evidence

All results, hashes, and private probe captures are in `runs/forge-loop/rl-tokenization-diagnosis/`. No original rollout request, response, sampled content, tool result, or reward was changed. No model weights were downloaded. Runtime dependencies were not changed.

| Original scored capture | Official template result | Diagnostic template preserving raw content |
| --- | --- | --- |
| `step-1-rollout-0-attempt-1.json` | `Could not locate a history message in the rendered history` | Pass |
| `step-1-rollout-1-attempt-1.json` | `Could not prove a sampled history message boundary with this tokenizer` | `Cannot map assistant message 4 into the rendered history` |
| `step-1-rollout-2-attempt-2.json` | `Could not locate a history message in the rendered history` | Pass |
| `step-1-rollout-3-attempt-1.json` | `Could not prove a sampled history message boundary with this tokenizer` | `Cannot map assistant message 4 into the rendered history` |

`results.json` records these eight checks and source capture SHA-256 hashes. Traceback locals are diagnostic snapshots: values from a previous loop iteration can remain in scope on early boundary errors; do not treat every `failing_text_prefix` as the current message. The two exact remote-error reproductions locate sampled assistant content at message index 4.

The original responses include literal Gemma channel delimiters in `message.content`, e.g. `<|channel>thought` followed by `<channel|>` and the final visible answer. Google's canonical template unconditionally runs assistant content through `strip_thinking`, which removes that prefix. ART's `_chat_message_parts` retains the original content; `_tokenize_chat_view` searches for it in the template-rendered token stream and fails at installed `_tokenize.py:5673`. Several intermediate messages also combine marker-only content with tool calls, exposing the additional boundary failures. The diagnostic template only disables those content-stripping expressions; its partial success isolates the mismatch and is **not a fix**.

## Genuine inference capability checks

The checks used immutable `wandb-artifact:///wandb-smle/bufo-gen/bufo-gemma-sol-pilot-20261015:v1`. These were explicitly authorized capability probes, not benchmark generations or training updates.

1. One reused training request, capped at 128 output tokens, requested `logprobs=True`, `return_token_ids=True`, and `return_tokens_as_token_ids=True`. The provider returned 128 exact output IDs in logprobs, recognized by ART, but no prompt IDs. Its tool arguments were truncated/invalid JSON, so it cannot validate complete-trajectory tokenization. See `exact-token-probe-{request,response,metadata,tokenization}.json`.
2. The complete probe requested a tiny SVG validation call, executed `xml.etree.ElementTree.fromstring` locally, returned the actual successful validation result, then requested the final answer. Both calls used the same exact-ID flags plus `chat_template_kwargs={"preserve_thinking": true}` and `logprobs=True`. ART automatically captured both exchanges. There were 39 exact output IDs for the valid tool call and 6 for the final answer; neither response supplied exact prompt IDs. The final content still included the channel delimiters. Tokenizing the untouched capture with the official template failed with `Could not prove the sampled content boundary in the rendered history`. ART also warned about prompt retokenization while preserving sampled IDs. See `two-exchange-probe.json` and `two-exchange-probe-tokenization.json`.

The private request/response/trajectory files have mode `0600`. `probe-gemma-exact-tokens.py` documents the exact two-call procedure and refuses to overwrite its existing capture. **Do not rerun it as part of local verification: it invokes paid inference.**

## Supported controls and remaining dependency

- CoreWeave documents `chat_template_kwargs.enable_thinking=false`, and lists this exact Gemma model as reasoning-disabled by default. ART also defaults `enable_thinking=False` when that template option exists. Explicitly disabling thinking is therefore not a demonstrated remedy, and cannot repair old captures. Reasoning is documented as a separate response field; actual delimiter-bearing content is the compatibility issue here.
- Installed ART `Model._default_chat_completion_extra_body` (`model.py:716`) supplies both exact-ID flags and `preserve_thinking=True` for trainable model clients. It does not turn on logprobs. On this provider the probe's usable output IDs arrived through logprobs, not `choice.token_ids`, so client injection alone is insufficient evidence of a fix.
- `_chat_choice_tokens` accepts prompt IDs from the choice or response and exact output IDs from the choice or numeric-ID logprob entries. `_tokenize_exact_projected_chat_history` requires exact final prompt and output IDs, then validates retained sampled spans; no prompt IDs were returned by either probe. Ordinary text logprobs without exact IDs do not supply this path.
- Exact output IDs alone still enter the renderer-dependent fallback, and the complete probe proves that fallback currently fails. A service-compatible solution must either preserve and expose enough **actual prompt and output token evidence** for ART to prove all sampled boundaries, or align the inference parser/template/ART boundary logic for Gemma channel markers and native tool formatting. Exact prompt IDs are a candidate contract, not a proven fix, until a complete genuine capture passes.
- A local tokenizer patch does not change remote ServerlessBackend tokenization. Any backend/SDK remedy must be verified on the real remote path before another optimizer update. Rewriting saved content, synthesizing token IDs, or transplanting logprobs from a new generation would violate sampled-action integrity.

## Local reproduction without changing runtime dependencies

The installed runtime is `openpipe-art==0.5.20`, `tokenizers==0.23.2`, `jinja2==3.1.6`, `openai==2.54.0`. Diagnosis used `transformers==5.19.0` and its safetensors wheel extracted only under `/private/tmp/bufo-transformers-diagnosis`. No PyTorch is needed for these tokenizer checks. Python is 3.12. Current temporary assets remain available in this workspace:

```sh
DIAG_TRANSFORMERS_PATH=/private/tmp/bufo-transformers-diagnosis DIAG_TOKENIZER_PATH=/private/tmp/bufo-gemma4-tokenizer .venv/bin/python docs/forge-loop-work/reproduce-gemma-tokenization.py
DIAG_TRANSFORMERS_PATH=/private/tmp/bufo-transformers-diagnosis DIAG_TOKENIZER_PATH=/private/tmp/bufo-gemma4-tokenizer .venv/bin/python docs/forge-loop-work/tokenize-gemma-probe.py
```

For another machine, install Transformers 5.19.0 into a separate target/virtual environment while retaining the listed ART/tokenizers versions. Download only `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and `config.json` from the exact `url` entries in `source-manifest.json`, to `DIAG_TOKENIZER_PATH`; verify every SHA-256 before running. Those URLs track Hugging Face `main`, so a hash mismatch is a changed source, not an equivalent reproduction. The manifest also hashes the actual installed ART sources. The remote service's own dependency/template versions are not exposed by this diagnosis; local reproduction does not prove they are byte-identical.

## Primary sources

- [Google canonical Gemma 4 chat template](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/chat_template.jinja): `strip_thinking`, assistant rendering, tool argument formatting, thinking options. Downloaded template SHA-256: `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`.
- [CoreWeave reasoning controls](https://docs.coreweave.com/products/inference/serverless/response-settings/reasoning.md): enable/disable request contract and Gemma default. A dated copy is `official-reasoning-doc.md` in the evidence directory.
- [CoreWeave Chat Completions API](https://docs.coreweave.com/products/inference/serverless/api-reference/chat-completions.md): endpoint/authentication and OpenAI-compatible contract. Extra exact-ID flags above come from ART's official SDK implementation, not a claim that this brief REST page specifies them.
- [ART tokenizer implementation](https://github.com/OpenPipe/ART/blob/main/src/art/trajectories/_tokenize.py), [ART model client](https://github.com/OpenPipe/ART/blob/main/src/art/model.py), [ART history projection](https://github.com/OpenPipe/ART/blob/main/src/art/trajectories/_history.py), [ART template defaults](https://github.com/OpenPipe/ART/blob/main/src/art/utils/chat_template.py). Implementation conclusions use the installed 0.5.20 files and hashes, rather than assuming GitHub `main` is that release. A `v0.5.20` raw URL returned 404.
