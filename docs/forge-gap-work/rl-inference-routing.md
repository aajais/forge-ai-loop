# RL inference gateway diagnosis and correction

October 15, 2026.

## Observed failure

`runs/bufo-gemma-dynamo-rl-20261015` produced one genuine, scored agent rollout.
Its three exchanges passed exact Dynamo evidence validation and full upstream
ART tokenization. The next rollout failed all three bounded inference attempts
with HTTP 524 and zero captured exchanges. No optimizer call was reached, no
training-job receipt exists, and no new checkpoint was produced.

The root process retained the response: the affected zone was
`api.training.wandb.ai`, proxy read timeout 120 seconds, ray
`a4b0f18a8f37303a`, timestamp `2026-10-15T18:41:00Z`, retry-after 120 seconds.
Cloudflare documents 524 as an origin/proxy timeout, distinct from an SDK
tokenization failure: https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/.

## Source-level cause

Pinned ART's `ServerlessBackend._prepare_backend_for_training` returns its
training API base URL (`art/serverless/backend.py:217`). `TrainableModel.register`
assigns that URL as its inference URL (`art/model.py:1652`). Therefore the newly
managed client sent inference through the training gateway rather than the
project's already verified `https://api.inference.wandb.ai/v1` serving endpoint.

The narrow correction keeps the registered model, exact checkpoint request,
credentials, and managed ART metadata wrapper. It adds
`base_url=base_config.base_url` to the existing `model.openai_client().with_options`
call. Training still uses its original training service. The actual inference
base URL is now recorded in `rl-training.json`.

## Verification

One authorized tiny inference probe began at `18:43:00.441328Z`, after the stated
120-second retry window. It restored the saved registration identity locally
(no new remote registration), retained the managed client, and overrode only
the serving base URL. It requested the same immutable SFT v1 artifact and
received `google/gemma-4-26B-A4B-it` in 0.745 seconds. Genuine engine metadata
contained 17 prompt and three completion tokens; full ART tokenization passed.
No optimizer was submitted.

Evidence: `runs/forge-gap/rl-probe/direct-managed-routing/receipt.json` and its
hash-bound raw capture. This tiny probe establishes routing and capture
compatibility, not that every large rollout avoids timeout.

Regression coverage drives the actual managed ART client through the Agents
chat model and verifies the serving URL plus all Dynamo metadata flags survive
`with_options`. Trainer orchestration checks the configured serving URL and
zero transport retries. **49 focused RL tests passed.**

The first failed attempt remains unchanged. Any subsequent live proof must use
a fresh attempt directory; no automatic optimizer retry was added. No agent,
model-configuration, or benchmark-fingerprinted source was modified for this
routing correction.
