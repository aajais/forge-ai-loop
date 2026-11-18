# Forge Post-Training demo changes — 2026-10-26

Status: **partially applied and verified through Chrome on 2026-10-26**. A fresh browser tab recovered access. Qwen default temperature and all three routing overrides persisted after fresh page loads. The dataset and historical native-evaluation names were saved. The existing Forge demo report was updated and published. The earlier local DNS and connector failures did not prevent these browser changes. No new training job or smoke inference was started; no routing weight or price was changed.

## Approved changes and implementation details

1. **Qwen inference defaults.** On the [one-epoch model](https://forge.coreweave.com/post-train/distillation/e/wandb-smle/p/bufo-gen/models/da6bd91a-d7b8-478c-ad00-f6f540925537), change default temperature from 0 to 0.7. Save, reload, and verify persistence. This does not regenerate comparison outputs.
2. **Qwen routing parameters.** On [Routing](https://forge.coreweave.com/post-train/distillation/e/wandb-smle/p/bufo-gen/routing), inspect the current version and revision before editing. For the existing Qwen target set `temperature` to JSON `0.7`, `max_tokens` to JSON `20000`, and `chat_template_kwargs` to JSON `{"enable_thinking": true}`. Preserve every target, its weight, and 100% trace sampling. The audited weights were 10% gpt-6.1-sol, 45% the older Gemma adapter, and 45% the Qwen one-epoch adapter. Re-read current values before saving; another worker may have changed them. Wait for Applied, then reload and verify. Also verify the Qwen target resolves to the evaluated `:v1` artifact; do not assume an alias proves checkpoint identity. Client timeout remains an application setting (900 seconds), not a routing parameter.
3. **Dataset label.** Rename the shared [911-row dataset](https://forge.coreweave.com/post-train/distillation/e/wandb-smle/p/bufo-gen/datasets/280e22a6-ae6b-401e-aafe-cc43182eae49), if Settings permits, to `bufo-gpt-6.1-sol-reviewed-trajectories-911-20261023`. Preserve its ID, revision, rows, splits, and review provenance. The current j1 name incorrectly suggests the dataset belongs only to one student arm.
4. **Model display labels.** Use `ft-Qwen/Qwen3.8-27B` with epoch count and artifact version separately. q3 is **1 epoch / v1**; q1 is **3 epochs / v3**. Use `ft-google/gemma-4-26B-A4B-it` and `ft-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` for the corresponding student models. Do not rename artifact collections or serving IDs just to alter presentation. If Forge exposes no independent label field, retain immutable job names and use the identity table below in the demo report.
5. **Historical native eval.** Rename [image-check](https://forge.coreweave.com/post-train/distillation/e/wandb-smle/p/bufo-gen/evals/a300e4f8-5177-42ba-9a78-6cfb70fe4747), if Settings permits, to `Historical caption continuation - 4 examples`. It compares closing captions after a teacher render; it does not measure two independently generated PNGs. Keep the results as historical evidence. Use the calibrated Weave image comparison for quality claims.
6. **Pricing.** Leave unverified prices unset. The model page's 0.60 / 2.50 placeholders are not a price quote. No exact official price for the audited Qwen model was verified. The analytics total excludes unpriced requests; do not present it as full cost or calculate savings from it.
7. **Proxy investigation and smoke.** Inspect the actual errors behind the 4 failed requests in the 30-day analytics view before changing provider settings. The audit showed 17 total requests, 4 failures, 76% priced, and no Qwen requests; zero-token rows alone do not identify the cause. Run the smoke below only after verifying the intended route/checkpoint. Record request IDs, routing revision, resolved model, finish reasons, token usage, PNG validity, and conversation URLs. Keep smoke traffic explicitly labeled.

Routing parameters replace caller values. See [routing and versions](https://docs.coreweave.com/model-distillation/studio/routing-and-versions). Conversation IDs keep routing stable within a revision; see [metadata and traces](https://docs.coreweave.com/model-distillation/proxy/metadata-and-traces).

## Bounded application smoke

Run from the project root with the existing local `.env`. This uses the existing agent, renderer, and telemetry path. It is a real generation, not a prerecorded tool response. Substitute the version verified in Forge; do not infer “latest.” The following is prepared for version 1, and must be checked against the live version before execution.

```bash
FORGE_PROXY_MODEL=bufo-gen@v1 BUFO_INFERENCE_TIMEOUT_S=900 BUFO_DISABLE_THINKING=false \
  uv run bufogen --stage run --backends forge_proxy \
  --prompts docs/forge-demo-work/proxy-smoke-prompts.yaml \
  --limit 1 --max-turns 20 --max-tokens 20000 \
  --output runs/forge-demo/proxy-smoke-20261026
```

This tests the weighted router and may select any positive-weight target. It is **not** a guaranteed Qwen test. A separate direct Qwen-artifact smoke is needed to establish that candidate's proxy serving health; use Forge's current Direct artifact snippet and preserve project/conversation metadata. Do not change traffic weights to force selection. No smoke is considered passed until its manifest has a valid PNG and the Forge conversation/analytics entry is visible.

## Report update to publish after verification

Append the following section to the existing [Bufo Forge AI loop demo](https://forge.coreweave.com/wandb/wandb-smle/bufo-gen/reports/Bufo-Forge-AI-loop-demo--VmlldzoxODE3NjU5MQ). Update the status text with verified results, not intended changes.

### Post-Training: demonstrate lineage, then prove image quality

Open the 911-row dataset, inspect a complete teacher trajectory including `plan_scene` and `render_svg`, then follow Models to the one-epoch Qwen candidate. The final caption preview is only part of the training conversation: the recorded recipe uses **Assistant turns: All**. The dataset has 911 training rows and zero native test rows; image validation is performed separately in Weave.

| Display model | Recipe | Exact checkpoint |
| --- | --- | --- |
| Qwen/Qwen3.8-27B | Base | Qwen/Qwen3.8-27B |
| ft-Qwen/Qwen3.8-27B | SFT, 1 epoch, batch 2, LR 1e-4 | wandb-artifact:///wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v1 |
| ft-Qwen/Qwen3.8-27B | SFT, 3 epochs, batch 2, LR 1e-4 | wandb-artifact:///wandb-smle/bufo-gen/bufo-stageb-sft-q1-20261023:v3 |

Follow the candidate's training run to show the training history. Then open the [saved Weave image comparison](https://forge.coreweave.com/wandb/wandb-smle/bufo-gen/weave/compare-evaluations?view=comparison_2026-10-24_22-07-09-875). Inspect strict pass, requirement coverage, similarity, render validity, and latency, plus paired output PNGs. The frozen October 24 comparison has 17/48 base passes, 30/48 one-epoch SFT passes, and 27/48 three-epoch SFT passes. These are validation results, not a final heldout or human-confirmed outcome. The historical teacher result of 39/48 used different inference settings and does not establish matched-runtime parity.

The historical native `image-check` result evaluates caption continuation after an existing render. Do not use its percentages as evidence of image-generation improvement. The saved Weave comparison also uses separately published dataset objects: its alignment warning remains until a shared dataset evaluation is published. Explain that limitation rather than hiding it.

Close with Routing and Analytics only after the labeled smoke is verified. Explain conversation-sticky routing and show the actual resolved model/version from the smoke trace. Unpriced requests are excluded from cost totals, so cost savings are not yet a verified demo claim.

## Verification ledger

| Item | Current status |
| --- | --- |
| Remote Qwen defaults and routing | Verified after reload: default temperature 0.7; v1 Qwen overrides temperature 0.7, max_tokens 20000, chat_template_kwargs.enable_thinking true; weights 10/45/45 and sampling 100% preserved |
| Dataset/model/native-eval display names | Dataset and native eval renamed. Model detail exposes no separate display-name editor; exact model IDs and epoch/checkpoint mapping are in the report. Artifact and serving IDs preserved |
| Published report update | Published October 26 with dataset, image-evaluation, checkpoint and routing links; caveats and remaining gaps retained |
| Proxy failure cause | Unknown; trace query failed |
| New smoke generation and trace | Not run. Refreshed Bufo UI has direct model choices but does not expose forge_proxy; ordinary direct inference would not validate the proxy |
| Exact model prices | Unverified; left unset |
| Training or routing-weight changes | None |
