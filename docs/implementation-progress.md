# Implementation ledger
Plan: /Users/aayush.jaiswal/.claude/plans/pasted-content-id-fc96-project-coreweav-abstract-spindle.md

Approved scope: Forge loop implementation and a simple local baseline-agent UI.

## Assistant reply media and composition — 2026-10-13

- Moved PNG publication inside rendering, before the final model reply. Added a
  model adapter that instruments the actual final LLM request and attaches the PNG
  to its assistant output, alongside the text. It preserves token/cache/reasoning
  usage and avoids a duplicate Agents SDK model span. No synthetic model call is
  emitted just to display media; the root assistant output carries the same image.
- Live Gemma and Sol runs both produced a remote Weave chat containing
  `render_svg`, then `render_png`, then exactly one assistant message with a PNG
  `content_refs` entry and no subsequent tool. Evidence:
  `runs/studio-reply-media/verification.json` and per-model trace-chat JSON.
- Diagnosed the user's F1 Arcade SVG: it overlays `#bufo-character` on top of
  `#bufo-fullbody`. Removed the legacy bust option from new generation guidance
  (old files still render). An initial prompt-only rerun removed the duplicate,
  but cropped the body and misplaced props; retained that failure in
  `runs/studio-presenter-v6/` rather than claiming it improved overall quality.
- Added and visually rendered a worked presenter example with shared local
  coordinates for the figure, shirt, arms and pointer, plus explicit canvas bounds
  and separate screen/presenter layout. Prompt v6 identifies these changes. Static
  asset cache revalidation/versioning also fixes the stale unstyled Copy name UI.
- The example-backed Gemma rerun fixed the disconnected limb/clothing layout but
  still clipped the feet (`runs/studio-presenter-example-v6/`). Added a measured
  transform/bounds check to the rendering tool and one extra agent correction pass
  for remaining clipping. It reports `composition_warnings` without invalidating
  the SVG or withholding an image; intentional close-up portraits are exempt.
- Final live Gemma run on the user's F1 Arcade prompt used two render attempts,
  corrected its framing, and returned a single connected presenter with visible
  feet, attached olive arms and no remaining bounds warnings. Inspected the PNG.
  Evidence: `runs/studio-presenter-checked-v6/verification.json`, image
  `1a2e06a4189a45408bed4363633230fb.png`, and `open_base-trace-chat.json`.
  The final assistant message contains its PNG content reference, and all tools
  precede it. This fixes the observed geometry failures, not every aesthetic or
  prompt-alignment failure; previous unsuccessful generations remain available.
- Verification: 134 Python tests and three Node controller tests passed; final
  image-placement regression checks span order, output parts, media references,
  actual usage and absence of the generated PNG from model input messages.

## Web context and Slack names — 2026-10-12

- Added a shared `search_web` function tool to Gemma and Sol. It uses OpenAI
  Responses hosted web search (`gpt-6.1-sol` by default), returns a sourced brief,
  and limits research to two 60-second calls per drawing. Missing credentials,
  disabled search and provider failures return a recoverable tool result. Public
  source URLs are validated; the isolated renderer has no new network access.
- Added agent-proposed `emoji_name` to `render_svg`, normalized to a portable
  `:bufo-...:` shortcode, persisted with each generation and shown with Copy name.
  Suggested names do not register custom Slack emojis. Research briefs, citations,
  model, response IDs and separate usage are saved in `web_searches`. The UI shows
  expandable research with clickable sources, including after history reload.
- Prompt version v5 records this tool change. `BUFO_WEB_SEARCH=false` disables
  external research for controlled evals; use the same research configuration
  across A/B arms. Character definitions still come from the shared system/tool
  asset, with no automatic reference PNGs in the user input.
- Live concurrent studio requests used official concert context and produced
  Gemma `:bufo-mr-worldwide:` (24,350-byte PNG, 52.475s) and Sol
  `:bufo-miami-to-moscone:` (54,486-byte PNG, 71.825s). Download and history APIs
  passed. Both outputs were visually inspected; Sol depicts the concert more
  completely, while Gemma's scene remains simpler. This is functional verification,
  not evidence of a calibrated quality improvement.
- Remote Weave Agents chat verified `search_web` → `render_svg` → `render_png`
  with PNG content references for both runs. Evidence lives in
  `runs/studio-web-v5/verification.json`, per-backend API/trace JSON and PNG files.
- Verification: 131 Python tests, three Node controller tests, Ruff lint/format
  and JavaScript syntax checks passed. Clipboard handler behavior is mocked in
  the controller test; browser visual/clipboard testing remains unavailable.

## Connected character, PNG tool media and concurrent chats — 2026-10-12

- Diagnosed the neck seam as a cropped 128x128 bust placed on a separately drawn
  body. Added `#bufo-fullbody`, a connected 128x176 asset with torso and feet, and
  system instructions for shared transforms, arm attachment and clothing placement.
  Preserved the legacy bust for portraits and old drawings. This improves the
  shared application asset, not the trained model. Rendered and visually inspected
  the connected asset and a live Sol concert scene.
- Moved character guidance and the SVG example into the system prompt. Generation
  no longer adds built-in PNG references or synthetic example turns to user inputs.
  Explicit screenshot attachments remain multimodal user inputs. Prompt version v4
  separates these runs from earlier benchmark results. Evaluation reference images
  are unchanged.
- Added an actual `render_png` execution after the assistant text: SVG input,
  cached renderer output, uploaded PNG Content object and `weave.content_refs` media
  attachment. Corrected a live-discovered SDK method mismatch, with a regression
  test. The final remote Agents chat response contains `render_png` immediately
  after the assistant message, with a PNG content reference and no error statuses.
  Evidence: `runs/studio-media-v4/verification.json`, `trace-chat.json`, and
  `8cc86446ea6b4cacac1b7033f2231856.png`. Earlier failed media attempts remain in traces.
- Replaced the global UI/API busy state with per-conversation guards and pending
  request tracking. Navigation/new chat stays usable during generation. Results
  arriving out of order stay in their own conversations; failed background requests
  preserve drafts. The sandbox's existing command lock keeps render files isolated.
- Live Sol and Gemma requests overlapped and both produced PNGs without built-in
  reference-image inputs. Evidence: `runs/studio-parallel-v4/verification.json`.
- Verification: 121 Python tests, two Node chat-controller concurrency tests, Ruff
  and JavaScript syntax checks passed. Remote Weave chat data verified PNG media;
  in-browser visual interaction remains unverified because no browser tool is exposed.

## Screenshot input — 2026-10-09

- Added clipboard image paste and a file attachment button, removable preview,
  image-only messages, per-conversation attachment drafts, and retry preservation.
  Sent images appear in user messages and persist across reloads and follow-ups.
- PNG/JPEG/WebP uploads accept up to 5 MB and 20 million pixels. The server decodes,
  strips metadata and normalizes them to PNG at a maximum 1600-pixel side. Current
  and historical attachments become multimodal model inputs; images also appear
  in root-turn telemetry. Generated manifests remain compatible with older rows.
- Live testing caught and corrected a Weave `start_turn` argument mismatch: its
  string-only convenience argument cannot accept a Message. Multimodal input now
  uses the turn's message list, covered by a regression check.
- A real Gemma request with an attached scene produced a red-hat Bufo, one render
  call and a 13,586-byte PNG; the normalized input was served back as a 22,536-byte
  PNG. Evidence: `runs/studio-image-input/verification.json`.
- 118 tests passed, including screenshot decoding/limits, agent image input,
  persistence/reload, follow-up context and telemetry. Ruff, JavaScript syntax,
  and attachment DOM-binding checks passed. Browser clipboard interaction remains
  visually unverified because of the existing browser-tool limitation.

## Model selection — 2026-10-09

- Added a chat-header model selector for the configured Gemma, Sol and tuned Gemma
  backends. Availability checks include the model ID, provider key, shared W&B key
  and required packages. Unconfigured options are disabled; API requests validate
  backend names and configuration before generating.
- Switching models preserves conversation history and parent links. Each assistant
  message identifies its backend; requests omitting a backend inherit the preceding
  message's model, or default to Gemma for a new conversation.
- Live studio API verification switched a Gemma conversation to `gpt-6.1-sol` and
  produced a valid 22,617-byte PNG with one render call and a plain-text reply.
  Evidence: `runs/studio-text-reply/model-switch-verification.json`.
- 108 tests passed, including model routing, context preservation and unavailable
  models; Ruff and JavaScript syntax checks passed. UI asset/DOM binding checks
  passed. The earlier browser visual-verification limitation remains.

## Chat and agent-turn verification — 2026-10-09

- Replaced the single-drawing studio with a conversation sidebar, message transcript,
  rendered images per assistant message, downloads, and a bottom composer. Follow-ups
  retain all earlier requests and the latest editable scene; parent links persist in
  the manifest. Enter sends; Shift + Enter inserts a newline.
- SVG is generated through `render_svg` tool arguments. The assistant finishes with
  a short plain-text reply, and API responses omit SVG source. Saved SVG downloads
  remain self-contained. Prompt version is now `bufo-v3-*`.
- Used Weave's Conversation/Turn API to create one independent `invoke_agent` root
  per exchange, preserving its conversation ID while keeping SDK model/tool spans
  beneath it. The HTTP span no longer becomes the agent turn's root. Historical
  traces are unchanged. Tool failures now expose a redacted, actionable error.
- Live follow-up testing caught Gemma copying the embedded canonical character paths,
  exhausting its output budget and producing truncated tool JSON. Added a regression
  test and compacted historical SVG to the editable scene plus character reference.
  Round-trip tests verify that rendering restores the same canonical geometry.
- Live generation and follow-up used real Gemma and cwsandbox through the studio API.
  The successful follow-up returned “Here you go—Bufo is ready to party!”, one render
  call and a 19,224-byte PNG. Remote Weave queries verified eight spans under one
  parentless `invoke_agent Bufo open_base` root, with no error spans in that turn.
  Evidence: `runs/studio-text-reply/chat-verification.json`, `followup-spans.json`,
  `followup-chat.json`, and `1c1660d7f10f4ac683de0109f3881188.png`.
- The earlier failed attempt remains recorded. The final image preserves Bufo but
  has a floating party hat: successful rendering is not a visual-quality pass.
- Verification: 106 tests passed; Ruff lint/changed-file formatting and JavaScript
  syntax checks passed. HTTP asset checks, conversation history and PNG serving
  passed. Browser visual/interaction verification remains unconfirmed: the in-app
  browser tool was unavailable and local Chrome was blocked from starting. Existing
  upstream Weave/Pydantic deprecation warnings remain.

## Latest verification — 2026-10-09

This entry supersedes earlier connectivity and verification limitations below.

- Updated the shared system prompt and included three original Bufo images plus
  an SVG example. Freehand copying of the original vector failed in a live run.
  Added a reserved `#bufo-character` fragment that embeds original-derived vector
  paths identically for every backend. This improves the application baseline;
  it is not evidence of learned model improvement. New poses remain limited.
- Ran four development probes through Gemma and real serverless CPU rendering.
  Three rendered; the fourth exposed a malformed final SVG after a tool call.
  Fixed the final repair path to cover tool-capable models as well. A subsequent
  expression run rendered successfully; the original failure remains recorded.
- Verified the studio API against live Gemma and cwsandbox: generated PNG was
  served back through `/api/files`, 19,258 bytes, valid PNG signature. Evidence:
  `runs/studio/live-verification.json`, generation `82bd06d64cb74effa733fde707aa90a1`.
- Queried the native Agents conversation API: seven stored spans including
  `invoke_agent`, two `chat` spans and `execute_tool`. Evidence:
  `runs/studio/agent-spans-verification.json`.
- Verified remote Weave image deserialization (PNG plus three references) and
  native scorer feedback. Added `eval-identity` and `review-queue` CLI stages.
  Corrected Dataset/Evaluation object-name collisions found during the live test.
- Published the `BufoIdentity` v2 evaluation of the four development probes.
  Results include the invalid-render row. The checklist catches the wrong-side,
  two-eye rack failure. It still overestimates some size/body/prop constraints;
  human calibration is required before using this as an RL reward.
  Evidence: `runs/bufo-checklist-eval/identity-eval/{scored.jsonl,summary.json,evaluations.json}`.
- Created and read back 11 items in the human annotation queue:
  https://wandb.ai/wandb-smle/bufo-gen/weave/annotation-queues/01a1212f-416c-7e89-b97c-4760ed9c69fa
  Updated ARIA handoff prompts for image-based review and actual dataset curation.
- Configured the reference arm as `gpt-6.1-sol` using OpenAI Responses. After the
  user added the key, ran Sol and Gemma on all four probes with the same prompt
  version (`bufo-v2-ea5f39347344`). All eight rendered successfully. Sol's actual
  Weave image and native agent/chat/tool spans were read back. Evidence:
  `runs/sol-gemma-probe/manifest.jsonl` and `sol-agent-spans.json`. The review queue
  now contains 19 items, including all eight new outputs.
- Published both paired Weave Evaluations; exported the eight images with trace
  links in `runs/sol-gemma-probe/comparison.html`. Automatic quality flags: Sol
  0/4, Gemma 2/4; mean identity estimates 0.765/0.6725 and checklist alignment
  1.0/0.75. The judge missed visible Gemma watering and expression failures;
  these are provisional results on development prompts, not calibrated accuracy.
- Researched the official CoreWeave Model Distillation product: managed proxy,
  curation, Serverless SFT, candidate evaluations and routing. It is distinct from
  the existing local SFT scaffolding. Internal Slack docs remain inaccessible
  without a connected search connector. No training or routing changes launched.
- Final checks: 102 tests passed; Ruff lint and changed-file format checks passed.
  Weave/Pydantic dependency deprecation warnings remain.

## Latest verification — 2026-10-08

This update supersedes the initial network limitations and pending baseline checks recorded below.

- Updated and locked the latest PyPI releases: `cwsandbox[wandb]==1.18.0`, `weave==0.53.11`, and `wandb==0.30.0`. Retained aligned OpenTelemetry 1.44.0 pins for Weave's exporter compatibility. `uv sync --locked` and the installed dependency compatibility check passed.
- CPU rendering now uses native `cwsandbox.AuthStrategy.WANDB` and serverless placement. Renderer assets are uploaded after the container starts, removing the large create-time file payload. Creation retries reuse the sandbox instance; failed jobs clean up and report available startup diagnostics.
- Live renderer smoke passed with a valid 512×512 PNG. A full baseline run on the latest dependencies also passed: Gemma generated a bufo tending a GPU garden, invoked the render tool once, and saved valid SVG and PNG files in 55.359 seconds. The remote sandbox completed and stopped.
- Evidence: `runs/e2e-latest/manifest.jsonl`, `runs/e2e-latest/c1a7c3e0338444aeb71206d95cd76a3a.png`, and `runs/e2e-latest/weave-spans.json`. The Weave API returned seven stored spans, including agent invocation, two model calls, and tool execution, for conversation `open_base:p01:01a294a5a5de`.
- Final checks: 77 tests passed; Ruff lint and formatting passed. The live run emitted a nonfatal upstream OpenTelemetry warning about a dictionary tool-result attribute; conversation spans were nevertheless stored. Unit tests emitted ten upstream Weave/Pydantic deprecation warnings.
- Live requests worked through the approved `.venv/bin/python -c` command. No DNS allowlist change was needed. GPU training, RL, evaluation, Registry publication, and browser visual verification remain outside this completed baseline check.

## Initial implementation record

### Original-emoji RL dataset follow-up — 2026-10-08

- Built `data/emoji-rl-v1` from original static PNGs in the bufo.fun catalog: 128 usable tasks, with 102 train / 26 held-out. Captions came from Qwen3.8-27B vision calls, not emoji-name guesses. Three spot-checked captions were corrected with original model annotations retained in the caption cache.
- Grouped dHash neighbors before splitting; verified every image decodes, source checksums match, split IDs/hashes/groups do not overlap, and minimum cross-split dHash distance is five. This does not guarantee no semantic overlap.
- Published and re-downloaded `wandb-smle/bufo-gen/bufo-emoji-rl-dataset:v0`. The project artifact is available for ART. Optional organization Registry linking failed because W&B could not resolve an organization for `wandb-smle`; project-artifact publishing now supports omitting that link.
- RL rollouts now pass their example's original PNG to the reward judge. Training rejects held-out rows. The student receives only text and generates its own SVG; these are RL tasks without SFT target answers.
- Live held-out check passed after correcting sandbox output transport: render results now use the file API because the stdout stream truncated a PNG response. Evidence: `runs/emoji-rl-check/manifest.jsonl` and its generated PNG. The VLM used exactly one original target reference.
- The live judge's 0.9125 score is visibly generous for the untrained cartoon. Reward calibration remains necessary before a larger RL run; no CLIP pass, optimizer step, checkpoint, or training improvement is claimed. Captions are machine annotations with a review gallery, not a fully human-reviewed gold dataset.

- Ruling: Work in the supplied empty project directory; there is no git repository or branch to isolate.
- Ruling: Existing plan is the approved design. UI adds a local FastAPI studio, sharing the same agent and renderer as CLI; no separate hosted application.
- Ruling: Never invent a tuned checkpoint, scores, training results, or Agent Lens evidence. Unconfigured stages fail with actionable instructions.
- Preflight: generation manifest is shared by run/eval/curate/report; renderer is shared by agent tools/final render/RL; all checkpoints use the same Registry contract.

Tasks:
1. SDK verification, baseline agent, isolated render, manifest and CLI.
2. Studio UI and local baseline smoke test.
3. Scorers, evaluation, backup dataset, ARIA/Registry curation.
4. SFT GPU sandbox, ART serverless RL, comparative reports.
5. Tests, live checks where reachable, final review and documentation.

Implementation and verification:
- Task 1: implemented baseline agent, warm isolated renderer, model configs, manifests and demo CLI. Contract tests passed.
- Task 2: implemented local FastAPI studio with refinement, PNG preview, SVG downloads, history and error states. HTTP tests passed in-process.
- Task 3: implemented shared scorers, Weave evaluation replay, public-reference backup builder, ARIA prompts, curation and Registry lineage.
- Task 4: implemented GPU SFT entrypoint, ART serverless RL, checkpoint promotion and held-out per-theme comparison.
- Task 5: fresh reviewer checked application and cached SDK contracts. Identified and fixed malformed dimension handling and Gemma vision-layer LoRA targeting; both had failing regression tests before fixes.
- Local suite: 49 tests passed, including a real resvg render of a fixed trusted fixture. 11 dependency deprecation warnings (FastAPI/Starlette and the locally available Weave version).
- Final verification: `.venv/bin/pytest -q` → 49 passed; Ruff check and format check passed; `node --check src/bufogen/static/app.js` passed. Editable package build/install and both console-entrypoint help commands passed using cached build dependencies.

Rulings and external limitations:
- Ruling: GPU SFT needs CWSANDBOX_API_KEY; W&B's CPU sandbox wrapper explicitly rejects GPU resources. Added separate credentials rather than assuming WANDB_API_KEY can allocate a GPU. Cost if wrong: deployments with unified future auth may need the adapter updated.
- Ruling: Use resvg rather than CairoSVG; it ships as a self-contained wheel, mounted before a deny-egress sandbox starts. Cost: first generation needs host access to PyPI unless a prebuilt image is configured.
- Ruling: RL is base→RL, separate from base→SFT; arbitrary external SFT adapter initialization was not verified through ART. Cost: the combined SFT+RL experiment remains a later live integration task.
- Ruling: Without an available Slack connector, public references are explicitly labeled unverified against Slack; --names accepts a supplied export. Cost: the reference subset may differ from #rare-bufos.
- Ruling: Missing real checkpoints fail instead of displaying a baseline under a tuned label; unknown cost remains null. Cost: the default three-model demo waits for an actual tuned model.
- Environment blocked outbound package/API networking; uv also panicked in macOS system-configuration initialization. Cached packages enabled offline unit testing, but openai-agents and vtracer were not available for a complete installation.
- A live doctor request to the inference catalog returned Connection error. No model calls were claimed successful.
- Starting the local UI failed to bind 127.0.0.1:8765 with Operation not permitted. Browser automation tools were not exposed. HTTP behavior was tested in-process; screenshot/visual-browser verification remains outstanding.
- No configured GPU key, SFT image or tuned model. No ARIA task, Registry publication, GPU training, RL job or demo video was executed.
- Live checks still required: Agent SDK tool calling; Weave Agents grouping/deep links; sandbox execution/cleanup; CLIP/VLM eval; Registry links; SFT and RL smoke jobs; serving both adapters; held-out improvement evidence; backup video.

Final review fixes and additional checks:
- Fixed malformed SVG dimensions: seven regression cases failed first, then passed after strict finite positive dimension validation.
- Fixed Gemma LoRA selection: regression test failed first, then passed with fully qualified language-attention targets that exclude vision/audio wrappers.
- Added bundled, licensed DejaVu Sans for deterministic text rendering inside the minimal deny-egress sandbox; tested font mounting and real PNG output.
- Added unmatched Slack-name retention and teacher generation for prompt-only backup examples; intersection/retention test failed first, then passed.
- Linked the source dataset directly to ART's actual training run, in addition to checkpoint-promotion lineage.
- No review findings deferred. External live verification remains outstanding as listed above.

## Stage 0: frozen experiment runtimes (2026-10-21)

- `scripts/restore_experiment_runtime.py` rebuilds one run as a project tree. It copies today's scaffolding, then overlays the run's `frozen/` snapshot and re-checks every `source_hashes` path. `RUNTIME.json` lists all files that came from today's tree, the package versions, and any reconstruction. Tests: `tests/test_restore_runtime.py`.
- Eight runtimes have been restored under `runs/restored/`, each with its own venv. Recorded package versions match; the only listed difference is the openpipe-art git pin against PyPI 0.5.20.
- All 8 pass the gate. Detail per target is in `runs/restored/<slug>/VERIFICATION.md`: the experiment tests pass from inside each tree, and each run re-scores from cached scores with byte-identical `scored.jsonl`, an empty summary diff, blank credentials and no paid calls. No evidence under `runs/forge-*` was modified (mtime check).
- Limitations:
  - The contact-pilot `fit_compare` check cannot run until the in-flight SFT candidate (`61b53c05-…`) returns; rerun it then.
  - forge-gap and iteration2 tests need their run evidence supplied by copy (gap) or by a read-only symlink (iteration2 stores absolute paths).
  - forge-loop uses one reconstructed `config/models.yaml`. It was derived from its `benchmark.json` arms plus env-var names for `base_url_env`. It is reproducible for scoring through the frozen benchmark code, but cannot regenerate through today's CLI.

## Stage 1: per-call instrumentation (2026-10-21)

- `src/bufogen/recording_model.py` records the finish reason, tokens, reasoning tokens and reasoning text length, served model and time for every model call. Each row gets `model_calls` and `truncated_calls`.
- `BUFO_DISABLE_THINKING` and `BufoAgent(disable_thinking=...)` are opt-in and default to false. When on, CoreWeave requests carry `chat_template_kwargs.enable_thinking=false`; OpenAI and Forge requests never do. `agent.py` and `models.py` hashes changed, so new benchmark dirs record new `source_hashes`. Frozen runs reproduce from `runs/restored/`.
- Precedence: the `--disable-thinking` CLI flag and `benchmark` runs override the env var (the benchmark never reads `BUFO_DISABLE_THINKING`; it always freezes an explicit true/false in its config). `run`, `demo` and the studio honour `BUFO_DISABLE_THINKING` when the flag is absent.
- Failed generations keep their per-call records: benchmark failure rows carry the `model_calls` recorded before the error, plus `truncated_calls`.

## Stage A: Astra judge, Weave evaluations and calibration runs — 2026-10-22

Direction changed on 2026-10-21: the hybrid image path is dropped (serverless inference has no
image generation); the goal is Sol-level strict pass with the existing SVG agent through
distillation into a Forge fine-tunable model. Design:
`docs/superpowers/specs/2026-10-21-svg-distillation-design.md`. Stage A plan:
`docs/superpowers/plans/2026-10-21-astra-judge-and-calibration.md`.

Shipped: `scorers/astra_judge.py` (`AstraReferenceJudge`, `bufo-astra-v1`, `gpt-6-astra`,
two-image strict-schema verdict with a Sol reference), `scorers/efficiency.py`,
`references.py` (reference manifests from a Sol run), `statistics.py` (paired family bootstrap),
`benchmark-score --judge astra` with Weave `Evaluation` publication and full metadata,
`--stage build-references`, `review_packet.py` (self-contained blinded HTML packet and
agreement gate), `config/distill_prompts.yaml` (validation enlarged 24 → 48 before any
training), and four `static/app.js` fixes (`response.ok` before parsing, `failedThreads`
hoist, independent status/history loads, model picker retry).

Results (train+validation only, heldout untouched): Sol run 2 judged against Sol run 1 strict
pass 174/176; self-equal 24/24 `equal`; base Gemma 1/48 and iter2-e3 4/48 on validation, both
about 90 pp below Sol with paired intervals. Review packet v1 (assistant blind agreement 0.925,
confusion 17/1/2/20, gate not passed) is set aside because its gate-breaking item was disclosed
before the user labelled; a first resample drew disclosed rows again and is also set aside; the
user labels packet v3 (seed 20261023, disclosed generations excluded, key outside the packet
directory, spec gate on requirements marked missing); an advisory blind pass on v3 agrees 0.95
with no judge-pass/reviewer-missing item, so calibration waits on the user's labels. The user
can label either the HTML packet or a Weave annotation queue (`bufo review packet v3`, built by
`scripts/weave_review_queue.py` from neutral per-item calls; `pull` turns the annotations into
the same decisions file; verified end to end on a one-item smoke queue, since deleted).
**User gate result (2026-10-22): not passed.** Agreement 0.95 (38/40), but four judge-passed
drawings were marked by the user as missing a requirement; three are implied or occluded
contact or state the judge accepted (eyes closed, stem entering vase, hand gripping cone). The
judge is not calibrated. Rubric revisions v2–v5 were developed against the labelled packet
(visibility rule with a per-check `directly_visible` flag, closed-eye definition, grip
definition that accepts mitt hands) and the judge gained k-sample conservative merging after a
stability probe showed single verdicts on borderline contact flipping; v5 with k=3 passes the
development gate. Full re-judge under v5 k=3: Sol 154/176 (validation 39/48), students 0/48
and 0/47, self-equal 24/24 equal; paired student − Sol −0.81 [−0.875, −0.75]. Confirmation
packet v4 (40 unseen validation drawings, Weave queue): agreement 0.875, no judge-pass /
user-missing item after the user re-read one craft note, **gate passed; `bufo-astra-v5` k=3 is
the calibrated judge** (`runs/distill/calibration-v1/calibration-record.json`). Stage A closed.
Full write-up with run
ids, Weave links, invalid publications and the two mid-run fixes:
`docs/distill/calibration-v1.md`.

## Stage 1 latency experiment — 2026-10-21

Three arms on the same 12 train prompts with `open_base` (served model `google/gemma-4-26B-A4B-it`) and `--concurrency 4`. Only thinking and `max_tokens` varied. All three `benchmark.json` files share `prompt_hash` `7a6d099b…`. Their configurations differ only in `policy.disable_thinking` (False/True/True), `policy.max_tokens` and `arms[0].sampling.max_tokens` (6000/6000/3000). Each arm was scored with `benchmark-score` without `--publish`.

Run dirs: `runs/latency-a-baseline/`, `runs/latency-b-nothink/`, `runs/latency-c-nothink-3000/`.

| arm | n | gen failures | rows truncated | length / calls | median latency s | p90 latency s | median out tok/call | median reasoning tok | median reasoning chars | median call latency s | strict passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| latency-a-baseline | 12 | 0 | 0 | 0/51 | 31.9 | 37.3 | 108 | not reported | 0 | 2.8 | 0/12 |
| latency-b-nothink | 12 | 0 | 0 | 0/57 | 33.7 | 48.7 | 108 | not reported | 0 | 2.5 | 0/12 |
| latency-c-nothink-3000 | 12 | 0 | 0 | 0/54 | 29.0 | 49.6 | 115.5 | not reported | 0.0 | 2.7 | 0/12 |

Note: the "median reasoning chars" column measures only a separate `reasoning_content`/`reasoning` response field, which this endpoint never sends. Gemma's thinking would appear inline as `<|channel>thought` blocks in the content, and that is measured in the "Extra evidence" bullet below.

Wall-clock generation times (UTC):
- a: 18:47:41–18:50:00 (2 min 19 s)
- b: 18:50:22–18:53:04 (2 min 42 s)
- c: 18:53:09–18:55:54 (2 min 45 s)

The arms ran one after another at different times, so endpoint load is a confound.

Extra evidence:
- The largest output from any single call was 1354 tokens in a, 1286 in b and 1187 in c.
- The endpoint returns `reasoning_tokens` as null and no `reasoning_content`.
- Gemma puts its thought channel inline in the assistant content. Every one of these blocks (58 in a, 65 in b, 59 in c) is the empty `<|channel>thought\n<channel|>`, including in the baseline. The counts span whole trajectories, including seeded prior turns, and every block is empty either way.

**Conclusion: inconclusive.** The baseline did not reproduce the truncation:
- Arm a had 0/51 `length` finishes, a maximum of 1354 output tokens per call, and a median of 31.9 s per row. Historically, 129/295 rows hit ≥5990 output tokens, and capped rows had a median of 93 s against 23 s uncapped. That baseline was measured across all `runs/**/manifest.jsonl` Gemma (`open_base`) rows that existed before 2026-10-21, mostly `runs/forge-loop/heldout-baselines`, the forge-loop pilot, and the forge-gap validation and confirmation runs (source: the spec's review findings, `/Users/aayush.jaiswal/.claude/plans/look-atthis-project-i-noble-scroll.md`). The forge-iteration2 runs already showed 0 capped rows, which is consistent with today's result.
- The thought channel was empty in every arm, so the `--disable-thinking` contrast had nothing to remove.
- Median latency did not move meaningfully (31.9 / 33.7 / 29.0 s).
- The historical truncation therefore cannot be attributed to thinking, or ruled out as thinking, from this data. It may depend on the endpoint or model state at the time, or on prompts and turns that are not among these 12.

**(c) vs (b) pass rate:** halving `max_tokens` to 3000 did not change the pass rate: 0/12 strict passes in both, and in a. Every row failed the judge with `prompt_mismatch`, sometimes together with low composition or readability. All 12 renders in every arm were valid.

Caveats:
- 12 prompts is a small sample. These pass counts are not statistically meaningful.
- The scene judge is backend `closed_quality_bar`, model `gpt-6.1-sol`, which is the teacher model. That biases the absolute pass counts.

Proposed follow-up default changes. None was made here, and each needs its own measurement:
1. Thinking: decide whether to default `disable_thinking` to on for CoreWeave Gemma. First reproduce a run with a non-empty thought channel or a `length` finish, because today's endpoint shows neither.
2. `max_tokens`: decide whether to lower the 6000 default. Measure the effect on pass rate and on `length` finishes across a larger prompt set, ideally the historical prompts that truncated.
3. Outer retry loop: decide whether to remove or shorten the outer retry loop (`attempts: 2`). Measure its latency cost against its pass-rate contribution separately.

### Follow-up: dependency-snapshot timestamp fix (2026-10-21)

`scripts/restore_experiment_runtime.py` picked the wandb `requirements.txt` that predates each run by reading wandb's directory names as UTC. wandb names those directories in local time (`run-20261015_120401` started at 16:04:01Z). The script now reads `startedAt` from `files/wandb-metadata.json` and falls back to the directory name read as local time. Seven of the eight restored runtimes had selected a snapshot taken after their run started, and their `RUNTIME.json` `wandb_requirements` were recomputed. The package conclusion is unchanged everywhere: the only difference is `openpipe-art`, installed from its pinned git commit, against wandb's `0.5.20` label. The gate results did not depend on this field.
