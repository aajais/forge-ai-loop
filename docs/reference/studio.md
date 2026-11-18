# Bufo studio and traces

Reference for the chat application (`bufo-studio`), its runtime policy, the model selector, and
the trace layout it emits.

## Running

```bash
uv run bufo-studio --reload        # http://127.0.0.1:8765, restarts on Python edits
uv run bufo-studio --port 9000
```

Refresh the browser after static HTML/CSS/JS edits. `/api/status` reports the effective runtime
policy. Requires `.env` with `WANDB_API_KEY` (and `OPENAI_API_KEY` for the teacher).

## What the chat supports

- Iterative drawing in a conversation; concurrent conversations; switching model per message.
- Pasted screenshots and PNG/JPEG/WebP attachments up to 5 MB as explicit image input.
- PNG and SVG downloads, Slack-style emoji names.
- **Compare all 4 models** (or `/static/compare.html`): one prompt to every available demo model
  concurrently, with independent timings, errors, checkpoint details and trace links. Failed cards
  offer **Retry this model**. The last 12 comparisons are kept in browser local storage. Keep the
  page open while requests run; after a reload, unfinished requests are marked interrupted and
  are never resubmitted automatically (check chat history for results that finished server side).

## Model selector

| Display label | Role | Backend key | Evaluated checkpoint |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` | Base | `qwen_base` | — |
| `ft-Qwen/Qwen3.8-27B` | SFT, 1 epoch | `sft_q3` | `bufo-stageb-sft-q3-20261023:v1` |
| `rl-Qwen/Qwen3.8-27B` | SFT + RL | `rl_q3` | `bufo-stageb-sft-q3-20261023:v6` |
| `gpt-6.1-sol` | Teacher | `closed_quality_bar` | — |

Backends are defined in `config/models.yaml`; environment variables named there override model
IDs (for example `RL_Q3_MODEL`). Fine-tuned labels derive from the configured base model and
must match the adapter's actual base. The immutable `:vN` artifact version is shown separately so
SFT and SFT+RL checkpoints stay distinguishable. Availability reflects configuration, not a live
health check. Other backends (Gemma, Nemotron, `forge_proxy`) remain configured for CLI work.

## Runtime policy

The studio always enables reasoning, including on retries: Qwen and its adapters send
`enable_thinking: true`; Sol uses `reasoning.effort: medium`. Studio requests omit output-token
limits and share a maximum of 100 agent turns across the initial attempt and automatic repair,
within a 30-minute deadline. The studio ignores `BUFO_DISABLE_THINKING`.

CLI and benchmark defaults are separate: Qwen arms use 20,000 output tokens, thinking enabled,
and a 900-second request timeout unless overridden. Controlled benchmarks freeze resolved settings
per arm and ignore later environment drift. Never migrate an existing frozen run; use a new
directory when the runtime changes.

## Agent and tools

All drawing arms share the same system instructions, the canonical connected Bufo SVG assets
(`assets/bufo_style/`) and worked examples. Tools:

- `plan_scene`: layout plan in Bufo-local coordinates.
- `render_svg`: validates the SVG (allowlist, bounds and transform checks) and renders a PNG in
  an isolated CoreWeave CPU sandbox via `cwsandbox`. The worker has no network or credentials and
  uses a pinned `resvg` wheel. A warm sandbox is reused and stopped on shutdown.
- `search_web` (optional, `BUFO_WEB_SEARCH`): bounded web context. Disabled per agent instance in
  controlled benchmarks and RL so comparisons stay reproducible.

Reference PNGs are never silently appended to prompts. Composition warnings (for example clipped
feet) are reported without withholding the image; intentional close-ups are exempt.

## Trace layout

Each user exchange is one root `invoke_agent` span in its own Weave trace, with `chat` and
`execute_tool` descendants sharing the conversation ID. Repairs stay inside the exchange;
follow-up messages start a new trace with the same conversation ID. This follows CoreWeave's
[Agents-view OTel contract](https://docs.coreweave.com/products/wandb/weave/guides/tracking/trace-agents-otel#shape-spans-for-the-agents-view).

Recorded on the trace: system prompt and tool definitions on the root, every model call with
inputs, tool requests, provider reasoning, token usage, served model, finish reason and errors.
The application `render_png` span nests inside the model-requested `render_svg` tool, and only
final text replies carry the output image. Agent version `bufo-native-v2` identifies this
instrumentation; traces ingested earlier keep their original shape. A warning that names an SDK
`Bufo ... turn N` span as the trace root means stale code: restart the studio.

Native custom feedback does not populate Agent Scores columns in the installed SDK; see
[telemetry notes](../forge-demo-work/telemetry-notes.md) for the source-only backfill command.

## Forge proxy routing

Backend `forge_proxy` sends Chat Completions to `https://proxy.training.coreweave.com/v1` with
alias `bufo-gen` and metadata keys `wandb.entity` and `wandb.thread_id` (stable across turns).
The requested alias and the returned serving model are recorded separately in provenance; the
application never changes routing itself. `gpt-6.1-sol` uses the Responses API directly and is
not routed through the proxy. See [metadata and traces](https://docs.coreweave.com/model-distillation/proxy/metadata-and-traces)
and [model routing](https://docs.coreweave.com/model-distillation/proxy/model-routing).
Evidence from the proxy smoke: `runs/forge-loop/proxy-agent-smoke/message-order-check.json`.
