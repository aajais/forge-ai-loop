# Source feedback and evaluation publication

Implemented against installed Weave 0.53.11 on 2026-10-24. No paid generation or scoring was performed and no frozen run evidence was changed.

## Source identities

New generation rows preserve `native_trace_id`, `native_span_id`, and `execution_metadata` with backward-compatible defaults. The agent captures the public OpenTelemetry span context at the Conversation turn boundary, before any child model or renderer spans. Classic `weave_call_id` remains the independent `visual_generation` call identity. Native span IDs must not be passed to `client.get_call()`.

The application-created `render_png` span now carries `bufo.tool_origin=application`, `bufo.model_requested=false`, and a description explaining its relationship to `render_svg`. The model-facing frozen prompt/tool description is unchanged by this telemetry change.

## Saved feedback

`_publish_source_feedback` attaches the exact saved Astra verdict to an existing classic visual-generation call through `Call.apply_scorer(..., {"saved_score": score})`, when that call ID exists. It uses a deterministic `SavedGenerationScore` scorer named `SavedAstraReferenceJudge`, validates source generation identity, and has no inference path. This distinction matters because original generation calls predate reference selection and cannot satisfy the original reference judge's inference-cache check. The raw saved payload retains original scorer and reference provenance. Scene publication retains its existing classic scorer flow and adds native-source feedback.

For native roots, Weave provides `client.get_agent_span_feedback(span_id).add(custom_type, payload, creator)`. The code uses `bufo.saved_astra_judge` or `bufo.saved_scene_quality`, including the exact score, its digest, generation ID and native trace/span IDs. It does not synthesize scores or convert judge errors into successful verdicts.

**Native Agent Scores columns are not populated by this custom feedback API.** The SDK reserves typed scorer ratings for `wandb.agent_monitor`, `wandb.agent_user_feedback`, or `wandb.runnable` feedback. Agent monitor feedback requires real runnable, scoring-call and trigger references. Ordinary saved verdicts do not establish those monitor identities; none were fabricated. The receipt therefore explicitly records `score_columns=unsupported_custom_feedback`.

The publication ledger has independent native/classic receipts for each generation and scorer. It distinguishes published feedback, missing historical root/call IDs, unsupported SDKs and retryable errors. Success is checkpointed per target. Remote feedback is reconciled before adding, covering the successful-remote-write/lost-local-receipt retry window. Changed saved scores have a different digest and receive a new receipt without deleting historical feedback.

Relevant installed SDK implementation: `weave/trace/weave_client.py` (`get_agent_span_feedback`), `weave/trace/feedback.py` (`RefFeedbackQuery.add`), `weave/trace/call.py` (`Call.apply_scorer`), and `weave/trace_server/feedback.py` (typed feedback validation).

## N/A investigation

The SDK distinguishes Evaluation object `name` from `evaluation_name`; only the latter controls the readable evaluation call name. Publication now sets both.

An offline test exercises the real installed SDK Evaluation path through `_publish_arm_evaluation`, including coverage validation and aggregation. One saved true verdict and one saved false verdict aggregate to `strict_pass.true_count=1` and `strict_pass.true_fraction=0.5`. Existing unmodified Stage-B publication receipts also contain populated Astra summaries: `sft_q1` has 27/48 and `sft_q3` has 2/48 in their saved sweep receipts. Thus this investigation did **not** reproduce an aggregate publication bug. A live UI showing N/A still needs verification of its selected field, evaluation identity and ingestion state; it must not be reported as fixed based only on these offline tests.

Pinned Forge LoRA checkpoint metadata now includes a `https://forge.coreweave.com/wandb/{entity}/{project}/artifacts/lora/{name}/{vN}` link, including the required `/wandb/` prefix observed in the live Forge UI. The `lora` collection type matches the repository's `Registry.link_checkpoint` and `train_rl` artifact access. Mutable aliases produce no purported immutable link.

## Publish source evidence after credentials/connectivity recover

The following command publishes only existing source feedback. It does not run generation, judge inference, Evaluation.evaluate, or modify the frozen scored JSONL. The output ledger is separate and safe to retry. Set `WANDB_ENTITY` and `WANDB_PROJECT` to the original source project, and use a scored file from that same project. Historical rows lacking native root IDs remain explicitly missing; this command does not guess roots from nearby conversations.

```bash
.venv/bin/python - runs/distill/stage-b/sweep/sft_q3/astra-v5-k3/scored.jsonl docs/forge-demo-work/source-feedback-sft-q3.json <<'PY'
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from bufogen.evaluation.benchmark import _publish_source_feedback
from bufogen.generation import Manifest
from bufogen.integrations.telemetry import init_weave

source, ledger = map(Path, sys.argv[1:])
rows = Manifest(source).read()
if not rows:
    raise SystemExit("No saved rows found")
ledger.parent.mkdir(parents=True, exist_ok=True)
state = json.loads(ledger.read_text()) if ledger.exists() else {}
client = init_weave()

async def publish():
    for key, name in (("astra_judge", "AstraReferenceJudge"), ("scene_quality", "SceneQuality")):
        selected = [row for row in rows if key in row.scores]
        await _publish_source_feedback(
            client, selected, {key: SimpleNamespace(name=name)}, state, ledger
        )

asyncio.run(publish())
client.flush()
statuses = Counter(
    f"{target}:{receipt['status']}"
    for scorers in state.get("source_feedback", {}).values()
    for targets in scorers.values()
    for target, receipt in targets.items()
)
print(json.dumps(dict(statuses), indent=2))
if any(status.endswith(":error") for status in statuses):
    raise SystemExit("Publication has retryable errors; inspect the ledger")
PY
```

Successful custom feedback is not evidence that native Agent Scores columns have been populated. Inspect each receipt's `score_columns` value. Live publication and UI verification remain blocked by the session's connector authentication and network failures.

## Controlled runtime

Benchmark defaults now resolve the shared policy per arm without environment overrides. Resolved policy is frozen in each arm's `runtime_policy`, supplied explicitly to the agent, and `runtime.py` is included in source hashes. Explicit token/thinking settings still override defaults. Historical manifests remain readable; existing frozen run configuration is never rewritten or silently migrated for resume.
