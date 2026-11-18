# Agent Lens status — 2026-10-15

**Agent Lens ingestion and Insights extraction are working for `wandb-smle/bufo-gen`.** At 16:58:54 UTC, authenticated read-only queries through installed Weave 0.53.11 returned seven agent identities and existing extracted intent/failure signals. MCP authentication failures do not describe the working local credential path.

| Live measure | Result | Scope |
| --- | ---: | --- |
| Agent identities | 7 | `agent_agents_query`, project-wide |
| Conversation groups | 404 | `agent_spans_query`, grouped by `conversation_id`, no time filter |
| Conversations with an extracted intent sentiment | 390 | Same grouping; OR across all five SDK sentiment values |
| Conversations with an extracted failure severity | 280 | Same grouping; OR across `info`, `major`, `minor` |

The intent match is **390/404 = 96.5% of returned conversation groups**, not classified-turn coverage. The failure match means a conversation has at least one extracted failure; it is not an error-span rate or a generation-quality pass rate. The all-conversation query did not explicitly exclude empty conversation IDs. These project-wide counts include historical activity, current evaluation traffic, retries, and smoke runs, so they are not counts for only the pilot or held-out dataset. Agent invocation counts can include nested invocations and must not be treated as completed generations.

The seven returned names were `Bufo closed_quality_bar`, `Bufo open_base`, `Bufo open_tuned`, `Bufo open_sft_rl`, `Bufo open_finetuned`, `Bufo forge_proxy`, and `bufo-gen`. The receipt retains exact invocation counts, timestamps, request filters, SDK version, and query scope: [agent-lens-status.json](../../runs/forge-loop/agent-lens-status.json).

## Remaining cluster/ARIA gap

**Completed clustering runs, stable topic IDs, cluster membership coverage, and project enablement settings were not verifiable through the installed SDK's exposed read methods.** The SDK exposes `AgentInsightFilter` for intent/failure category, sentiment/severity, and known topic UUIDs. It does not expose a method to list clustering runs or discover topic UUIDs. No undocumented endpoint was guessed or called.

Actual nonzero sentiment and severity matches prove existing extraction; they do **not** prove that a successful clustering run exists. Official documentation says clustering runs periodically, the Insights page shows its last completion or “No completed clustering run,” and Top clusters appears after sufficient turns have been clustered. The next definitive check is the project's **Agent Lens → Insights** page and its latest-cluster timestamp. [Agent Lens Insights](https://docs.coreweave.com/products/agent-lens/insights).

This inspection also does not establish that ARIA performed the pilot curation. The pilot's assistant-reviewed targets and ARIA prompt artifacts must retain their existing provenance; extracted Insights are not an ARIA review receipt.

## Reproducible interface

The read client was `RemoteHTTPTraceServer.from_env()` with `ApiKeyCredentials` loaded through `bufogen.settings`; `weave.init()` was not called. Only these SDK methods were used:

- `agent_agents_query(AgentsQueryReq(project_id="wandb-smle/bufo-gen", limit=100))`
- `agent_spans_query(AgentSpansQueryReq(project_id="wandb-smle/bufo-gen", group_by=[AgentGroupByRef(key="conversation_id")], insight_filters=..., limit=1))`; response `total_count` supplies the complete group count, independent of the one-row result limit.

Insight filters were `intent_sentiment` with `frustrated,dissatisfied,neutral,satisfied,delighted`, and `failure_severity` with `info,major,minor`. The installed SDK describes these as conversation filters backed by extracted Insights data. Its source is `weave/trace_server/agents/types.py` and `weave/trace_server/query_builder/agent_insight_filters.py`. [Official activity semantics](https://docs.coreweave.com/products/agent-lens/conversations/view-activity).

No remote changes, runtime edits, new traces, or colleague messages were made.
