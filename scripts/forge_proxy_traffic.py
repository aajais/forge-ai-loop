"""Send labelled demo traffic through the Forge distillation proxy.

Every generation calls the routing alias (default backend ``forge_proxy``), so Forge's
weighted router picks the target per conversation and Forge Analytics / Observe see the
traffic. Prompts come from the ``train`` split of config/distill_prompts.yaml, taken
round-robin across families; validation and heldout prompts are never used.

The batch is identifiable by its conversation ids (``<run-id>:<prompt-id>`` in
``wandb.thread_id``). Extra request metadata is opt-in via ``--label KEY=VALUE``: the proxy
strips ``wandb.*`` keys but forwards any other key to the provider, and OpenAI targets reject
``metadata`` unless ``store`` is enabled, so labels break Sol-routed conversations. The served
model is read from each response and kept in the manifest
(trajectory.provenance.resolved_serving_models).

The proxy speaks Chat Completions and may route to OpenAI reasoning models, which reject
``max_tokens`` (they want ``max_completion_tokens``); the client built here rewrites that
field on the wire. W&B Inference targets accept either.

  BUFO_INFERENCE_TIMEOUT_S=900 uv run python scripts/forge_proxy_traffic.py \
      --count 100 --concurrency 4 --output runs/forge-demo/proxy-traffic-20261026
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from bufogen.models import BackendConfig, load_backends
from bufogen.settings import ROOT

PROMPTS = ROOT / "config/distill_prompts.yaml"


def select_prompts(prompts, *, split="train", count=100, offset=0):
    """Round-robin over families (stable order) so the traffic covers every family evenly."""
    pool = sorted((p for p in prompts if p.get("split") == split), key=lambda p: p["id"])
    by_family = defaultdict(list)
    for prompt in pool:
        by_family[prompt["family"]].append(prompt)
    families = sorted(by_family)
    ordered, index = [], 0
    while any(by_family.values()):
        family = families[index % len(families)]
        index += 1
        if by_family[family]:
            ordered.append(by_family[family].pop(0))
    return ordered[offset : offset + count]


@dataclass(frozen=True)
class LabeledProxyBackend(BackendConfig):
    """A forge backend whose proxied requests carry extra non-reserved metadata."""

    labels: dict = field(default_factory=dict)

    def request_extra_body(self, conversation_id):
        body = super().request_extra_body(conversation_id)
        if body is None:
            return None
        reserved = sorted(k for k in self.labels if k.startswith("wandb."))
        if reserved:
            raise ValueError(f"labels may not use reserved metadata keys: {reserved}")
        body["metadata"].update(self.labels)
        return body


def labeled_backend(backend: BackendConfig, labels: dict, *, model: str | None = None):
    if backend.provider != "forge":
        raise ValueError(f"{backend.name} is not a Forge proxy backend")
    if model:
        backend = replace(backend, model=model)
    return LabeledProxyBackend(**asdict(backend), labels=dict(labels))


def served_model(row) -> str:
    provenance = ((row.trajectory or {}).get("provenance") or {}) if row else {}
    models = provenance.get("resolved_serving_models") or []
    return models[-1] if models else "unresolved"


def rewrite_completion_tokens(request):
    """Return a request whose chat-completions body uses max_completion_tokens, if needed."""
    if not request.url.path.endswith("/chat/completions") or not request.content:
        return request
    try:
        body = json.loads(request.content)
    except ValueError:
        return request
    if not isinstance(body, dict) or "max_tokens" not in body:
        return request
    body["max_completion_tokens"] = body.pop("max_tokens")
    content = json.dumps(body).encode()
    headers = dict(request.headers)
    headers["content-length"] = str(len(content))
    import httpx

    return httpx.Request(
        request.method, request.url, headers=headers, content=content, extensions=request.extensions
    )


def proxy_client(backend, *, timeout_s):
    """AsyncOpenAI client for the proxy: records served models, rewrites max_tokens."""
    import httpx
    from openai import AsyncOpenAI

    serving_models = []

    class RewritingTransport(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            return await super().handle_async_request(rewrite_completion_tokens(request))

    async def observe_model(response):
        if response.is_success:
            await response.aread()
            try:
                model = response.json().get("model")
            except (ValueError, AttributeError):
                return
            if isinstance(model, str) and model and model not in serving_models:
                serving_models.append(model)

    client = AsyncOpenAI(
        api_key=os.environ[backend.api_key_env],
        base_url=backend.base_url,
        timeout=timeout_s,
        max_retries=2,
        http_client=httpx.AsyncClient(
            transport=RewritingTransport(),
            event_hooks={"response": [observe_model]},
            timeout=timeout_s,
        ),
    )
    client.bufo_serving_models = serving_models
    return client


EDITS = [
    "Add a pair of sunglasses.",
    "Make Bufo wave with one hand.",
    "Add a small CoreWeave logo on the chest.",
    "Give Bufo a party hat.",
    "Make the background a soft blue gradient.",
    "Add a speech bubble that says 'LGTM'.",
    "Make Bufo look surprised.",
    "Add a tiny laptop in front of Bufo.",
]


def follow_up_for(prompt_id, *, rate, seed):
    """Deterministically pick an edit prompt for this conversation, or None."""
    rng = random.Random(f"{seed}:{prompt_id}")
    if rng.random() >= rate:
        return None
    return rng.choice(EDITS)


def phases(items, ramp):
    """Split items into consecutive chunks, one per concurrency level in ramp."""
    if not ramp:
        raise ValueError("ramp needs at least one concurrency level")
    if any(c < 1 for c in ramp):
        raise ValueError("ramp levels must be positive")
    n = len(items)
    bounds = [round(i * n / len(ramp)) for i in range(len(ramp) + 1)]
    return [(ramp[i], items[bounds[i] : bounds[i + 1]]) for i in range(len(ramp))]


def failure_row(prompt, backend, run_id, exc, latency_s):
    """Record a provider/agent failure as a manifest row (exception class + HTTP status only)."""
    from bufogen.generation import BufoGeneration

    status = getattr(exc, "status_code", None)
    error = type(exc).__name__ + (f" (HTTP {status})" if status else "")
    body = getattr(exc, "body", None)
    detail = body.get("error", body) if isinstance(body, dict) else {}
    if isinstance(detail, dict):
        # Parameter name and error code only; never the provider's message text.
        tags = [str(detail[k]) for k in ("param", "code") if detail.get(k)]
        if tags:
            error += " [" + ", ".join(tags) + "]"
    return BufoGeneration(
        id=uuid4().hex,
        prompt_id=prompt["id"],
        group_id=f"{run_id}:{prompt['id']}",
        prompt=prompt["prompt"],
        backend=backend.name,
        model=backend.model,
        svg="",
        valid=False,
        latency_s=round(latency_s, 1),
        error=error,
        theme=prompt.get("family", "general"),
    )


def summarize(rows, *, run_id, backend, labels, started_at):
    by_model = defaultdict(lambda: Counter())
    latencies = []
    for row in rows:
        model = served_model(row)
        by_model[model]["requests"] += 1
        by_model[model]["valid"] += int(bool(row.valid))
        by_model[model]["errors"] += int(bool(row.error))
        by_model[model]["truncated_calls"] += row.truncated_calls
        by_model[model]["input_tokens"] += row.input_tokens
        by_model[model]["output_tokens"] += row.output_tokens
        if row.latency_s is not None:
            latencies.append(row.latency_s)
    return {
        "run_id": run_id,
        "backend": backend.name,
        "requested_alias": backend.model,
        "labels": labels,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "prompts": sum(1 for r in rows if ":turn" not in r.prompt_id),
        "follow_up_turns": sum(1 for r in rows if ":turn" in r.prompt_id),
        "valid": sum(bool(r.valid) for r in rows),
        "errors": sum(bool(r.error) for r in rows),
        "latency_p50_s": round(statistics.median(latencies), 1) if latencies else None,
        "served_models": {m: dict(c) for m, c in sorted(by_model.items())},
    }


async def run(args):
    from bufogen.agents.agent import BufoAgent
    from bufogen.generation import Manifest
    from bufogen.integrations.telemetry import init_weave
    from bufogen.rendering.renderer import Renderer

    run_id = args.run_id or f"proxy-traffic-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    labels = dict(kv.split("=", 1) for kv in args.label)
    backend = labeled_backend(load_backends()[args.backend], labels, model=args.model)
    prompts = select_prompts(
        yaml.safe_load(PROMPTS.read_text()), split=args.split, count=args.count, offset=args.offset
    )
    output = Path(args.output)
    manifest = Manifest(output / "manifest.jsonl")
    done = {r.prompt_id for r in manifest.read()} if manifest.path.exists() else set()
    todo = [p for p in prompts if p["id"] not in done]
    print(
        f"{run_id}: {len(todo)} prompts to send via {backend.name} ({backend.model}), "
        f"{len(done)} already in manifest, concurrency {args.concurrency}",
        flush=True,
    )
    init_weave()
    renderer = Renderer()
    rows = list(manifest.read()) if done else []
    started_at = datetime.now(timezone.utc).isoformat()
    ramp = [int(c) for c in args.ramp.split(",")] if args.ramp else [args.concurrency]

    async def one(index, prompt, semaphore):
        async with semaphore:
            agent = BufoAgent(
                backend,
                renderer,
                output / "items" / prompt["id"],
                client=proxy_client(backend, timeout_s=args.timeout_s),
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                inference_timeout_s=args.timeout_s,
                web_search_enabled=False,
            )
            started = time.monotonic()
            try:
                row = await agent.generate(
                    prompt["prompt"],
                    prompt_id=prompt["id"],
                    group_id=f"{run_id}:{prompt['id']}",
                    theme=prompt.get("family", "general"),
                )
            except Exception as exc:  # one failing target must not abort the batch
                row = failure_row(prompt, backend, run_id, exc, time.monotonic() - started)
            manifest.append(row)
            rows.append(row)
            print(
                f"[{index + 1}/{len(todo)}] {prompt['id']} -> {served_model(row)} "
                f"valid={row.valid} error={'yes' if row.error else 'no'} "
                f"{time.monotonic() - started:.0f}s",
                flush=True,
            )
            edit = follow_up_for(prompt["id"], rate=args.follow_up_rate, seed=args.follow_up_seed)
            if edit is None or not row.valid:
                return
            # Second turn in the same conversation (same thread id): the proxy keeps it on
            # the target that served the first turn.
            started = time.monotonic()
            try:
                follow = await agent.generate(
                    edit,
                    prompt_id=f"{prompt['id']}:turn1",
                    group_id=f"{run_id}:{prompt['id']}",
                    theme=prompt.get("family", "general"),
                    history=[row.to_dict()],
                )
            except Exception as exc:
                follow = failure_row(
                    {**prompt, "id": f"{prompt['id']}:turn1", "prompt": edit},
                    backend,
                    run_id,
                    exc,
                    time.monotonic() - started,
                )
            manifest.append(follow)
            rows.append(follow)
            print(
                f"[{index + 1}/{len(todo)}] {prompt['id']}:turn1 ({edit}) -> "
                f"{served_model(follow)} valid={follow.valid} "
                f"error={'yes' if follow.error else 'no'} {time.monotonic() - started:.0f}s",
                flush=True,
            )

    try:
        offset = 0
        for concurrency, chunk in phases(todo, ramp):
            print(f"phase: concurrency {concurrency}, {len(chunk)} prompts", flush=True)
            semaphore = asyncio.Semaphore(concurrency)
            await asyncio.gather(*(one(offset + i, p, semaphore) for i, p in enumerate(chunk)))
            offset += len(chunk)
    finally:
        await renderer.close()
        summary = summarize(
            rows, run_id=run_id, backend=backend, labels=labels, started_at=started_at
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", default="forge_proxy")
    parser.add_argument("--model", help="override the alias, e.g. bufo-gen@v1")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=None, help="default: runtime policy")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra request metadata (forwarded to the provider; see module docstring)",
    )
    parser.add_argument(
        "--ramp", help="comma-separated concurrency per consecutive phase, e.g. 2,6,2"
    )
    parser.add_argument(
        "--follow-up-rate",
        type=float,
        default=0.0,
        help="fraction of valid conversations that get one edit turn (default 0)",
    )
    parser.add_argument("--follow-up-seed", type=int, default=20261026)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.count < 1 or args.concurrency < 1:
        parser.error("--count and --concurrency must be positive")
    if not 0 <= args.follow_up_rate <= 1:
        parser.error("--follow-up-rate must be between 0 and 1")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
