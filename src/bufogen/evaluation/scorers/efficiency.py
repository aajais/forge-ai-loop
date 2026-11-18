"""Deterministic cost and latency facts for a generation; no model calls."""

import weave

from bufogen.models import load_backends

SERVING_PATHS = {"coreweave": "wandb_inference", "openai": "openai", "forge": "forge_proxy"}


def serving_path(backend_name: str, model: str) -> str:
    backend = load_backends().get(backend_name)
    if backend is None:
        return "unknown"
    return SERVING_PATHS.get(backend.provider, "unknown")


class EfficiencyScorer(weave.Scorer):
    @weave.op()
    def score(self, output: dict) -> dict:
        output = output.get("generation", output)
        metadata = output.get("benchmark_metadata") or {}
        calls = output.get("model_calls") or []
        return {
            "status": metadata.get("status") or "unknown",
            "render_ok": bool(output.get("valid")),
            "latency_s": float(output.get("latency_s") or 0.0),
            "input_tokens": int(output.get("input_tokens") or 0),
            "output_tokens": int(output.get("output_tokens") or 0),
            "reasoning_tokens": sum(int(call.get("reasoning_tokens") or 0) for call in calls),
            "tool_calls": int(output.get("tool_calls") or 0),
            "model_call_count": len(calls),
            "truncated_calls": int(output.get("truncated_calls") or 0),
            "serving_path": serving_path(output.get("backend", ""), output.get("model", "")),
            "generation_id": output.get("id"),
        }
