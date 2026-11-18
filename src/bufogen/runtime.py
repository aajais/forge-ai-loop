"""Shared inference defaults; explicit run settings always take precedence."""

import os
from dataclasses import asdict, dataclass

QWEN_MODEL = "Qwen/Qwen3.8-27B"


def is_evaluated_qwen(backend) -> bool:
    return backend.model == QWEN_MODEL or backend.base_model == QWEN_MODEL


@dataclass(frozen=True)
class InferencePolicy:
    max_tokens: int | None
    disable_thinking: bool
    # Injected clients may have fractional or unbounded read timeouts.
    timeout_s: int | float | None
    policy_id: str

    def to_dict(self):
        return asdict(self)


def resolve_inference_policy(
    backend,
    *,
    max_tokens: int | None = None,
    disable_thinking: bool | None = None,
    timeout_s: int | None = None,
    use_environment: bool = True,
    studio_mode: bool = False,
) -> InferencePolicy:
    """Resolve model defaults, then environment settings, then explicit overrides.

    Controlled callers can disable environment overrides and freeze the returned
    values. The profile identifies default selection, not an evaluation claim.
    The studio profile always enables thinking and omits client output-token limits.
    """
    qwen = is_evaluated_qwen(backend)
    if studio_mode:
        # Live demos always reason and leave output length to the serving provider.
        max_tokens = None
        disable_thinking = False
    elif max_tokens is None:
        max_tokens = 20000 if qwen else (10000 if backend.provider == "openai" else 6000)
    if disable_thinking is None:
        disable_thinking = use_environment and os.getenv(
            "BUFO_DISABLE_THINKING", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
    if timeout_s is None:
        raw_timeout = os.getenv("BUFO_INFERENCE_TIMEOUT_S") if use_environment else None
        if raw_timeout is not None:
            try:
                timeout_s = int(raw_timeout.strip())
            except ValueError:
                raise ValueError(
                    "BUFO_INFERENCE_TIMEOUT_S must be an integer number of seconds"
                ) from None
        else:
            timeout_s = 900 if qwen else 180
    if not studio_mode and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("max_tokens must be a positive integer")
    if type(timeout_s) is not int or timeout_s < 1:
        raise ValueError("BUFO_INFERENCE_TIMEOUT_S / timeout_s must be at least 1")
    if type(disable_thinking) is not bool:
        raise ValueError("disable_thinking must be a boolean")
    return InferencePolicy(
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
        timeout_s=timeout_s,
        policy_id=(
            "studio-thinking-provider-limits-v1"
            if studio_mode
            else "qwen-20k-thinking-v1"
            if qwen
            else "provider-defaults-v1"
        ),
    )
