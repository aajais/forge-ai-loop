import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from bufogen.runtime import is_evaluated_qwen, resolve_inference_policy
from bufogen.settings import ROOT, project_path, required


def inference_timeout_s() -> int:
    """Per-request inference timeout in seconds (default 180).

    BUFO_INFERENCE_TIMEOUT_S raises it for runs whose max_tokens let a single
    thinking call exceed three minutes; a value below 1 is rejected.
    """
    raw = os.getenv("BUFO_INFERENCE_TIMEOUT_S", "180").strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("BUFO_INFERENCE_TIMEOUT_S must be an integer number of seconds") from None
    if value < 1:
        raise ValueError("BUFO_INFERENCE_TIMEOUT_S must be at least 1")
    return value


def model_presentation(
    model, *, base_model="", tuned=False, alias=False, resolved_models=(), prefix="ft"
):
    """Display identity separately from the exact ID sent to the serving API."""
    if alias:
        resolved = list(dict.fromkeys(resolved_models))
        presentations = [
            model_presentation(item, base_model=base_model if base_model != model else "")
            for item in resolved
        ]
        label = (
            " / ".join(item["model_label"] for item in presentations)
            if resolved
            else f"{model} (routing alias; model pending)"
        )
        versions = [item["model_version"] for item in presentations if item["model_version"]]
        return {
            "model_label": label,
            "model_version": " / ".join(versions) or None,
            "routing_alias": model,
        }
    tuned = tuned or model.startswith("wandb-artifact:///")
    label = (
        f"{prefix}-{base_model}"
        if tuned and base_model
        else (f"{prefix}-model (base ID unrecorded)" if tuned else model)
    )
    return {"model_label": label, "model_version": model if tuned and model else None}


def generation_model_presentation(row, *, config=None):
    """Prefer recorded identity; never relabel a past response with today's model."""
    value = row if isinstance(row, dict) else row.to_dict()
    provenance = (value.get("trajectory") or {}).get("provenance", {})
    config = config or {}
    # A caller may provide the frozen arm or an exactly matching live config.
    matching = config.get("model") == value["model"]
    base = provenance.get("base_model") or (config.get("base_model", "") if matching else "")
    method = provenance.get("tuning_method") or (config.get("tuning_method") if matching else "")
    return model_presentation(
        value["model"],
        base_model=base,
        prefix="rl" if method == "rl" else "ft",
        tuned=bool(method)
        or value["backend"] in {"open_tuned", "open_rl"}
        or value["backend"].endswith("_rl"),
        alias=bool(provenance.get("requested_alias")) or value["backend"] == "forge_proxy",
        resolved_models=provenance.get("resolved_serving_models", []),
    )


def frozen_model_presentations(arms):
    """Read legacy frozen arms without rewriting their stored labels or hashes."""
    base = next((arm["model"] for arm in arms if arm["name"] == "open_base"), "")
    result = {}
    for arm in arms:
        config = dict(arm)
        if config["model"].startswith("wandb-artifact:///") and not config.get("base_model"):
            config["base_model"] = base
        result[arm["name"]] = config
    return result


@dataclass(frozen=True)
class BackendConfig:
    name: str
    label: str
    model: str
    model_env: str
    provider: str
    base_url: str
    api_key_env: str
    supports_tools: bool = True
    base_model: str = ""
    tuning_method: str = ""

    @property
    def presentation(self):
        return model_presentation(
            self.model,
            base_model=self.base_model,
            tuned=bool(self.tuning_method),
            alias=self.provider == "forge",
            prefix="rl" if self.tuning_method == "rl" else "ft",
        )

    @property
    def display_name(self):
        return self.presentation["model_label"]

    def require_configured(self, *, require_pinned: bool = True):
        if not self.model:
            raise ValueError(f"Set {self.model_env} to a served model or LoRA artifact URI.")
        if self.model.upper() == "PENDING" or re.search(r"(?:/|:)PENDING(?:$|:)", self.model, re.I):
            raise ValueError(f"Set {self.model_env}: PENDING is not a served checkpoint.")
        if (
            require_pinned
            and self.model.startswith("wandb-artifact:///")
            and not re.fullmatch(r"wandb-artifact:///[^/\s]+/[^/\s]+/[^/:\s]+:v[0-9]+", self.model)
        ):
            raise ValueError(f"Set {self.model_env} to a pinned artifact checkpoint ending in :vN.")
        return self

    def request_extra_body(self, conversation_id: str) -> dict | None:
        if self.provider != "forge":
            return None
        if not conversation_id:
            raise ValueError("Forge requests require a conversation ID")
        return {
            "metadata": {
                "wandb.entity": required("WANDB_ENTITY"),
                "wandb.thread_id": conversation_id,
            }
        }

    def thinking_extra_body(self, disable: bool) -> dict | None:
        if self.provider == "coreweave" and (disable or is_evaluated_qwen(self)):
            return {"chat_template_kwargs": {"enable_thinking": not disable}}
        return None

    def client(self, *, timeout_s: int | None = None):
        from openai import AsyncOpenAI

        self.require_configured()
        timeout_s = resolve_inference_policy(self, timeout_s=timeout_s).timeout_s
        headers = {"OpenAI-Project": project_path()} if self.provider == "coreweave" else {}
        options = {}
        serving_models = []
        if self.provider == "forge":
            import httpx

            async def observe_model(response):
                # Non-streaming chat requests only. Record the server's actual model,
                # never infer it from the stable routing alias.
                if response.is_success:
                    await response.aread()
                    try:
                        model = response.json().get("model")
                    except (ValueError, AttributeError):
                        return
                    if isinstance(model, str) and model and model not in serving_models:
                        serving_models.append(model)

            options["http_client"] = httpx.AsyncClient(
                event_hooks={"response": [observe_model]}, timeout=timeout_s
            )
        client = AsyncOpenAI(
            api_key=required(self.api_key_env),
            base_url=self.base_url,
            default_headers=headers,
            timeout=timeout_s,
            max_retries=2,
            **options,
        )
        client.bufo_serving_models = serving_models
        return client


def load_backends(path: Path | None = None) -> dict[str, BackendConfig]:
    data = yaml.safe_load((path or ROOT / "config/models.yaml").read_text())
    result = {}
    for name, values in data.items():
        values = dict(values)
        values["model"] = os.getenv(values["model_env"], values["model"]).strip()
        base_env = values.pop("base_model_env", None)
        values["base_model"] = (
            os.getenv(base_env, values.get("base_model", "")).strip()
            if base_env
            else values.get("base_model", "" if values["provider"] == "forge" else values["model"])
        )
        url_env = values.pop("base_url_env")
        values["base_url"] = os.getenv(url_env, values["base_url"])
        config = BackendConfig(name=name, **values)
        values["label"] = config.display_name
        result[name] = BackendConfig(name=name, **values)
    return result
