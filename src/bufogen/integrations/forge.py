"""Documented Model Distillation API; no implicit providers, routing, or retries."""

import json
import math
import re
import ssl
from copy import deepcopy
from types import SimpleNamespace
from urllib.parse import quote

import certifi
import httpx

from bufogen.curation.curate import _validate_review
from bufogen.curation.trajectories import CAPTURE_SOURCE, _validate_messages

DEFAULT_BASE_URL = "https://forge.coreweave.com/api/distillation/v1"
DEFAULT_TRAINING_URL = "https://forge.coreweave.com/api/training/v1"
STUDENT_MODEL = "google/gemma-4-26B-A4B-it"
# Allowlisted students; Forge fixes the LoRA rank per base model (Qwen3.8-27B and Nemotron: rank 8).
NEMOTRON_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"
STUDENT_MODELS = {STUDENT_MODEL, "Qwen/Qwen3.8-27B", NEMOTRON_MODEL}
# LoRA rank field inside model_config.config (design open item 1), PROVISIONAL: NOT ACCEPTED.
# Discovery on 2026-10-23, no job created. Every probe was a POST /finetunes against a
# nonexistent dataset id, so nothing could be created; the API validates the config keys
# strictly BEFORE the dataset lookup (valid keys -> 404 not_found; unknown -> 422
# "Unrecognized key"). Accepted (404): epochs, batch_size, assistant_turns, learning_rate,
# metric_logging. Rejected (422): lora_rank, rank, lora_r, r, adapter_rank, lora, lora_config,
# peft, rank_r, lora_alpha, weight_decay, warmup_ratio, max_seq_length, max_length, seed,
# lr_scheduler, gradient_accumulation_steps, dropout. The training API OpenAPI schema
# (SFTTrainingConfig, additionalProperties=false) also has no rank field, and no OpenAPI is
# served for the distillation API. Conclusion: no rank field is exposed; the sweep runs
# rank 1 only (adapter-capacity.md observed r=1). "lora_rank" is kept as the best guess so
# lora_rank=N is sent verbatim and the API's 422 is the authoritative answer.
FORGE_LORA_RANK_FIELD = "lora_rank"
ENTRY_FIELDS = {
    "messages",
    "tools",
    "tool_choice",
    "response_format",
    "row_id",
    "group_id",
    "split",
    "metadata",
    "provenance",
}


def validate_learning_rate(learning_rate):
    """Only explicit finite positive scalar overrides are supported here."""
    if learning_rate is not None and (
        type(learning_rate) not in (int, float)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be a finite positive number")


def validate_lora_rank(lora_rank):
    if lora_rank is not None and (type(lora_rank) is not int or lora_rank < 1):
        raise ValueError("lora_rank must be a positive integer")


class ForgeError(RuntimeError):
    def __init__(self, message, *, status_code=None, uncertain=False):
        super().__init__(message)
        self.status_code = status_code
        self.uncertain = uncertain


class ForgeClient:
    def __init__(
        self,
        *,
        api_key,
        entity,
        project_alias,
        base_url=DEFAULT_BASE_URL,
        training_base_url=DEFAULT_TRAINING_URL,
        transport=None,
        timeout=60,
        tls12=False,
    ):
        if tls12 and transport is not None:
            raise ValueError("tls12 cannot be combined with an explicit transport")
        if tls12:
            # Verified TLS 1.2 upload workaround (docs/forge-iteration2/train_reviewed.py).
            context = ssl.create_default_context(cafile=certifi.where())
            context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_2
            transport = httpx.HTTPTransport(verify=context)
        if not api_key or not entity or not project_alias:
            raise ValueError("Forge requires explicit API key, entity and project alias")
        self.entity, self.project_alias = entity, project_alias
        self.base_url, self.training_base_url = base_url.rstrip("/"), training_base_url.rstrip("/")
        self._key = api_key
        self._http = httpx.Client(
            transport=transport,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Wandb-Entity": entity},
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._http.close()

    @property
    def project_path(self):
        return "/projects/" + quote(self.project_alias, safe="")

    def _request(self, method, path, *, training=False, **kwargs):
        mutation = method != "GET" and kwargs.get("params", {}).get("dry_run") != "true"
        try:
            response = self._http.request(
                method, (self.training_base_url if training else self.base_url) + path, **kwargs
            )
        except httpx.TransportError:
            raise ForgeError(
                "Forge transport interrupted; reconcile the remote operation before retrying"
                if mutation
                else "Forge read/validation transport interrupted; retry is safe",
                uncertain=mutation,
            ) from None
        if response.is_error:
            hints = {
                401: "check WANDB_API_KEY authentication",
                403: "check entity/project permissions and model enablement",
                404: "check the explicit project alias and resource ID",
                409: "inspect resource conflicts before retrying",
                422: "check dataset or training configuration validation",
                429: "check quota and service rate limits",
            }
            detail = ""
            try:
                payload = response.json()
                error = payload.get("error", {})
                if isinstance(error, dict) and isinstance(error.get("message"), str):
                    value = f"{error.get('type', 'request_error')}: {error['message']}"
                else:
                    value = payload.get("message", payload.get("detail", ""))
                if isinstance(value, str):
                    detail = value.replace(self._key, "[REDACTED]")
                    detail = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", detail)
                    detail = re.sub(
                        r"(?i)(api[_-]?key|token|authorization)\s*[:=]\s*\S+",
                        r"\1=[REDACTED]",
                        detail,
                    )[:1000]
            except (ValueError, AttributeError):
                pass
            raise ForgeError(
                f"Forge HTTP {response.status_code}: "
                f"{hints.get(response.status_code, 'inspect service status and request configuration')}"
                + (f"; {detail}" if detail else ""),
                status_code=response.status_code,
                uncertain=mutation and response.status_code >= 500,
            )
        try:
            return response.json()
        except ValueError:
            raise ForgeError(
                "Forge returned invalid JSON; reconcile remote state", uncertain=mutation
            ) from None

    def list_projects(self):
        return self._request("GET", "/projects")

    def list_base_models(self):
        return self._request("GET", "/preview/base-models", training=True)

    def preflight(self, base_model=STUDENT_MODEL, *, project=None, project_id=None):
        projects = self.list_projects()
        projects = projects if isinstance(projects, list) else projects.get("projects", [])
        matches = [p for p in projects if p.get("alias") == self.project_alias]
        if len(matches) != 1:
            raise ValueError("Forge alias must resolve to exactly one existing project")
        found = matches[0]
        if found.get("entity") != self.entity or (project and found.get("project") != project):
            raise ValueError("Forge project entity/name does not match explicit configuration")
        if project_id and found.get("id") != project_id:
            raise ValueError("Forge project ID does not match explicit configuration")
        catalog = self.list_base_models()
        models = catalog if isinstance(catalog, list) else catalog.get("models", [])
        eligible = any(
            (
                m == base_model
                if isinstance(m, str)
                else (
                    m.get("name", m.get("id", m.get("base_model"))) == base_model
                    and m.get("enabled", True) is not False
                    and m.get("available", True) is not False
                )
            )
            for m in models
        )
        if not eligible:
            raise ValueError(f"Model is not advertised as eligible: {base_model}")
        return {
            "project": found,
            "base_model": base_model,
            "catalog_eligible": True,
            "job_acceptance_verified": False,
        }

    def setup_project(self, *, project, targets):
        """Explicit opt-in operation: imports a project AND publishes initial routing."""
        if not targets:
            raise ValueError("Project setup requires explicit initial routing targets")
        return self._request(
            "PUT",
            self.project_path,
            json={"project": project, "project_mode": "import", "targets": targets},
        )

    def create_dataset(self, name):
        return self._request("POST", self.project_path + "/datasets", json={"name": name})

    def get_dataset(self, dataset_id):
        return self._request("GET", self.project_path + "/datasets/" + quote(dataset_id, safe=""))

    def upload_entries(self, dataset_id, entries, *, dry_run):
        if not 1 <= len(entries) <= 5000:
            raise ValueError("Upload requires 1–5000 validated entries per request")
        if any(set(row) - ENTRY_FIELDS for row in entries):
            raise ValueError("Unsupported Forge entry fields")
        return self._request(
            "POST",
            self.project_path + "/datasets/" + quote(dataset_id, safe="") + "/entries",
            params={"mode": "append", "dry_run": str(dry_run).lower()},
            json=entries,
        )

    def create_finetune(
        self,
        *,
        dataset_id,
        model_name,
        base_model=STUDENT_MODEL,
        epochs=1,
        batch_size=2,
        learning_rate=None,
        lora_rank=None,
    ):
        if base_model not in STUDENT_MODELS:
            raise ValueError("This workflow requires an exact configured student model")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if batch_size != "auto" and (type(batch_size) is not int or not 1 <= batch_size <= 32):
            raise ValueError("batch_size must be auto or 1–32")
        validate_learning_rate(learning_rate)
        validate_lora_rank(lora_rank)
        config = {"epochs": epochs, "batch_size": batch_size, "assistant_turns": "all"}
        if learning_rate is not None:
            config["learning_rate"] = learning_rate
        if lora_rank is not None:
            config[FORGE_LORA_RANK_FIELD] = lora_rank
        return self._request(
            "POST",
            self.project_path + "/finetunes",
            json={
                "dataset_id": dataset_id,
                "model_name": model_name,
                "base_model": base_model,
                "model_config": {"config": config},
            },
        )

    def get_finetune(self, finetune_id):
        return self._request("GET", self.project_path + "/finetunes/" + quote(finetune_id, safe=""))

    def _pricing_body(self, input_usd_per_mtok, output_usd_per_mtok):
        for label, value in (("input", input_usd_per_mtok), ("output", output_usd_per_mtok)):
            if value is not None and not (0 < value <= 1_000_000):
                raise ValueError(f"{label} price must be in (0, 1000000] USD per million tokens")
        return {
            "input_usd_per_mtok": input_usd_per_mtok,
            "output_usd_per_mtok": output_usd_per_mtok,
        }

    def set_finetune_pricing(self, finetune_id, *, input_usd_per_mtok, output_usd_per_mtok):
        """Manual inference price for a fine-tune; analytics prefer it over base-model rates."""
        return self._request(
            "PUT",
            f"{self.project_path}/finetunes/{quote(str(finetune_id), safe='')}/pricing",
            json=self._pricing_body(input_usd_per_mtok, output_usd_per_mtok),
        )

    def get_provider(self, provider):
        return self._request("GET", f"/providers/{quote(provider, safe='')}")

    def set_provider_model_pricing(
        self, provider, model_id, *, input_usd_per_mtok, output_usd_per_mtok
    ):
        """Manual price for a provider model (identified by its UUID, not its name)."""
        return self._request(
            "PUT",
            f"/providers/{quote(provider, safe='')}/models/{quote(str(model_id), safe='')}",
            json=self._pricing_body(input_usd_per_mtok, output_usd_per_mtok),
        )


def _bounded_object(value, label):
    def depth(obj):
        if isinstance(obj, dict):
            return 1 + max((depth(v) for v in obj.values()), default=0)
        if isinstance(obj, list):
            return 1 + max((depth(v) for v in obj), default=0)
        return 0

    if not isinstance(value, dict) or len(json.dumps(value).encode()) > 65536 or depth(value) > 10:
        raise ValueError(f"{label} exceeds Forge object limits")


def reviewed_entries(examples, *, heldout_ids=(), heldout_families=()):
    """Map curated local exports to Forge's whitelist without losing original evidence."""
    entries, ids, groups, families = [], set(), {}, {}
    for example in examples:
        review = example.get("training_review")
        _validate_review(
            review,
            SimpleNamespace(
                id=example.get("source_generation_id"), prompt_id=example.get("prompt_id")
            ),
        )
        partition, family = example.get("partition"), example.get("prompt_family")
        if review["decision"] != "accept":
            raise ValueError("Explicit accepted review required")
        if partition not in {"train", "validation"} or review["partition"] != partition:
            raise ValueError("Only reviewed train/validation partitions may upload")
        if (
            family != review["prompt_family"]
            or family in heldout_families
            or example.get("prompt_id") in heldout_ids
        ):
            raise ValueError("Prompt family is inconsistent or held out")
        split = "val" if partition == "validation" else "train"
        if family in families and families[family] != split:
            raise ValueError("Prompt family spans train and validation")
        families[family] = split
        row_id, group_id = example.get("source_generation_id"), example.get("source_group_id")
        for identifier in (row_id, group_id):
            if not isinstance(identifier, str) or not identifier or len(identifier) > 512:
                raise ValueError("Missing or oversized row/group identifier")
        if row_id in ids or (group_id in groups and groups[group_id] != split):
            raise ValueError("Duplicate row ID or group spans train and validation")
        ids.add(row_id)
        groups[group_id] = split
        messages, tools = example.get("messages", []), example.get("tools", [])
        if not 2 <= len(messages) <= 1000 or not 1 <= len(tools) <= 128:
            raise ValueError("Invalid Forge message/tool count")
        if messages[0].get("role") != "system" or not messages[0].get("content"):
            raise ValueError("Frozen system instructions required")
        _validate_messages(messages, tools)
        provenance = example.get("provenance", {})
        if (
            provenance.get("capture_source") != CAPTURE_SOURCE
            or provenance.get("history_authentic") is not True
            or provenance.get("prior_conversation")
            or provenance.get("history_snapshot_compatible") is not True
            or provenance.get("source_models") != [example.get("source_model")]
            or provenance.get("final_output") != messages[-1].get("content")
        ):
            raise ValueError("Unqualified trajectory provenance")
        for local, remote in [
            ("source_model", "model"),
            ("prompt_id", "prompt_id"),
            ("source_group_id", "group_id"),
            ("prompt_version", "prompt_version"),
        ]:
            if not example.get(local) or provenance.get(remote) != example[local]:
                raise ValueError(f"Trajectory provenance mismatch: {local}")
        metadata = deepcopy(example.get("metadata", {}))
        _bounded_object(metadata, "metadata")
        # Validated source evidence wins over conflicting custom annotations.
        metadata.update({k: deepcopy(v) for k, v in example.items() if k not in ENTRY_FIELDS})
        _bounded_object(metadata, "metadata")
        _bounded_object(provenance, "provenance")
        entry = {
            k: deepcopy(example[k])
            for k in ("messages", "tools", "tool_choice", "response_format")
            if k in example
        }
        entry.update(
            row_id=row_id,
            group_id=group_id,
            split=split,
            metadata=metadata,
            provenance=deepcopy(provenance),
        )
        entries.append(entry)
    if not entries or not any(row["split"] == "train" for row in entries):
        raise ValueError("Dataset requires reviewed training examples")
    if len(entries) > 5000:
        raise ValueError("Pilot workflow supports at most 5000 entries in one atomic upload")
    return entries
