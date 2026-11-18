"""Registry of judge calibration records, keyed by (rubric_version, samples) and scorer configuration."""

import json

from bufogen.settings import ROOT

CALIBRATED: dict[tuple[str, int], str] = {
    ("bufo-astra-v5", 3): "runs/distill/calibration-v1/calibration-record.json",
}

# The full scorer configuration (astra_judge.config_key form) each record calibrated. The record
# itself stores only rubric, hashes, judge_model, samples, threshold and confidence_floor, so the
# remaining fields come from runs/distill/arms-v1/astra-v5-k3/scorer.json, the configuration the
# calibrated k=3 rows were scored under. Any difference in any field means "not calibrated".
CALIBRATED_CONFIGS: dict[tuple[str, int], dict] = {
    ("bufo-astra-v5", 3): {
        "rubric_version": "bufo-astra-v5",
        "rubric_hash": "3e89d3db7ca42c1d6af67d0e38f79c7271f310565dbc6c1b1851b614d7a3b0cf",
        "schema_hash": "694e23a8ebcae180c11365e63185a69655757af92476b6f6044c01b632d6cc23",
        "judge_model": "gpt-6-astra",
        "judge_provider": "openai",
        "judge_base_url": "https://api.openai.com/v1",
        "api_mode": "chat_completions",
        "response_format": "json_schema_strict",
        "image_detail": "high",
        "temperature": "provider_default",
        "max_completion_tokens": 3200,
        "threshold": 0.65,
        "confidence_floor": 0.6,
        "samples": 3,
    },
}


def _registry_key(config: dict) -> tuple[str, int]:
    return (config.get("rubric_version"), config.get("samples", 1))


def record(config: dict) -> dict | None:
    """The calibration record for exactly this scorer configuration, or None unless its gate passed."""
    from bufogen.evaluation.scorers.astra_judge import config_key

    if not isinstance(config, dict):
        return None
    key = _registry_key(config)
    path = CALIBRATED.get(key)
    if path is None or config_key(config) != CALIBRATED_CONFIGS.get(key):
        return None
    try:
        data = json.loads((ROOT / path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    gate = data.get("gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        return None
    return data


def record_path(config: dict) -> str | None:
    return CALIBRATED[_registry_key(config)] if record(config) is not None else None


def status(config: dict) -> str:
    return "calibrated" if record(config) is not None else "uncalibrated"
