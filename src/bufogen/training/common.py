import json
from abc import ABC, abstractmethod
from pathlib import Path

from bufogen.generation import BufoGeneration
from bufogen.integrations.registry import RegistryClient


def validate_adapter(path: Path, base_model: str):
    config = json.loads((path / "adapter_config.json").read_text())
    rank = config.get("r", 0)
    if (
        not isinstance(rank, int)
        or not 1 <= rank <= 16
        or any(r > 16 for r in config.get("rank_pattern", {}).values())
    ):
        raise ValueError("LoRA rank must be between 1 and 16.")
    if config.get("base_model_name_or_path") != base_model:
        raise ValueError("Adapter base model does not match the served base model.")
    if not (path / "adapter_model.safetensors").is_file():
        raise ValueError("Missing PEFT adapter_model.safetensors.")


def reward(row: BufoGeneration, *, scorer_configuration: dict) -> float:
    """Joint identity/action reward from a bound, validated scene judgment.

    Judge failures remain errors in evaluation, but earn zero training reward.
    The caller must separately qualify the scorer with reviewed calibration data.
    """
    from bufogen.evaluation.scorers.scene_quality import scene_result

    score = row.scores.get("scene_quality") or {}
    requirements = (row.benchmark_metadata or {}).get("requirements")
    if (
        not row.valid
        or row.error
        or score.get("status") != "ok"
        or score.get("generation_id") != row.id
        or score.get("scorer_configuration") != scorer_configuration
        or score.get("requirements") != requirements
        or score.get("rubric_version") != scorer_configuration.get("rubric_version")
    ):
        return 0.0
    try:
        checked = scene_result(score, requirements, threshold=scorer_configuration["threshold"])
    except (ValueError, TypeError, KeyError):
        return 0.0
    if not checked["bufo_present"]:
        return 0.0
    # Multiplication prevents an attractive unrelated Bufo from earning a high reward.
    craft = (checked["anatomy"] + checked["composition"] + checked["readability"]) / 3
    return checked["similarity"] * checked["constraint_coverage"] * craft


def astra_reward(row: BufoGeneration, *, scorer_configuration: dict) -> float:
    """Reference-anchored Astra reward: similarity x requirement coverage x mean craft.

    Zero unless the merged judgment is ok, bound to this generation and judge configuration
    (compared through config_key), a Bufo is present, and the generation is valid and completed.
    The caller must separately qualify the judge with require_astra_calibration.
    """
    from bufogen.evaluation.scorers.astra_judge import config_key

    score = (row.scores or {}).get("astra_judge") or {}
    metadata = row.benchmark_metadata or {}
    if (
        not row.valid
        or row.error
        or metadata.get("status") != "completed"
        or score.get("status") != "ok"
        or score.get("generation_id") != row.id
        or score.get("bufo_present") is not True
        or config_key(score.get("scorer_configuration")) != config_key(scorer_configuration)
    ):
        return 0.0
    values = []
    for key in ("similarity", "requirement_coverage", "anatomy", "composition", "readability"):
        value = score.get(key)
        if type(value) not in (int, float) or not 0 <= value <= 1:
            return 0.0
        values.append(float(value))
    similarity, coverage, *craft = values
    return similarity * coverage * (sum(craft) / 3)


class PostTrainingJob(ABC):
    def __init__(self, dataset_ref: str, *, name: str, base_model: str):
        self.dataset_ref, self.name, self.base_model = dataset_ref, name, base_model
        self.registry = RegistryClient()

    def dataset(self) -> Path:
        path = self.registry.download(self.dataset_ref, kind="dataset") / "train.jsonl"
        if not path.is_file() or not path.read_text().strip():
            raise ValueError("Dataset artifact must contain a nonempty train.jsonl.")
        return path

    @abstractmethod
    async def train(self, *, steps: int = 5) -> str:
        """Train, record lineage, and return a served checkpoint URI."""
