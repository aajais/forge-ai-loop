"""Replay a benchmark arm's saved generations as a Weave Model over one shared prompt dataset.

Every arm's evaluation is published against the same prompt-only dataset (identical rows, so
identical Weave digest), and the arm's rendered outputs come from a ``ReplayArm`` model keyed by
``prompt_id``. That is what lets the Weave UI compare the arms row by row.
"""

import weave
from pydantic import PrivateAttr

from bufogen.integrations.observability import visual_generation


def shared_dataset_name(split: str, count: int) -> str:
    return f"bufo-{split}-{count}"


def shared_dataset_rows(rows) -> list[dict]:
    """Prompt-only rows in prompt_id order; the same for every arm of the same split."""
    shared = []
    for row in sorted(rows, key=lambda r: r.prompt_id):
        metadata = row.benchmark_metadata
        shared.append(
            {
                "prompt_id": row.prompt_id,
                "family": metadata["family"],
                "split": metadata["split"],
                "prompt": row.prompt,
                "requirements": list(metadata["requirements"]),
            }
        )
    return shared


def split_of(rows) -> str:
    splits = {r.benchmark_metadata["split"] for r in rows}
    return splits.pop() if len(splits) == 1 else "mixed"


class ReplayArm(weave.Model):
    """One benchmark arm; ``predict`` returns the saved generation for a dataset row."""

    arm: str
    model_label: str
    checkpoint: str | None = None
    _generations: dict = PrivateAttr(default_factory=dict)

    def __init__(self, *, generations: dict[str, dict], **data):
        super().__init__(**data)
        self._generations = dict(generations)

    @weave.op()
    def predict(self, prompt: str, prompt_id: str) -> dict:
        return visual_generation(prompt, self._generations[prompt_id])
