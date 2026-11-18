import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4


@dataclass
class BufoGeneration:
    prompt_id: str
    prompt: str
    backend: str
    model: str
    group_id: str
    svg: str
    valid: bool
    latency_s: float
    id: str = field(default_factory=lambda: uuid4().hex)
    theme: str = "general"
    png_path: str | None = None
    svg_path: str | None = None
    error: str | None = None
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    conversation_url: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    origin: str = "generation"
    scores: dict = field(default_factory=dict)
    prompt_version: str | None = None
    weave_call_id: str | None = None
    response_text: str = ""
    parent_id: str | None = None
    input_image_path: str | None = None
    emoji_name: str = ""
    png_content_ref: str | None = None
    composition_warnings: list[str] = field(default_factory=list)
    web_searches: list[dict] = field(default_factory=list)
    trajectory: dict | None = None
    model_calls: list[dict] = field(default_factory=list)
    truncated_calls: int = 0
    training_review: dict | None = None
    benchmark_metadata: dict | None = None
    native_trace_id: str | None = None
    native_span_id: str | None = None
    execution_metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = Lock()

    def append(self, generation: BufoGeneration):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a") as file:
            file.write(json.dumps(generation.to_dict()) + "\n")

    def read(self) -> list[BufoGeneration]:
        if not self.path.exists():
            return []
        with self._lock:
            rows = [
                BufoGeneration(**json.loads(line))
                for line in self.path.read_text().splitlines()
                if line.strip()
            ]
            for row in rows:
                for field_name in ("png_path", "svg_path", "input_image_path"):
                    value = getattr(row, field_name)
                    if value and not Path(value).is_absolute():
                        setattr(row, field_name, str((self.path.parent / value).resolve()))
            return rows
