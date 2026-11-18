import json
from dataclasses import replace

from bufogen.evaluation.benchmark import run_benchmark
from bufogen.generation import BufoGeneration
from test_benchmark import FakeRenderer, backend, prompts


async def test_controlled_benchmark_freezes_resolved_runtime_and_explicit_overrides(
    tmp_path, monkeypatch
):
    import bufogen.evaluation.benchmark as benchmark

    received = []

    class Agent:
        def __init__(self, config, renderer, output_dir, **kwargs):
            self.backend = config
            received.append(kwargs)

        async def generate(self, prompt, **kwargs):
            return BufoGeneration(
                kwargs["prompt_id"],
                prompt,
                self.backend.name,
                self.backend.model,
                kwargs["group_id"],
                "",
                False,
                0,
            )

    monkeypatch.setattr(benchmark, "BufoAgent", Agent)
    monkeypatch.setattr(benchmark, "Renderer", FakeRenderer)
    monkeypatch.setenv("BUFO_DISABLE_THINKING", "true")
    monkeypatch.setenv("BUFO_INFERENCE_TIMEOUT_S", "1")
    qwen = replace(backend(), model="Qwen/Qwen3.8-27B")
    await run_benchmark(prompts()[:1], [qwen], tmp_path / "default")
    config = json.loads((tmp_path / "default/benchmark.json").read_text())["configuration"]
    assert received[0]["max_tokens"] == 20000
    assert received[0]["disable_thinking"] is False
    assert received[0]["inference_timeout_s"] == 900
    assert config["arms"][0]["sampling"]["max_tokens"] == 20000
    assert config["arms"][0]["runtime_policy"]["timeout_s"] == 900
    assert "src/bufogen/runtime.py" in config["source_hashes"]
    await run_benchmark(
        prompts()[:1], [qwen], tmp_path / "explicit", max_tokens=1234, disable_thinking=True
    )
    assert received[1]["max_tokens"] == 1234
    assert received[1]["disable_thinking"] is True
