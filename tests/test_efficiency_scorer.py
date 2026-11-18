from bufogen.evaluation.scorers.efficiency import EfficiencyScorer, serving_path


def generation(**overrides):
    row = dict(
        id="gen-1",
        prompt_id="p1",
        prompt="a bufo waving",
        backend="open_base",
        model="google/gemma-4-26B-A4B-it",
        group_id="g1",
        svg="<svg/>",
        valid=True,
        latency_s=12.5,
        input_tokens=900,
        output_tokens=300,
        tool_calls=2,
        truncated_calls=1,
        model_calls=[
            dict(
                index=0,
                finish_reason="stop",
                input_tokens=500,
                output_tokens=200,
                reasoning_tokens=10,
                reasoning_chars=40,
                latency_s=5.0,
                served_model=None,
            ),
            dict(
                index=1,
                finish_reason="length",
                input_tokens=400,
                output_tokens=100,
                reasoning_chars=0,
                latency_s=7.5,
                served_model=None,
            ),
        ],
        benchmark_metadata={"status": "complete"},
    )
    row.update(overrides)
    return row


def test_efficiency_sums_reasoning_tokens_and_counts_calls():
    result = EfficiencyScorer().score(generation())
    assert result["reasoning_tokens"] == 10
    assert result["model_call_count"] == 2
    assert result["truncated_calls"] == 1
    assert result["tool_calls"] == 2
    assert result["input_tokens"] == 900
    assert result["output_tokens"] == 300
    assert result["latency_s"] == 12.5
    assert result["render_ok"] is True
    assert result["status"] == "complete"
    assert result["generation_id"] == "gen-1"
    assert result["serving_path"] == "wandb_inference"
    assert set(result) == {
        "status",
        "render_ok",
        "latency_s",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "tool_calls",
        "model_call_count",
        "truncated_calls",
        "serving_path",
        "generation_id",
    }


def test_status_defaults_to_unknown_without_benchmark_metadata():
    result = EfficiencyScorer().score(generation(benchmark_metadata=None))
    assert result["status"] == "unknown"


def test_serving_path_mapping():
    assert serving_path("open_base", "m") == "wandb_inference"
    assert serving_path("closed_quality_bar", "m") == "openai"
    assert serving_path("forge_proxy", "m") == "forge_proxy"
    assert serving_path("sft-x", "m") == "unknown"


def test_output_wrapped_in_generation_key_is_unwrapped():
    wrapped = EfficiencyScorer().score({"generation": generation(backend="forge_proxy")})
    assert wrapped["generation_id"] == "gen-1"
    assert wrapped["serving_path"] == "forge_proxy"
    assert wrapped["reasoning_tokens"] == 10
