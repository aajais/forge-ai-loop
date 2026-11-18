import pytest

from bufogen.models import inference_timeout_s


def test_inference_timeout_defaults_to_180(monkeypatch):
    monkeypatch.delenv("BUFO_INFERENCE_TIMEOUT_S", raising=False)
    assert inference_timeout_s() == 180


def test_inference_timeout_reads_env(monkeypatch):
    monkeypatch.setenv("BUFO_INFERENCE_TIMEOUT_S", " 900 ")
    assert inference_timeout_s() == 900


@pytest.mark.parametrize("raw", ["0", "-5", "soon"])
def test_inference_timeout_rejects_bad_values(monkeypatch, raw):
    monkeypatch.setenv("BUFO_INFERENCE_TIMEOUT_S", raw)
    with pytest.raises(ValueError):
        inference_timeout_s()


def test_rl_arms_present_under_their_own_label():
    from bufogen.models import generation_model_presentation, load_backends

    arms = load_backends()
    assert arms["rl_q3"].label == "rl-Qwen/Qwen3.8-27B"
    assert arms["rl_q3"].display_name == "rl-Qwen/Qwen3.8-27B"
    assert arms["sft_q3"].display_name == "ft-Qwen/Qwen3.8-27B"
    frozen = {"model": "wandb-artifact:///e/p/m:v6", "base_model": "Qwen/Qwen3.8-27B"}
    row = {"backend": "rl_q3", "model": frozen["model"]}
    assert (
        generation_model_presentation(row, config=frozen | {"tuning_method": "rl"})["model_label"]
        == "rl-Qwen/Qwen3.8-27B"
    )
    assert (
        generation_model_presentation(row, config=frozen | {"tuning_method": "sft"})["model_label"]
        == "ft-Qwen/Qwen3.8-27B"
    )
