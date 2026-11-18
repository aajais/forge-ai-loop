import json
import ssl
from pathlib import Path

import httpx
import pytest
from test_forge import client
from test_forge_learning_rate import imported_state

from bufogen.cli import dispatch, parser
from bufogen.integrations import forge
from bufogen.integrations.forge import ForgeClient
from bufogen.workflows.forge_loop import submit_training


def sent_config(requests):
    return json.loads(requests[-1].content)["model_config"]["config"]


def test_rank_field_name_is_recorded_with_evidence():
    assert isinstance(forge.FORGE_LORA_RANK_FIELD, str) and forge.FORGE_LORA_RANK_FIELD
    source = Path(forge.__file__).read_text()
    comment = source[: source.index("FORGE_LORA_RANK_FIELD =")]
    block = comment[comment.rindex("# LoRA rank field") :]
    assert "PROVISIONAL" in block and "422" in block and "Unrecognized key" in block


def test_rank_goes_under_discovered_field_and_is_omitted_by_default():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(202, json={"id": "job-1"})

    with client(handler) as api:
        api.create_finetune(dataset_id="d", model_name="m")
        assert forge.FORGE_LORA_RANK_FIELD not in sent_config(requests)
        api.create_finetune(dataset_id="d", model_name="m", lora_rank=8)
    assert sent_config(requests)[forge.FORGE_LORA_RANK_FIELD] == 8


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, "8"])
def test_invalid_rank_fails_before_network(rank):
    requests = []
    with client(lambda request: requests.append(request)) as api:
        with pytest.raises(ValueError, match="lora_rank"):
            api.create_finetune(dataset_id="d", model_name="m", lora_rank=rank)
    assert not requests


def test_submit_records_rank_in_durable_intent(tmp_path):
    requests = []
    field = forge.FORGE_LORA_RANK_FIELD

    def handler(request):
        requests.append(request)
        return httpx.Response(202, json={"id": "job-1", "status": "queued"})

    with client(handler) as api:
        path = imported_state(tmp_path, api)
        state = submit_training(path, api, epochs=3, lora_rank=8)
    assert state["training_config"][field] == 8
    assert sent_config(requests)[field] == 8


def test_rank_drift_on_existing_job_is_refused(tmp_path):
    field = forge.FORGE_LORA_RANK_FIELD
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(202, json={"id": "job-1", "status": "queued"})

    with client(handler) as api:
        path = imported_state(tmp_path, api)
        submit_training(path, api, lora_rank=8)
        with pytest.raises(ValueError, match="differs"):
            submit_training(path, api, lora_rank=16)
        with pytest.raises(ValueError, match="differs"):
            submit_training(path, api, lora_rank=1)
    assert json.loads(path.read_text())["training_config"][field] == 8
    assert [r.method for r in requests] == ["POST"]


def test_tls12_client_pins_verified_tls12_context(monkeypatch):
    seen = {}

    class Transport(httpx.BaseTransport):
        def __init__(self, verify):
            seen["context"] = verify

    monkeypatch.setattr(forge.httpx, "HTTPTransport", Transport)
    ForgeClient(api_key="k", entity="e", project_alias="a", tls12=True).close()
    context = seen["context"]
    assert context.minimum_version == context.maximum_version == ssl.TLSVersion.TLSv1_2
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


def test_tls12_conflicts_with_explicit_transport():
    with pytest.raises(ValueError, match="tls12"):
        ForgeClient(
            api_key="k",
            entity="e",
            project_alias="a",
            tls12=True,
            transport=httpx.MockTransport(lambda r: None),
        )


async def test_cli_forwards_rank_and_tls12(tmp_path, monkeypatch):
    import bufogen.cli as cli
    import bufogen.workflows.forge_loop as loop

    class Api:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    built, calls = [], []
    monkeypatch.setattr(cli, "forge_client", lambda args: built.append(args.tls12) or Api())
    monkeypatch.setattr(
        loop, "submit_training", lambda path, client, **kw: calls.append(kw) or {"status": "queued"}
    )
    state = tmp_path / "s.json"
    args = parser().parse_args(
        ["--stage", "forge-submit", "--forge-state", str(state), "--lora-rank", "8", "--tls12"]
    )
    await dispatch(args)
    assert calls == [{"epochs": 1, "batch_size": 2, "lora_rank": 8}]
    assert built == [True]
    args = parser().parse_args(["--stage", "forge-submit", "--forge-state", str(state)])
    await dispatch(args)
    assert calls[-1] == {"epochs": 1, "batch_size": 2} and built[-1] is False


async def test_cli_forwards_learning_rate_only_when_given(tmp_path, monkeypatch):
    import bufogen.cli as cli
    import bufogen.workflows.forge_loop as loop

    class Api:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    calls = []
    monkeypatch.setattr(cli, "forge_client", lambda args: Api())
    monkeypatch.setattr(
        loop, "submit_training", lambda path, client, **kw: calls.append(kw) or {"status": "q"}
    )
    base = ["--stage", "forge-submit", "--forge-state", str(tmp_path / "s.json")]
    await dispatch(parser().parse_args([*base, "--learning-rate", "1e-4"]))
    await dispatch(parser().parse_args(base))
    assert calls == [
        {"epochs": 1, "batch_size": 2, "learning_rate": 1e-4},
        {"epochs": 1, "batch_size": 2},
    ]


QWEN = "Qwen/Qwen3.8-27B"


def test_student_allowlist_accepts_both_models_and_rejects_others():
    assert forge.STUDENT_MODELS == {
        "google/gemma-4-26B-A4B-it",
        QWEN,
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
    }
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(202, json={"id": "job-1"})

    with client(handler) as api:
        api.create_finetune(dataset_id="d", model_name="m")
        api.create_finetune(dataset_id="d", model_name="m", base_model=QWEN)
        with pytest.raises(ValueError, match="configured student"):
            api.create_finetune(dataset_id="d", model_name="m", base_model="Qwen/Qwen3.6-27B")
    assert [r["base_model"] for r in requests] == ["google/gemma-4-26B-A4B-it", QWEN]


def test_submit_uses_state_base_model_and_refuses_base_model_drift(tmp_path):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(202, json={"id": "job-1", "status": "queued"})

    with client(handler) as api:
        path = imported_state(tmp_path, api)
        state = json.loads(path.read_text())
        state["base_model"] = QWEN
        path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="base_model"):
            submit_training(path, api, base_model="google/gemma-4-26B-A4B-it")
        assert not requests
        submit_training(path, api, base_model=QWEN)
        assert requests[-1]["base_model"] == QWEN
        with pytest.raises(ValueError, match="base_model"):
            submit_training(path, api, base_model="google/gemma-4-26B-A4B-it")
    assert len(requests) == 1


def test_workflow_initialisation_accepts_only_allowlisted_base_models(tmp_path):
    from bufogen.workflows.forge_loop import initialize_workflow

    with client(lambda r: None) as api:
        with pytest.raises(ValueError, match="configured student"):
            initialize_workflow(
                tmp_path / "s.json",
                api,
                source_path=tmp_path / "x",
                dataset_name="d",
                model_name="m",
                project_id="p",
                base_model="other/model",
            )


async def test_cli_forwards_base_model_only_when_given(tmp_path, monkeypatch):
    import bufogen.cli as cli
    import bufogen.workflows.forge_loop as loop

    class Api:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    calls = []
    monkeypatch.setattr(cli, "forge_client", lambda args: Api())
    monkeypatch.setattr(
        loop, "submit_training", lambda path, client, **kw: calls.append(kw) or {"status": "q"}
    )
    base = ["--stage", "forge-submit", "--forge-state", str(tmp_path / "s.json")]
    await dispatch(parser().parse_args([*base, "--base-model", QWEN]))
    await dispatch(parser().parse_args(base))
    assert calls == [
        {"epochs": 1, "batch_size": 2, "base_model": QWEN},
        {"epochs": 1, "batch_size": 2},
    ]
