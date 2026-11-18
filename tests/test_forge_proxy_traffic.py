import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from bufogen.models import load_backends

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/forge_proxy_traffic.py"
spec = importlib.util.spec_from_file_location("forge_proxy_traffic", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def prompts():
    rows = []
    for family in ("b", "a", "c"):
        for i in range(3):
            rows.append({"id": f"{family}{i}", "family": family, "split": "train", "prompt": "x"})
    rows.append({"id": "h0", "family": "h", "split": "heldout", "prompt": "x"})
    rows.append({"id": "v0", "family": "v", "split": "validation", "prompt": "x"})
    return rows


def test_select_prompts_round_robins_families_and_excludes_other_splits():
    chosen = mod.select_prompts(prompts(), count=5)
    assert [p["id"] for p in chosen] == ["a0", "b0", "c0", "a1", "b1"]
    everything = mod.select_prompts(prompts(), count=100)
    assert Counter(p["split"] for p in everything) == {"train": 9}
    assert [p["id"] for p in mod.select_prompts(prompts(), count=2, offset=3)] == ["a1", "b1"]


def test_labeled_backend_adds_non_reserved_metadata(monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    backend = mod.labeled_backend(
        load_backends()["forge_proxy"], {"bufo.purpose": "demo-traffic"}, model="bufo-gen@v1"
    )
    assert backend.model == "bufo-gen@v1"
    assert backend.request_extra_body("conv-1") == {
        "metadata": {
            "wandb.entity": "test-team",
            "wandb.thread_id": "conv-1",
            "bufo.purpose": "demo-traffic",
        }
    }
    bad = mod.labeled_backend(load_backends()["forge_proxy"], {"wandb.project": "x"})
    with pytest.raises(ValueError):
        bad.request_extra_body("conv-1")
    with pytest.raises(ValueError):
        mod.labeled_backend(load_backends()["open_base"], {})


def test_failure_row_records_exception_class_and_status_only():
    class Boom(Exception):
        status_code = 401

    backend = load_backends()["forge_proxy"]
    row = mod.failure_row(
        {"id": "p1", "prompt": "draw", "family": "f"},
        backend,
        "run-1",
        Boom("secret body text"),
        12.34,
    )
    assert row.valid is False and row.error == "Boom (HTTP 401)"
    assert "secret" not in row.error and row.prompt_id == "p1" and row.backend == "forge_proxy"
    assert mod.summarize([row], run_id="run-1", backend=backend, labels={}, started_at="t")[
        "served_models"
    ] == {
        "unresolved": {
            "requests": 1,
            "valid": 0,
            "errors": 1,
            "truncated_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    }


def test_rewrite_completion_tokens_only_touches_chat_completions_max_tokens():
    import json

    import httpx

    req = httpx.Request(
        "POST",
        "https://proxy.example/v1/chat/completions",
        json={"model": "bufo-gen", "max_tokens": 6000, "messages": []},
    )
    out = mod.rewrite_completion_tokens(req)
    body = json.loads(out.content)
    assert body == {"model": "bufo-gen", "max_completion_tokens": 6000, "messages": []}
    assert out.headers["content-length"] == str(len(out.content))
    other = httpx.Request("POST", "https://proxy.example/v1/embeddings", json={"max_tokens": 1})
    assert mod.rewrite_completion_tokens(other) is other
    plain = httpx.Request("POST", "https://proxy.example/v1/chat/completions", json={"model": "x"})
    assert mod.rewrite_completion_tokens(plain) is plain


def test_follow_up_selection_is_deterministic_and_phases_split_evenly():
    picks = {pid: mod.follow_up_for(pid, rate=0.4, seed=1) for pid in (f"p{i}" for i in range(200))}
    assert picks == {pid: mod.follow_up_for(pid, rate=0.4, seed=1) for pid in picks}
    chosen = [v for v in picks.values() if v]
    assert 50 <= len(chosen) <= 110 and set(chosen) <= set(mod.EDITS)
    assert mod.follow_up_for("p1", rate=0.0, seed=1) is None
    items = list(range(10))
    assert mod.phases(items, [2, 6, 2]) == [(2, [0, 1, 2]), (6, [3, 4, 5, 6]), (2, [7, 8, 9])]
    assert mod.phases(items, [4]) == [(4, items)]
    with pytest.raises(ValueError):
        mod.phases(items, [0, 2])
