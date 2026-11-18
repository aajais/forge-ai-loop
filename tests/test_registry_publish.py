import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/registry_publish.py"
spec = importlib.util.spec_from_file_location("registry_publish", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_manifest_links_are_pinned_and_collections_named():
    links, uploads, collections = mod.plan()
    assert all(ref.endswith(tuple(f":v{i}" for i in range(0, 40))) for _, ref, *_ in links)
    assert "bufo-sft-qwen3.8-27b-1ep" in collections and "bufo-distill-prompts" in collections
    selected = [link for link in links if "selected" in link[3]]
    assert [link[1] for link in selected] == ["wandb-smle/bufo-gen/bufo-stageb-sft-q3-20261023:v1"]


def test_plan_rejects_unpinned_or_missing():
    with pytest.raises(ValueError):
        mod.plan(links=[("c", "e/p/name:latest", "lora", ["a"], None)], uploads=[])
    with pytest.raises(ValueError):
        mod.plan(
            links=[], uploads=[("c", "prompts", [Path("/nonexistent/x.yaml")], ["a"], "d", {})]
        )
