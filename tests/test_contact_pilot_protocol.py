import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "docs/forge-contact-pilot/evaluate.py"
    spec = importlib.util.spec_from_file_location("contact_pilot_protocol_test", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_protocol_refuses_to_adopt_prior_outputs_without_frozen_inputs(tmp_path):
    m = module()
    (tmp_path / "manifest.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="existing"):
        m.freeze_protocol(tmp_path, {"instructions": "original"})


def test_protocol_rejects_changed_instructions_before_resume(tmp_path):
    m = module()
    m.freeze_protocol(tmp_path, {"instructions": "original"})
    m.freeze_protocol(tmp_path, {"instructions": "original"})
    with pytest.raises(ValueError, match="changed"):
        m.freeze_protocol(tmp_path, {"instructions": "new"})


def test_pilot_has_balanced_diagnostic_relations_and_no_confirmation_prompts():
    m = module()
    prompts = m.load_prompts()
    assert len(prompts) == 16
    assert len({p["family"] for p in prompts}) == 4
    assert all(p["split"] == "development" for p in prompts)
    assert all(len(p["requirements"]) >= 1 for p in prompts)
    import yaml
    existing = yaml.safe_load((m.ROOT / "config/forge_iteration2_prompts.yaml").read_text())
    sealed = {p["prompt"] for p in existing if p["split"] == "heldout"}
    assert not sealed.intersection(p["prompt"] for p in prompts)


def test_fixture_gate_rejects_reviews_that_do_not_match_current_compiler(tmp_path):
    m = module()
    import json
    (tmp_path / "assistant-review.json").write_text(json.dumps({
        "compiler_sha256": "old-compiler", "accepted_for_interface_test": True,
        "reviewer_kind": "assistant", "fixtures": [],
    }))
    with pytest.raises(ValueError, match="compiler"):
        m.validate_fixture_review(tmp_path, "new-compiler")


def test_live_interface_change_is_rejected_before_paid_generation(monkeypatch):
    m = module()
    monkeypatch.setattr(m.benchmark, "_configuration", lambda *a, **kw: {"instructions": "edited"})
    with pytest.raises(ValueError, match="changed"):
        m.verify_current_configuration([], [], {"instructions": "frozen"})


def test_fixture_gate_rejects_asset_change_even_when_compiler_is_unchanged(monkeypatch):
    m = module()
    monkeypatch.setattr(m.benchmark, "_source_hashes", lambda: {"fullbody.svg": "new"})
    with pytest.raises(ValueError, match="dependencies"):
        m.verify_fixture_dependencies([{"source_hashes": {"fullbody.svg": "old"}}])
