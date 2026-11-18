"""Adapter audit must report exported capacity and never resolve mutable aliases."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'docs/forge-iteration2/verify_adapters.py'
    assert path.exists(), 'Adapter audit script not implemented'
    spec = importlib.util.spec_from_file_location('iteration2_adapter_audit', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ArtifactFile:
    size = 1000
    digest = 'config-file-digest'
    def __init__(self, config): self.config = config
    def download(self, root):
        p = Path(root) / 'adapter_config.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.config))
        return str(p)


class Artifact:
    id = 'immutable-artifact-id'
    digest = 'immutable-artifact-digest'
    qualified_name = 'entity/project/model:v7'
    def __init__(self, config): self.config = config
    def get_entry(self, name):
        assert name == 'adapter_config.json', 'Only adapter config may be downloaded'
        return ArtifactFile(self.config)


class API:
    def __init__(self, config): self.config, self.calls = config, []
    def artifact(self, name):
        self.calls.append(name)
        return Artifact(self.config)


@pytest.fixture
def state(tmp_path):
    path = tmp_path / 'training/e1/forge-state.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'deployed', 'finetune_id': 'job1',
        'model_artifact_ref': 'wandb-artifact:///entity/project/model:v7',
        'base_model': 'google/gemma-4-26B-A4B-it'}))
    return path


def test_reports_actual_rank_patterns_and_downloads_only_config(state):
    config = {'r': 4, 'lora_alpha': 32, 'target_modules': ['q_proj', 'experts'],
              'base_model_name_or_path': 'google/gemma-4-26B-A4B-it', 'rank_pattern': {'special': 8}}
    api = API(config)
    result = module().audit(state, api)
    assert result['adapter']['r'] == 4
    assert result['adapter']['rank_pattern'] == {'special': 8}
    assert result['adapter']['lora_alpha'] == 32
    assert result['artifact']['digest'] == Artifact.digest
    assert result['tensor_shapes_verified'] is False
    assert api.calls == ['entity/project/model:v7']
    assert result['adapter_config_sha256']
    assert (state.parent / 'adapter-verification.json').exists()


@pytest.mark.parametrize('changes', [
    {'status': 'training'}, {'model_artifact_ref': 'wandb-artifact:///entity/project/model:latest'},
    {'model_artifact_ref': ''},
])
def test_refuses_unready_or_mutable_artifact_before_network(state, changes):
    saved = json.loads(state.read_text()); saved.update(changes); state.write_text(json.dumps(saved))
    api = API({})
    with pytest.raises(ValueError): module().audit(state, api)
    assert not api.calls


def test_receipt_resume_checks_local_bytes_and_refuses_artifact_change(state):
    m = module(); api = API({'r': 1, 'lora_alpha': 32, 'target_modules': ['q_proj'], 'base_model_name_or_path': 'google/gemma-4-26B-A4B-it'})
    first = m.audit(state, api)
    assert m.audit(state, api) == first
    assert len(api.calls) == 1
    p = state.parent / 'adapter-audit/adapter_config.json'
    p.write_text('{}')
    with pytest.raises(ValueError, match='changed'): m.audit(state, api)


def test_exported_base_must_match_training_state(state):
    api = API({'r': 32, 'lora_alpha': 32, 'base_model_name_or_path': 'different/base'})
    with pytest.raises(ValueError, match='base'): module().audit(state, api)
    assert not (state.parent / 'adapter-verification.json').exists()
