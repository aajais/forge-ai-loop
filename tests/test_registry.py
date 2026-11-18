from unittest.mock import MagicMock

import pytest

from bufogen.integrations.registry import RegistryClient


@pytest.mark.parametrize("link_registry", [False, True])
def test_dataset_artifact_can_be_published_without_registry_membership(
    tmp_path, monkeypatch, link_registry
):
    import wandb

    run = MagicMock()
    artifact = MagicMock()
    artifact.name = "emoji-rl:v0"
    run.log_artifact.return_value = artifact
    context = MagicMock()
    context.__enter__.return_value = run
    monkeypatch.setattr(wandb, "init", lambda **kw: context)
    monkeypatch.setattr(wandb, "Artifact", lambda *a, **kw: artifact)
    registry = RegistryClient()
    reference = registry.push_dataset(tmp_path, name="emoji-rl", link_registry=link_registry)
    assert reference.endswith("/emoji-rl:v0")
    artifact.wait.assert_called_once()
    assert run.link_artifact.call_count == int(link_registry)


def test_registry_target_does_not_confuse_team_with_organization(monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "a-team")
    monkeypatch.setenv("WANDB_REGISTRY", "Bufo Forge")
    monkeypatch.delenv("WANDB_REGISTRY_ORGANIZATION", raising=False)
    assert RegistryClient().target_path("checkpoints") == "wandb-registry-Bufo Forge/checkpoints"
    monkeypatch.setenv("WANDB_REGISTRY_ORGANIZATION", "owning-org")
    assert RegistryClient().target_path("checkpoints") == (
        "owning-org/wandb-registry-Bufo Forge/checkpoints"
    )


def test_candidate_registry_link_uses_requested_alias_without_changing_route(monkeypatch):
    import wandb

    run = MagicMock()
    artifact = MagicMock()
    artifact.name = "checkpoint:v1"
    run.use_artifact.return_value = artifact
    context = MagicMock()
    context.__enter__.return_value = run
    monkeypatch.setattr(wandb, "init", lambda **kw: context)
    monkeypatch.delenv("WANDB_REGISTRY_ORGANIZATION", raising=False)
    registry = RegistryClient()
    registry.link_checkpoint(
        "wandb-artifact:///team/project/checkpoint:v1",
        name="bufo-qwen",
        aliases=("demo-candidate",),
    )
    assert run.link_artifact.call_args.kwargs["aliases"] == ["demo-candidate"]
    assert run.link_artifact.call_args.args[1] == registry.target_path("bufo-qwen")
