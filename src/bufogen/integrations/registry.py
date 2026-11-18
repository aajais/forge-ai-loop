"""Artifacts and Registry links with explicit input lineage."""

import os
from pathlib import Path

from bufogen.settings import project_path


class RegistryClient:
    def __init__(self):
        self.entity, self.project = project_path().split("/")
        self.registry = os.getenv("WANDB_REGISTRY", "Bufo Forge")

    def target_path(self, name: str) -> str:
        # Registry ownership is organization-level, not the run's team entity.
        # Omitting the prefix lets the SDK resolve the current organization.
        organization = os.getenv("WANDB_REGISTRY_ORGANIZATION", "").strip()
        path = f"wandb-registry-{self.registry}/{name}"
        return f"{organization}/{path}" if organization else path

    def _push(
        self,
        name: str,
        kind: str,
        files: list[Path],
        *,
        sources=(),
        metadata=None,
        aliases=("latest",),
        link_registry=True,
    ) -> str:
        import wandb

        with wandb.init(
            entity=self.entity,
            project=self.project,
            job_type=f"publish-{kind}",
            config={"sources": list(sources)},
        ) as run:
            for source in sources:
                run.use_artifact(source.removeprefix("wandb-artifact:///"))
            artifact = wandb.Artifact(
                name,
                type=kind,
                metadata=metadata or {},
                storage_region="coreweave-us" if kind == "lora" else None,
            )
            for path in files:
                if path.is_dir():
                    artifact.add_dir(str(path))
                else:
                    artifact.add_file(str(path), name=path.name)
            artifact = run.log_artifact(artifact, aliases=list(aliases))
            artifact.wait()
            if link_registry:
                run.link_artifact(
                    artifact,
                    target_path=self.target_path(name),
                    aliases=list(aliases),
                )
            return f"{self.entity}/{self.project}/{artifact.name}"

    def push_dataset(
        self,
        path: Path,
        *,
        name="bufo-curated-dataset",
        sources=(),
        metadata=None,
        link_registry=True,
    ):
        return self._push(
            name, "dataset", [path], sources=sources, metadata=metadata, link_registry=link_registry
        )

    def push_prompts(self, paths: list[Path], *, sources=()):
        return self._push("bufo-aria-prompts", "prompts", paths, sources=sources)

    def push_checkpoint(
        self, path: Path, *, name: str, base_model: str, sources=(), aliases=("latest",)
    ):
        from bufogen.training.common import validate_adapter

        validate_adapter(path, base_model)
        return self._push(
            name,
            "lora",
            [path],
            sources=sources,
            aliases=aliases,
            metadata={"wandb.base_model": base_model},
        )

    def link_checkpoint(self, reference: str, *, name: str, sources=(), aliases=("latest",)):
        import wandb

        with wandb.init(
            entity=self.entity, project=self.project, job_type="promote-checkpoint"
        ) as run:
            for source in sources:
                run.use_artifact(source.removeprefix("wandb-artifact:///"))
            artifact = run.use_artifact(reference.removeprefix("wandb-artifact:///"), type="lora")
            run.link_artifact(
                artifact, self.target_path(name), aliases=list(aliases)
            )
            return f"wandb-artifact:///{self.entity}/{self.project}/{artifact.name}"

    def download(self, reference: str, *, kind: str | None = None) -> Path:
        import wandb

        artifact = wandb.Api().artifact(reference.removeprefix("wandb-artifact:///"), type=kind)
        return Path(artifact.download())
