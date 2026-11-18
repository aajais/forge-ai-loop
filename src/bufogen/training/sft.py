import os

from bufogen.integrations.sandbox import SandboxJob
from bufogen.settings import ROOT, project_path, required
from bufogen.training.common import PostTrainingJob


class SFTTrainer(PostTrainingJob):
    async def train(self, *, steps: int = 5) -> str:
        if steps < 1:
            raise ValueError("Training steps must be positive.")
        required("CWSANDBOX_API_KEY")
        required("BUFO_SFT_IMAGE")
        self.dataset()  # Validate the artifact contract before allocating a GPU.
        entity, project = project_path().split("/")
        payload = {
            "dataset_ref": self.dataset_ref,
            "name": self.name,
            "base_model": self.base_model,
            "steps": steps,
            "rank": 8,
            "entity": entity,
            "project": project,
            "wandb_api_key": required("WANDB_API_KEY"),
            "hf_token": os.getenv("HF_TOKEN", ""),
        }
        async with SandboxJob(gpu=True, lifetime=int(os.getenv("BUFO_SFT_TIMEOUT", "7200"))) as job:
            result = await job.execute(
                ROOT / "scripts/train_entrypoints/sft_entrypoint.py", payload, stream_logs=True
            )
        return self.registry.link_checkpoint(
            result["artifact"], name=self.name, sources=[self.dataset_ref]
        )
