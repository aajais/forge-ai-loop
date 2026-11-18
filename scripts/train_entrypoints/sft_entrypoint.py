"""GPU sandbox entrypoint. Credentials arrive on stdin, never argv or logs."""

import json
import os
import sys
from pathlib import Path


def lora_targets(model, *, multimodal: bool) -> list[str]:
    """Avoid Gemma vision/audio wrapper projections, which PEFT cannot adapt."""
    targets = [
        name
        for name, _ in model.named_modules()
        if name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
        and (not multimodal or ".language_model." in name)
    ]
    if not targets:
        raise ValueError("No supported language attention projections found for LoRA.")
    return targets


def main():
    payload = json.load(sys.stdin)
    os.environ["WANDB_API_KEY"] = payload.pop("wandb_api_key")
    token = payload.pop("hf_token", "")
    if token:
        os.environ["HF_TOKEN"] = token
    os.environ["WANDB_PROJECT"] = payload["project"]
    os.environ["WANDB_ENTITY"] = payload["entity"]
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    import wandb

    if not 1 <= payload["rank"] <= 16:
        raise ValueError("Rank must be <=16 for serverless serving.")
    with wandb.init(
        entity=payload["entity"],
        project=payload["project"],
        name=payload["name"],
        job_type="sft-distillation",
        config=payload,
    ) as run:
        artifact = run.use_artifact(
            payload["dataset_ref"].removeprefix("wandb-artifact:///"), type="dataset"
        )
        source = Path(artifact.download()) / "train.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError("Training dataset is empty.")
        dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
        base = payload["base_model"]
        config = AutoConfig.from_pretrained(base)
        multimodal = hasattr(config, "vision_config")
        model_cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        processor = (
            AutoProcessor.from_pretrained(base)
            if multimodal
            else AutoTokenizer.from_pretrained(base)
        )
        model = model_cls.from_pretrained(
            base,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            ),
        )
        lora = LoraConfig(
            r=payload["rank"],
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=lora_targets(model, multimodal=multimodal),
            task_type="CAUSAL_LM",
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=processor,
            train_dataset=dataset,
            peft_config=lora,
            args=SFTConfig(
                output_dir="/tmp/training",
                max_steps=payload["steps"],
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=1e-4,
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=1,
                save_strategy="no",
                report_to="wandb",
                max_length=8192,
                seed=42,
            ),
        )
        trainer.train()
        output = Path("/tmp/adapter")
        trainer.model.save_pretrained(output, safe_serialization=True)
        processor.save_pretrained(output)
        adapter = json.loads((output / "adapter_config.json").read_text())
        if adapter["r"] > 16 or adapter["base_model_name_or_path"] != base:
            raise ValueError("Saved adapter violates the serving contract.")
        artifact = wandb.Artifact(
            payload["name"],
            type="lora",
            storage_region="coreweave-us",
            metadata={"wandb.base_model": base, "training_method": "sft", "rank": payload["rank"]},
        )
        artifact.add_dir(str(output))
        artifact = run.log_artifact(artifact, aliases=["latest", "sft"])
        artifact.wait()
        reference = f"{payload['entity']}/{payload['project']}/{artifact.name}"
    print(json.dumps({"artifact": reference}))


if __name__ == "__main__":
    main()
