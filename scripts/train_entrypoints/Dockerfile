FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir 'transformers>=5.5,<6' 'trl>=1.0,<2' 'peft>=0.18,<1' \
    'accelerate>=1.10,<2' 'bitsandbytes>=0.48,<1' 'datasets>=4,<5' 'wandb>=0.28,<0.29' pillow
WORKDIR /app
CMD ["sleep", "infinity"]
