"""Set manual inference prices in the Forge distillation project.

Forge analytics price a request from (1) a manual price on the fine-tune or provider
model, (2) the W&B Inference rate of a fine-tune's base model, (3) the LiteLLM catalog.
LoRA adapters are billed at their base model's rate, so this script records that rate on
each fine-tune explicitly and fills in provider models that have no catalog price.

Usage (prices are USD per million tokens, input then output):
  uv run python scripts/forge_set_pricing.py \
      --finetune da6bd91a-d7b8-478c-ad00-f6f540925537 0.40 3.00 \
      --provider-model openai 1ac24d8c-5878-539d-9c08-1a60c55d5ba7 2.00 10.00 \
      [--dry-run]
"""

import argparse
import os

import bufogen.settings  # noqa: F401  (loads .env)
from bufogen.integrations.forge import ForgeClient

PRICE_KEYS = ("input_usd_per_mtok", "output_usd_per_mtok")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", default="bufo-gen")
    parser.add_argument(
        "--finetune", nargs=3, action="append", default=[], metavar=("ID", "IN", "OUT")
    )
    parser.add_argument(
        "--provider-model",
        nargs=4,
        action="append",
        default=[],
        metavar=("PROVIDER", "MODEL_UUID", "IN", "OUT"),
    )
    parser.add_argument("--dry-run", action="store_true", help="print planned changes only")
    args = parser.parse_args(argv)
    if not args.finetune and not args.provider_model:
        parser.error("nothing to set")

    with ForgeClient(
        api_key=os.environ["WANDB_API_KEY"],
        entity=os.environ["WANDB_ENTITY"],
        project_alias=args.project,
    ) as api:
        for finetune_id, cost_in, cost_out in args.finetune:
            before = api.get_finetune(finetune_id)
            print(f"finetune {before.get('model_name')} ({finetune_id})")
            print("  before:", {k: before.get(k) for k in PRICE_KEYS})
            if args.dry_run:
                print("  would set:", {"input": float(cost_in), "output": float(cost_out)})
                continue
            after = api.set_finetune_pricing(
                finetune_id, input_usd_per_mtok=float(cost_in), output_usd_per_mtok=float(cost_out)
            )
            print("  after: ", {k: after.get(k) for k in PRICE_KEYS})
        providers = {}
        for provider, model_id, cost_in, cost_out in args.provider_model:
            providers.setdefault(provider, api.get_provider(provider))
            model = next(
                (m for m in providers[provider].get("models", []) if m.get("id") == model_id), None
            )
            if model is None:
                raise SystemExit(f"provider {provider} has no model with id {model_id}")
            print(f"provider model {provider}/{model.get('model')} ({model_id})")
            print("  before:", {k: model.get(k) for k in PRICE_KEYS})
            if args.dry_run:
                print("  would set:", {"input": float(cost_in), "output": float(cost_out)})
                continue
            after = api.set_provider_model_pricing(
                provider,
                model_id,
                input_usd_per_mtok=float(cost_in),
                output_usd_per_mtok=float(cost_out),
            )
            print("  after: ", {k: after.get(k) for k in PRICE_KEYS})


if __name__ == "__main__":
    main()
