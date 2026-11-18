import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from bufogen.generation import Manifest
from bufogen.models import load_backends
from bufogen.settings import ROOT, project_path
from bufogen.workflows.pipeline import load_prompts, run_pipeline, write_demo


def parser():
    result = argparse.ArgumentParser(description="Run the CoreWeave Forge bufo AI loop.")
    result.add_argument(
        "--stage",
        required=True,
        choices=[
            "forge-preflight",
            "benchmark",
            "benchmark-score",
            "build-references",
            "scene-review-queue",
            "export-reviewed",
            "forge-import",
            "forge-submit",
            "forge-status",
            "loop-report",
            "doctor",
            "sandbox-status",
            "render-smoke",
            "demo",
            "run",
            "eval",
            "eval-identity",
            "review-queue",
            "curate",
            "train-sft",
            "train-rl",
            "evaluate",
            "build-backup-dataset",
            "publish-prompts",
        ],
    )
    result.add_argument("--theme", help="Audience theme for demo mode")
    result.add_argument("--sandbox-id", help="Sandbox ID to inspect with --stage sandbox-status")
    result.add_argument("--backends", nargs="+", default=None)
    result.add_argument("--prompts", type=Path)
    result.add_argument(
        "--heldout-prompts", type=Path, default=ROOT / "config/prompts_heldout.yaml"
    )
    result.add_argument("--limit", type=int, default=None)
    result.add_argument("--output", type=Path, default=ROOT / "runs/latest")
    result.add_argument("--manifest", type=Path)
    result.add_argument("--prompts-ref", help="ARIA bufo-curated-prompts artifact reference")
    result.add_argument("--backup-ref", help="Optional bufo-backup-seed-dataset artifact reference")
    result.add_argument("--dataset-ref", help="Curated training dataset artifact reference")
    result.add_argument("--name", help="Unique checkpoint artifact name")
    result.add_argument("--steps", type=int, default=5)
    result.add_argument(
        "--tasks-per-step", type=int, default=1, help="train-rl: tasks (one group each) per step"
    )
    result.add_argument(
        "--expect-start-step",
        type=int,
        help="train-rl: resume a later RL block; the registered model must be at this step on the recorded RL checkpoint",
    )
    result.add_argument("--rollouts", type=int, default=4, help="train-rl: rollouts per task")
    result.add_argument(
        "--output-dir",
        type=Path,
        help="train-rl: attempt folder (default runs/<name>); relative paths resolve from the repo root",
    )
    result.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=MODEL",
        help="Additional served checkpoint for evaluate; repeat for SFT and RL",
    )
    result.add_argument(
        "--weak-clusters-ref", help="ARIA failure-clusters artifact to assess in final report"
    )
    result.add_argument(
        "--live",
        action="store_true",
        help="Doctor: check inference catalog with configured credentials",
    )
    result.add_argument("--benchmark-config", type=Path, default=ROOT / "config/forge_prompts.yaml")
    result.add_argument(
        "--split",
        nargs="+",
        choices=["train", "validation", "heldout", "test", "development"],
        default=["heldout"],
    )
    result.add_argument("--concurrency", type=int, default=2)
    result.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override model-specific inference token budget",
    )
    result.add_argument("--max-turns", type=int, default=100)
    result.add_argument(
        "--disable-thinking",
        action="store_true",
        default=None,
        help=(
            "Ask the model not to emit reasoning. run/demo fall back to BUFO_DISABLE_THINKING "
            "when absent; benchmark ignores the env var and records the flag in its frozen config"
        ),
    )
    result.add_argument("--publish", action="store_true", help="Publish native evaluation feedback")
    result.add_argument("--judge", choices=["scene", "astra"], default="scene")
    result.add_argument("--references", type=Path, help="Reference manifest directory or file")
    result.add_argument("--self-reference", action="store_true")
    result.add_argument("--baseline-arm")
    result.add_argument(
        "--confirmation",
        action="store_true",
        help=(
            "benchmark-score --judge astra only: allow judging heldout/test rows. For the single "
            "confirmation run only; never for tuning or calibration"
        ),
    )
    result.add_argument("--prompt-ids-file", type=Path, help="One prompt id per line")
    result.add_argument(
        "--generation-ids-file",
        type=Path,
        help="benchmark-score --judge astra only: one generation id per line; score only those rows",
    )
    result.add_argument(
        "--rubric",
        default="bufo-astra-v1",
        help="benchmark-score --judge astra only: rubric version (bufo-astra-v1 default, "
        "bufo-astra-v2 ... bufo-astra-v5); a non-default rubric writes to astra-<suffix>/",
    )
    result.add_argument(
        "--judge-samples",
        type=int,
        default=1,
        help="benchmark-score --judge astra only: independent judge calls per drawing, merged "
        "conservatively (a requirement passes only if every sample passes it); k>1 writes to "
        "astra-<suffix>-k<k>/",
    )
    result.add_argument("--reference-backend", default="closed_quality_bar")
    result.add_argument("--references-out", type=Path)
    result.add_argument(
        "--reviews",
        type=Path,
        action="append",
        default=[],
        help="Explicit review JSONL; repeatable",
    )
    result.add_argument("--source", type=Path, help="Reviewed trajectory JSONL for Forge import")
    result.add_argument("--include-validation", action="store_true")
    result.add_argument(
        "--forge-state",
        type=Path,
        default=None,
        help="Report: defaults to LOOP_ROOT/forge-state.json. Forge writes/status: FORGE_STATE_PATH or runs/forge-loop/forge-state.json.",
    )
    result.add_argument("--forge-alias", default=os.getenv("FORGE_PROJECT_ALIAS", "bufo-gen"))
    result.add_argument("--project-id", help="Existing Forge project ID")
    result.add_argument("--dataset-name")
    result.add_argument("--epochs", type=int, default=1)
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--learning-rate", type=float, help="forge-submit: explicit learning rate")
    result.add_argument(
        "--lora-rank",
        type=int,
        help="forge-submit: PROVISIONAL; Forge currently rejects the rank field (422), so omit it for rank 1",
    )
    result.add_argument(
        "--base-model",
        help="forge-submit: must equal the workflow's recorded base model; train-rl: student base "
        "(default: open_base)",
    )
    result.add_argument(
        "--tls12", action="store_true", help="Forge: verified TLS 1.2 upload workaround"
    )
    result.add_argument("--calibration", type=Path, help="Evidence-bound scene calibration JSON")
    result.add_argument(
        "--reward",
        choices=("scene", "astra"),
        default="scene",
        help="train-rl reward; astra needs an Astra calibration report and task references",
    )
    result.add_argument(
        "--initial-checkpoint", help="Immutable deployed SFT artifact for RL continuation"
    )
    result.add_argument("--forge-finetune-id", help="Actual deployed Forge fine-tune ID")
    result.add_argument("--loop-root", type=Path, default=ROOT / "runs/forge-loop")
    result.add_argument("--benchmark-dir", type=Path, action="append", default=[])
    result.add_argument("--rl-state", type=Path)
    return result


def forge_client(args):
    from bufogen.integrations.forge import DEFAULT_BASE_URL, DEFAULT_TRAINING_URL, ForgeClient
    from bufogen.settings import required

    return ForgeClient(
        api_key=required("WANDB_API_KEY"),
        entity=required("WANDB_ENTITY"),
        project_alias=args.forge_alias,
        base_url=os.getenv("FORGE_BASE_URL", DEFAULT_BASE_URL),
        training_base_url=os.getenv("FORGE_TRAINING_BASE_URL", DEFAULT_TRAINING_URL),
        tls12=args.tls12,
    )


def _checkpoint_template(configs, label):
    """Checkpoints serve like open_base, except one whose own models.yaml entry names another base.

    A label with no models.yaml entry is refused: cloning the Gemma template would silently serve
    a checkpoint of another base under the wrong provenance.
    """
    own = configs.get(label)
    if own is None:
        raise ValueError(f"Checkpoint label {label!r} needs a models.yaml entry (none found)")
    if own.base_model and own.base_model != configs["open_base"].model:
        return own
    return configs["open_base"]


def heldout_contract(args):
    from bufogen.evaluation.benchmark import load_benchmark

    rows = load_benchmark(args.benchmark_config)
    heldout = [p for p in rows if p["split"] in {"heldout", "test"}]
    if not heldout:
        raise ValueError("The benchmark contract must declare heldout/test families")
    return rows, {p["id"] for p in heldout}, {p["family"] for p in heldout}


def export_reviewed(args):
    from collections import Counter

    from bufogen.curation.curate import import_review_decisions, select_examples
    from bufogen.curation.judge_reviews import verify_decisions
    from bufogen.integrations.forge import reviewed_entries

    if not args.manifest or not args.reviews:
        raise ValueError("export-reviewed requires --manifest and --reviews")
    rows = Manifest(args.manifest).read()
    if not rows:
        raise ValueError("No existing generations in --manifest")
    prompts, heldout_ids, heldout_families = heldout_contract(args)
    contracts = {p["id"]: p for p in prompts}
    seen = set()
    for path in args.reviews:
        decisions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        verify_decisions(decisions, root=ROOT)
        ids = [d["source_generation_id"] for d in decisions]
        if seen.intersection(ids) or len(ids) != len(set(ids)):
            raise ValueError("Duplicate review decisions; resolve them explicitly before export")
        seen.update(ids)
        rows = import_review_decisions(rows, path)
    for row in rows:
        if row.id not in seen:
            row.training_review = None  # Only explicitly supplied decisions qualify this export.
        if row.training_review:
            # Current YAML must never override the source generation's frozen split.
            for contract in (contracts.get(row.prompt_id), row.benchmark_metadata):
                if contract is not None and (
                    row.training_review["partition"] != contract.get("split")
                    or row.training_review["prompt_family"] != contract.get("family")
                ):
                    raise ValueError(
                        "Review partition/family conflicts with frozen prompt provenance"
                    )
    diagnostics = []
    examples = select_examples(
        rows,
        heldout_ids=heldout_ids,
        heldout_families=heldout_families,
        diagnostics=diagnostics,
        allowed_partitions={"train", "validation"} if args.include_validation else {"train"},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "curation-diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    if not examples:
        raise ValueError(
            "No explicitly accepted, eligible trajectories; see curation-diagnostics.json"
        )
    reviewed_entries(examples, heldout_ids=heldout_ids, heldout_families=heldout_families)
    destination = args.output / "reviewed-examples.jsonl"
    text = "".join(json.dumps(e) + "\n" for e in examples)
    if destination.exists() and destination.read_text() != text:
        raise ValueError(
            "Reviewed output already exists with different content; use a new --output"
        )
    destination.write_text(text)
    summary = {
        "source_manifest": str(args.manifest.resolve()),
        "review_files": [str(p.resolve()) for p in args.reviews],
        "examples": len(examples),
        "partitions": dict(Counter(e["partition"] for e in examples)),
        "reviewer_kinds": dict(Counter(e["training_review"]["reviewer_kind"] for e in examples)),
        "excluded": len(diagnostics),
        "heldout_ids": sorted(heldout_ids),
        "heldout_families": sorted(heldout_families),
    }
    (args.output / "curation-summary.json").write_text(json.dumps(summary, indent=2))
    return destination


async def forge_dispatch(args):
    if args.stage == "export-reviewed":
        print(export_reviewed(args))
    elif args.stage == "loop-report":
        from bufogen.workflows.loop_report import write_loop_report

        print(
            write_loop_report(
                args.loop_root,
                args.output,
                benchmark_dirs=args.benchmark_dir or None,
                forge_state=args.forge_state,
                review_paths=args.reviews or None,
                calibration_path=args.calibration,
                rl_state=args.rl_state,
            )
        )
    elif args.stage == "build-references":
        from bufogen.evaluation.references import build_reference_manifest

        if args.references_out is None:
            raise ValueError("build-references requires --references-out")
        build_reference_manifest(args.output, args.references_out, backend=args.reference_backend)
        print(args.references_out / "references.json")
    elif args.stage in {"benchmark", "benchmark-score", "scene-review-queue"}:
        from bufogen.evaluation.benchmark import load_benchmark, run_benchmark, score_benchmark

        if args.stage == "benchmark":
            prompts = [
                p
                for p in load_benchmark(args.prompts or args.benchmark_config)
                if p["split"] in args.split
            ]
            benchmark_ids = None
            if args.prompt_ids_file is not None:
                benchmark_ids = [
                    line.strip()
                    for line in args.prompt_ids_file.read_text().splitlines()
                    if line.strip()
                ]
            configs = load_backends()
            chosen = {
                name: configs[name].require_configured()
                for name in (
                    args.backends
                    if args.backends is not None
                    else ([] if args.checkpoint else ["open_base", "closed_quality_bar"])
                )
            }
            for checkpoint in args.checkpoint:
                label, separator, model = checkpoint.partition("=")
                if not separator or not label or not model or label in chosen:
                    raise ValueError("--checkpoint requires a unique LABEL=MODEL")
                chosen[label] = replace(
                    _checkpoint_template(configs, label), name=label, model=model
                )
            await run_benchmark(
                prompts,
                chosen,
                args.output,
                limit=args.limit,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                max_turns=args.max_turns,
                disable_thinking=bool(args.disable_thinking),
                prompt_ids=benchmark_ids,
            )
            print(args.output / "manifest.jsonl")
        elif args.stage == "benchmark-score":
            prompt_ids = None
            if args.prompt_ids_file is not None:
                prompt_ids = [
                    line.strip()
                    for line in args.prompt_ids_file.read_text().splitlines()
                    if line.strip()
                ]
            generation_ids = None
            if args.generation_ids_file is not None:
                generation_ids = [
                    line.strip()
                    for line in args.generation_ids_file.read_text().splitlines()
                    if line.strip()
                ]
            await score_benchmark(
                args.output,
                concurrency=args.concurrency,
                publish=args.publish,
                judge=args.judge,
                references=args.references,
                self_reference=args.self_reference,
                baseline_arm=args.baseline_arm,
                limit_prompt_ids=prompt_ids,
                confirmation=args.confirmation,
                rubric_version=args.rubric,
                limit_generation_ids=generation_ids,
                judge_samples=args.judge_samples,
            )
            print(args.output / "comparison.html")
        else:
            from bufogen.integrations.observability import create_scene_review_queue

            rows = Manifest(args.manifest or args.output / "scored.jsonl").read()
            if not rows:
                raise ValueError("No scene-scored rows available for annotation queue")
            receipt = create_scene_review_queue(rows, name=args.name or "Bufo scene review")
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "review-queue.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt, indent=2))
    else:
        if args.forge_state is None:
            args.forge_state = Path(
                os.getenv("FORGE_STATE_PATH", str(ROOT / "runs/forge-loop/forge-state.json"))
            )
        from bufogen.workflows.forge_loop import (
            import_reviewed_dataset,
            initialize_workflow,
            resume_training,
            submit_training,
        )

        if args.stage == "forge-import":
            from bufogen.integrations.forge import reviewed_entries

            if not all([args.source, args.dataset_name, args.name, args.project_id]):
                raise ValueError(
                    "forge-import requires --source, --dataset-name, --name and --project-id"
                )
            _, ids, families = heldout_contract(args)
            examples = [
                json.loads(line) for line in args.source.read_text().splitlines() if line.strip()
            ]
            reviewed_entries(examples, heldout_ids=ids, heldout_families=families)
        with forge_client(args) as client:
            if args.stage == "forge-preflight":
                state = client.preflight(
                    project=project_path().split("/", 1)[1], project_id=args.project_id
                )
            elif args.stage == "forge-import":
                eligibility = client.preflight(
                    project=project_path().split("/", 1)[1], project_id=args.project_id
                )
                initialize_workflow(
                    args.forge_state,
                    client,
                    source_path=args.source,
                    dataset_name=args.dataset_name,
                    model_name=args.name,
                    project=project_path().split("/", 1)[1],
                    project_id=args.project_id,
                    heldout_ids=ids,
                    heldout_families=families,
                    lineage={"preflight": eligibility},
                )
                state = import_reviewed_dataset(args.forge_state, client)
            elif args.stage == "forge-submit":
                extra = {} if args.lora_rank is None else {"lora_rank": args.lora_rank}
                if args.learning_rate is not None:
                    extra["learning_rate"] = args.learning_rate
                if args.base_model is not None:
                    extra["base_model"] = args.base_model
                state = submit_training(
                    args.forge_state,
                    client,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    **extra,
                )
            else:
                state = resume_training(args.forge_state, client)
            print(json.dumps(state, indent=2))
    return 0


async def doctor(live=False):
    import os

    from bufogen.app.ui import public_error

    packages = ["agents", "cwsandbox", "weave", "wandb", "fastapi", "vtracer"]
    report = {
        "project": project_path(),
        "packages": {name: importlib.util.find_spec(name) is not None for name in packages},
        "credentials": {
            key: bool(os.getenv(key))
            for key in ["WANDB_API_KEY", "OPENAI_API_KEY", "CWSANDBOX_API_KEY", "HF_TOKEN"]
        },
        "sft_image_configured": bool(os.getenv("BUFO_SFT_IMAGE")),
        "models": {name: config.model or None for name, config in load_backends().items()},
    }
    if live:
        try:
            async with load_backends()["open_base"].client() as client:
                models = await client.models.list()
            report["catalog"] = [model.id for model in models.data]
            report["baseline_available"] = load_backends()["open_base"].model in report["catalog"]
        except Exception as exc:
            report["live_error"] = public_error(exc)
    print(json.dumps(report, indent=2))
    return 1 if "live_error" in report or not all(report["packages"].values()) else 0


async def dispatch(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    for name in ("concurrency", "max_tokens", "max_turns", "epochs", "batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.stage in {
        "forge-preflight",
        "benchmark",
        "benchmark-score",
        "build-references",
        "scene-review-queue",
        "export-reviewed",
        "forge-import",
        "forge-submit",
        "forge-status",
        "loop-report",
    }:
        return await forge_dispatch(args)
    if args.stage == "doctor":
        return await doctor(args.live)
    if args.stage == "sandbox-status":
        from bufogen.integrations.sandbox import sandbox_status

        if not args.sandbox_id:
            raise ValueError("--stage sandbox-status requires --sandbox-id.")
        print(json.dumps(await sandbox_status(args.sandbox_id), indent=2))
        return 0
    if args.stage == "render-smoke":
        from bufogen.rendering.renderer import Renderer

        renderer = Renderer()
        try:
            result = await renderer.render(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
                '<circle cx="256" cy="256" r="160" fill="#76b85c"/></svg>'
            )
            if not result.valid or not result.png_bytes:
                raise ValueError(result.error or "Sandbox returned no PNG.")
            args.output.mkdir(parents=True, exist_ok=True)
            path = args.output / "render-smoke.png"
            path.write_bytes(result.png_bytes)
            print(path)
        finally:
            await renderer.close()
        return 0
    if args.stage in {"run", "demo"}:
        configs = load_backends()
        names = args.backends or (
            [name for name in configs if name not in {"forge_proxy", "open_rl"}]
            if args.stage == "demo"
            else ["open_base", "closed_quality_bar"]
        )
        if args.stage == "demo":
            if not args.theme:
                raise ValueError("--stage demo requires --theme.")
            prompts = [{"id": "live-demo", "prompt": args.theme, "theme": "audience"}]
        else:
            prompts = load_prompts(args.prompts)
            if args.limit:
                prompts = prompts[: args.limit]
        rows = await run_pipeline(
            prompts,
            names,
            args.output,
            max_turns=args.max_turns,
            disable_thinking=args.disable_thinking,
            **({"max_tokens": args.max_tokens} if args.max_tokens is not None else {}),
        )
        print(args.output / "manifest.jsonl")
        if args.stage == "demo":
            print(write_demo(rows, args.output / "demo.html"))
    elif args.stage == "eval-identity":
        from bufogen.evaluation.suite import evaluate_identity_manifest

        await evaluate_identity_manifest(
            Manifest(args.manifest or args.output / "manifest.jsonl").read(),
            args.output / "identity-eval",
        )
        print(args.output / "identity-eval/summary.json")
    elif args.stage == "review-queue":
        from bufogen.integrations.observability import create_review_queue

        rows = Manifest(args.manifest or args.output / "manifest.jsonl").read()
        if not rows:
            raise ValueError("No generations to review. Provide a populated --manifest.")
        result = create_review_queue(rows, name=args.name or "Bufo identity calibration v1")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "review-queue.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    elif args.stage == "eval":
        from bufogen.evaluation.suite import evaluate_manifest

        await evaluate_manifest(
            Manifest(args.manifest or args.output / "manifest.jsonl").read(), args.output / "eval"
        )
        print(args.output / "eval/summary.json")
    elif args.stage in {"curate", "publish-prompts"}:
        from bufogen.integrations.registry import RegistryClient

        if args.stage == "publish-prompts":
            print(RegistryClient().push_prompts(sorted((ROOT / "aria_prompts").glob("*.md"))))
        else:
            from bufogen.curation.curate import curate

            if not args.prompts_ref:
                raise ValueError("Provide --prompts-ref from the ARIA curation task.")
            print(
                await curate(
                    prompts_ref=args.prompts_ref,
                    backup_ref=args.backup_ref,
                    output_dir=args.output,
                    heldout_ids={p["id"] for p in load_prompts(args.heldout_prompts)},
                )
            )
    elif args.stage in {"train-sft", "train-rl"}:
        if not args.dataset_ref or not args.name:
            raise ValueError("Training requires --dataset-ref and --name.")
        if args.stage == "train-sft":
            from bufogen.training.sft import SFTTrainer as Trainer
        else:
            from bufogen.training.rl import RLTrainer as Trainer
        options = {}
        if args.stage == "train-rl":
            if not args.calibration:
                raise ValueError("train-rl requires --calibration from reviewed scene evidence")
            if bool(args.initial_checkpoint) != bool(args.forge_finetune_id):
                raise ValueError(
                    "SFT continuation requires both --initial-checkpoint and --forge-finetune-id"
                )
            options = {
                "calibration_path": args.calibration,
                "initial_checkpoint": args.initial_checkpoint,
                "forge_finetune_id": args.forge_finetune_id,
                "forge_project_alias": args.forge_alias,
                "reward_kind": args.reward,
                # Rollouts must use the evaluation's agent configuration.
                "max_tokens": args.max_tokens if args.max_tokens is not None else 6000,
                "disable_thinking": bool(args.disable_thinking),
                "tasks_per_step": args.tasks_per_step,
                "rollouts": args.rollouts,
                "output_dir": args.output_dir,
                "expect_start_step": args.expect_start_step,
            }
        base_model = load_backends()["open_base"].model
        if args.stage == "train-rl" and args.base_model:
            base_model = args.base_model  # RLTrainer rejects anything outside forge.STUDENT_MODELS
        trainer = Trainer(
            args.dataset_ref,
            name=args.name,
            base_model=base_model,
            **options,
        )
        print(await trainer.train(steps=args.steps))
    elif args.stage == "evaluate":
        from bufogen.evaluation.report import evaluate_checkpoints
        from bufogen.integrations.registry import RegistryClient

        configs = load_backends()
        names = args.backends or ["open_base", "open_tuned", "closed_quality_bar"]
        chosen = {name: configs[name] for name in names}
        for checkpoint in args.checkpoint:
            label, separator, model = checkpoint.partition("=")
            if not separator or not label or not model or label in chosen:
                raise ValueError("--checkpoint requires a unique LABEL=MODEL.")
            chosen[label] = replace(configs["open_base"], name=label, model=model)
        themes = set()
        if args.weak_clusters_ref:
            folder = RegistryClient().download(args.weak_clusters_ref)
            themes = {
                c["theme"] for c in json.loads((folder / "clusters.json").read_text())["clusters"]
            }
        prompts = load_prompts(args.prompts or args.heldout_prompts)
        if args.limit:
            prompts = prompts[: args.limit]
        print(await evaluate_checkpoints(prompts, chosen, args.output, weak_themes=themes))
    elif args.stage == "build-backup-dataset":
        from bufogen.curation.backup import build_backup_dataset

        print(await build_backup_dataset(args.output, limit=args.limit or 12))
    return 0


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parser().parse_args()
    try:
        result = asyncio.run(dispatch(args))
    except (ValueError, KeyError, ModuleNotFoundError, RuntimeError, OSError) as exc:
        from bufogen.app.ui import public_error

        print(f"Error: {public_error(exc)}", file=sys.stderr)
        result = 1
    raise SystemExit(result)


if __name__ == "__main__":
    main()
