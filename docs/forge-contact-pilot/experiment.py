"""Run a frozen contact-v2 benchmark; default validates without paid calls."""

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

from bufogen.evaluation import benchmark
from bufogen.models import load_backends
from bufogen.settings import ROOT

HERE = Path(__file__).resolve().parent
POLICY = dict(concurrency=8, temperature=0.7, max_tokens=6000, max_turns=8)
BASE = "google/gemma-4-26B-A4B-it"
TEACHER = "gpt-6.1-sol"
FIXTURES = ROOT / "runs/forge-contact-pilot/fixtures"


def require_checkpoint(checkpoint):
    if not isinstance(checkpoint, str) or not re.fullmatch(
        r"wandb-artifact:///[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+:v[0-9]+",
        checkpoint,
    ):
        raise ValueError("Require an immutable checkpoint: wandb-artifact:///entity/project/artifact:vN")
    return checkpoint


@contextmanager
def harness():
    """Only in this isolated CLI process; restore the production runner afterward."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import contact_agent_v2 as agent

    names = ("BufoAgent", "instructions", "prompt_version", "_source_hashes")
    original = {name: getattr(benchmark, name) for name in names}

    def sources():
        return original["_source_hashes"]() | agent.source_hashes() | {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), HERE / "evaluate.py")
        }

    try:
        benchmark.BufoAgent = agent.ContactAgent
        benchmark.instructions = agent.instructions
        benchmark.prompt_version = agent.prompt_version
        benchmark._source_hashes = sources
        yield
    finally:
        for name, value in original.items():
            setattr(benchmark, name, value)


def prepare(prompts_path, names, checkpoint=None):
    if not names or len(set(names)) != len(names):
        raise ValueError("Choose distinct model arms")
    backends = load_backends()
    models = {"open_base": BASE, "closed_quality_bar": TEACHER}
    if "open_tuned" in names:
        models["open_tuned"] = require_checkpoint(checkpoint)
    elif checkpoint:
        raise ValueError("Checkpoint supplied without a fine-tuned arm")
    arms = [
        replace(
            backends[name], model=models[name],
            label=f"ft-{BASE}" if name == "open_tuned" else models[name],
            base_model=BASE if name == "open_tuned" else backends[name].base_model,
        )
        for name in names
    ]
    prompts = benchmark.load_benchmark(Path(prompts_path))
    # Preserve the already accepted compiler and rendering dependencies.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from evaluate import sha, validate_fixture_review
    fixtures = validate_fixture_review(
        FIXTURES, sha(HERE / "scene_renderer.py")
    )
    with harness():
        configuration = benchmark._configuration(prompts, arms, **POLICY)
    return prompts, arms, configuration, fixtures


def freeze_protocol(output, configuration, fixtures):
    """Freeze reviewed dependencies before tracing or paid requests; resume exactly."""
    output = Path(output)
    verifier = HERE / "evaluate.py"
    sources = {
        "fixture-manifest.json": (FIXTURES / "manifest.json", fixtures["manifest_sha256"]),
        "fixture-assistant-review.json": (FIXTURES / "assistant-review.json", fixtures["review_sha256"]),
        "fixture-verifier.py": (
            verifier, configuration["source_hashes"][str(verifier.relative_to(ROOT))]
        ),
    }
    # Read and verify all bytes before creating any protocol evidence.
    contents = {}
    for name, (path, expected_hash) in sources.items():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("Fixture review or verifier changed after preparation")
        contents[name] = data
    protocol = {
        "schema_version": 1,
        "configuration": configuration,
        "configuration_hash": benchmark.digest(configuration),
        "fixture_evidence": fixtures,
        "evidence_sha256": {name: expected for name, (_, expected) in sources.items()},
    }
    with benchmark._locked(output):
        path = output / "experiment-protocol.json"
        snapshot = output / "protocol-evidence"
        if path.exists():
            if json.loads(path.read_text()) != protocol:
                raise ValueError("Experiment protocol changed; use a new output directory")
            for name, data in contents.items():
                frozen_path = snapshot / name
                if not frozen_path.is_file() or frozen_path.read_bytes() != data:
                    raise ValueError("Frozen fixture evidence changed or is missing")
        else:
            if any(item.name != ".benchmark.lock" for item in output.iterdir()):
                raise ValueError("Cannot adopt nonempty experiment outputs without a protocol")
            snapshot.mkdir()
            for name, data in contents.items():
                (snapshot / name).write_bytes(data)
            benchmark._write_json(path, protocol)
    return protocol


async def run(args):
    prompts, arms, frozen, fixtures = prepare(args.prompts, args.arm, args.checkpoint)
    freeze_protocol(args.output, frozen, fixtures)
    from bufogen.integrations.telemetry import init_weave
    client = init_weave()
    try:
        with harness():
            if benchmark._configuration(prompts, arms, **POLICY) != frozen:
                raise ValueError("Harness changed before inference")
            await benchmark.run_benchmark(prompts, arms, args.output, **POLICY)
            await benchmark.score_benchmark(args.output, concurrency=8)
        return {"output": str(args.output), "fixture_evidence": fixtures,
                "summary": json.loads((args.output / "summary.json").read_text())}
    finally:
        client.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", action="append",
                        choices=["open_base", "closed_quality_bar", "open_tuned"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    args.arm = args.arm or ["open_base", "closed_quality_bar"]
    args.output = args.output.resolve()
    if args.run:
        result = asyncio.run(run(args))
    else:
        prompts, arms, config, fixtures = prepare(args.prompts, args.arm, args.checkpoint)
        result = {
            "offline_validation": "passed", "network_calls": 0,
            "attempts": len(prompts) * len(arms),
            "models": [arm.display_name for arm in arms],
            "configuration_hash": benchmark.digest(config), "fixture_evidence": fixtures,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
