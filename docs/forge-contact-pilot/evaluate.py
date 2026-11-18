"""Frozen old/new interface pilot; default is offline preparation, --run is paid."""
import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from bufogen.evaluation import benchmark
from bufogen.models import load_backends
from bufogen.settings import ROOT

HERE = Path(__file__).resolve().parent
POLICY = dict(concurrency=8, temperature=0.7, max_tokens=6000, max_turns=8)
INTERFACES = ("svg", "contact")
BASE, TEACHER = "google/gemma-4-26B-A4B-it", "gpt-6.1-sol"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_prompts():
    prompts = benchmark._validate(read(HERE / "prompts.json"))
    if (
        len(prompts) != 16
        or sorted(Counter(p["family"] for p in prompts).values()) != [4, 4, 4, 4]
        or any(p["split"] != "development" for p in prompts)
    ):
        raise ValueError("Expected 16 development prompts in four balanced relation families")
    return prompts


def freeze_protocol(directory, protocol):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with benchmark._locked(directory):
        path = directory / "pilot-protocol.json"
        if path.exists():
            if read(path) != protocol:
                raise ValueError("Pilot protocol changed; use a new experiment directory")
        elif any(p.name != ".benchmark.lock" for p in directory.iterdir()):
            raise ValueError("Cannot adopt existing outputs without a frozen protocol")
        else:
            benchmark._write_json(path, protocol)


def verify_current_configuration(prompts, arms, frozen):
    if benchmark._configuration(prompts, arms, **POLICY) != frozen:
        raise ValueError("Live interface configuration changed before inference")


def verify_fixture_dependencies(rows):
    current = benchmark._source_hashes()
    if any(row.get("source_hashes") != current for row in rows):
        raise ValueError("Fixture assets or rendering dependencies changed after review")


def validate_fixture_review(directory, compiler_sha):
    directory = Path(directory)
    review_path = directory / "assistant-review.json"
    review = read(review_path)
    if review.get("compiler_sha256") != compiler_sha:
        raise ValueError("Fixture review compiler differs from the current implementation")
    if (
        review.get("reviewer_kind") != "assistant"
        or review.get("accepted_for_interface_test") is not True
    ):
        raise ValueError("Fixtures need explicit assistant acceptance before model inference")
    rows = read(directory / "manifest.json")
    verify_fixture_dependencies(rows)
    decisions = review.get("fixtures", [])
    if (
        len(rows) != 12 or len(decisions) != 12
        or len({r["id"] for r in rows}) != 12
        or {r["id"] for r in rows} != {d["id"] for d in decisions}
    ):
        raise ValueError("Require all twelve fixture renders and reviews")
    by_id = {d["id"]: d for d in decisions}
    for row in rows:
        decision = by_id[row["id"]]
        if (
            row["compiler_sha256"] != compiler_sha
            or sha(row["png_path"]) != row["png_sha256"]
            or sha(row["scene_path"]) != row["scene_sha256"]
            or sha(row["svg_path"]) != row["svg_sha256"]
            or decision.get("png_sha256") != row["png_sha256"]
            or decision.get("accepted_for_interface_test") is not True
            or not decision.get("evidence")
        ):
            raise ValueError("Fixture evidence or acceptance changed")
    return {"manifest_sha256": sha(directory / "manifest.json"), "review_sha256": sha(review_path)}


@contextmanager
def interface(name):
    """Sequential process-local adapter; frozen files and config identify the real runner."""
    if name not in INTERFACES:
        raise ValueError("Unknown interface")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import contact_agent

    names = ("BufoAgent", "instructions", "prompt_version", "_source_hashes")
    originals = {key: getattr(benchmark, key) for key in names}

    def sources():
        paths = ["evaluate.py", "contact_agent.py", "scene_renderer.py", "prompts.json"]
        return originals["_source_hashes"]() | contact_agent.source_hashes() | {
            str((HERE / filename).relative_to(ROOT)): sha(HERE / filename)
            for filename in paths
        }

    try:
        benchmark._source_hashes = sources
        if name == "contact":
            benchmark.BufoAgent = contact_agent.ContactAgent
            benchmark.instructions = contact_agent.instructions
            benchmark.prompt_version = contact_agent.prompt_version
        yield
    finally:
        for key, value in originals.items():
            setattr(benchmark, key, value)


def prepare(root):
    prompts = load_prompts()
    backends = load_backends()
    arms = [
        replace(backends["open_base"], model=BASE, label=BASE),
        replace(backends["closed_quality_bar"], model=TEACHER, label=TEACHER),
    ]
    fixtures = validate_fixture_review(root / "fixtures", sha(HERE / "scene_renderer.py"))
    configs = {}
    for name in INTERFACES:
        with interface(name):
            configs[name] = benchmark._configuration(prompts, arms, **POLICY)
    return prompts, arms, {
        "version": "contact-interface-pilot-v1",
        "purpose": "Development diagnostic; interface effect, not a fine-tuning gain.",
        "attempts": 64,
        "fixture_evidence": fixtures,
        "configurations": configs,
        "policy": POLICY,
        "confirmation_outputs_used": False,
    }


def summarize(root, protocol):
    summaries = {}
    for name in INTERFACES:
        directory = root / name
        settings = read(directory / "benchmark.json")
        if settings["configuration"] != protocol["configurations"][name]:
            raise ValueError("Interface configuration changed")
        rows = [json.loads(line) for line in (directory / "scored.jsonl").read_text().splitlines()]
        expected = {
            (p["id"], a["name"])
            for p in settings["configuration"]["prompts"]
            for a in settings["configuration"]["arms"]
        }
        if len(rows) != 32 or {(r["prompt_id"], r["backend"]) for r in rows} != expected:
            raise ValueError("Incomplete pilot; every requested attempt must be scored")
        from bufogen.generation import BufoGeneration
        values = [BufoGeneration(**row) for row in rows]
        benchmark._validate_rows(values, settings)
        summary = benchmark._summary(values, settings)
        if summary != read(directory / "summary.json"):
            raise ValueError("Summary differs from saved attempts")
        summaries[name] = summary
    result = {
        "purpose": protocol["purpose"],
        "protocol_sha256": sha(root / "pilot-protocol.json"),
        "summaries": summaries,
        "note": "All attempts counted. Same image scorer; assistant visual review also required.",
    }
    benchmark._write_json(root / "comparison.json", result)
    return result


async def run(root):
    prompts, arms, protocol = prepare(root)
    freeze_protocol(root / "comparison", protocol)
    output = root / "comparison"
    from bufogen.integrations.telemetry import init_weave
    client = init_weave()
    try:
        for name in INTERFACES:
            if validate_fixture_review(root / "fixtures", sha(HERE / "scene_renderer.py")) != protocol["fixture_evidence"]:
                raise ValueError("Fixture review changed before inference")
            with interface(name):
                verify_current_configuration(prompts, arms, protocol["configurations"][name])
                await benchmark.run_benchmark(prompts, arms, output / name, **POLICY)
                await benchmark.score_benchmark(output / name, concurrency=8)
        return summarize(output, protocol)
    finally:
        client.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/forge-contact-pilot")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        print(json.dumps(asyncio.run(run(args.root.resolve())), indent=2))
    else:
        prompts, arms, protocol = prepare(args.root.resolve())
        print(json.dumps({"offline_validation": "passed", "prompts": len(prompts),
                          "models": [a.model for a in arms], "attempts": protocol["attempts"]}))


if __name__ == "__main__":
    main()
