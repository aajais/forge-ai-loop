"""Offline checks for complete, hash-bound v2 visualization evidence."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "docs/forge-contact-pilot/publish_v2.py"


def publisher():
    assert PATH.exists(), "v2 publisher is not implemented"
    name = "contact_publisher_v2_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, values):
    path.write_text("".join(json.dumps(v) + "\n" for v in values))


@pytest.fixture
def packets(tmp_path):
    p = publisher()
    folder = tmp_path / "visual-review"
    (folder / "images").mkdir(parents=True)
    for name in ("benchmark.json", "manifest.jsonl"):
        (tmp_path / name).write_text("{}\n")
    rows, key, inputs, decisions = [], [], [[], [], []], [[], [], []]
    for i in range(32):
        original = tmp_path / f"original-{i}.png"
        neutral = folder / "images" / f"image-{i + 1:02d}.png"
        if i:
            original.write_bytes(f"synthetic PNG fixture {i}".encode())
            neutral.write_bytes(original.read_bytes())
        row = {
            "id": f"id-{i}",
            "prompt_id": f"p-{i // 2}",
            "backend": p.ARMS[i % 2],
            "model": p.MODELS[i % 2],
            "prompt": f"prompt {i // 2}",
            "png_path": str(original) if i else None,
            "valid": bool(i),
            "benchmark_metadata": {
                "requirements": ["visible grip"],
                "png_sha256": sha(original) if i else None,
                "status": "completed" if i else "generation_error",
            },
        }
        rows.append(row)
        key.append({"index": i, **{k: row[k] for k in ("id", "prompt_id", "backend", "model")}})
        item = {
            "index": i,
            "label": f"Image {i + 1:02d}",
            "prompt": row["prompt"],
            "requirements": ["visible grip"],
            "png_path": str(neutral) if i else None,
            "png_sha256": sha(original) if i else None,
        }
        inputs[i % 3].append(item)
        decisions[i % 3].append(
            {k: item[k] for k in ("index", "label", "png_sha256")}
            | {
                "reviewer_kind": "assistant",
                "passed": bool(i),
                "evidence": "Synthetic review.",
                "failed_requirements": [] if i else ["visible grip"],
            }
        )
    save(folder / "unblinding-key.json", key)
    save(
        folder / "protocol.json",
        {
            "version": "neutral-visual-review-v1",
            "attempts": 32,
            "reviewers": 3,
            "benchmark_sha256": sha(tmp_path / "benchmark.json"),
            "manifest_sha256": sha(tmp_path / "manifest.jsonl"),
            "packet_source_sha256": sha(p.HERE / "neutral_review_packets.py"),
            "blinding": "Model identities and original image paths omitted from reviewer inputs.",
        },
    )
    for reviewer in range(3):
        d = folder / f"reviewer-{reviewer}"
        save(d / "inputs.json", inputs[reviewer])
        jsonl(d / "decisions.jsonl", decisions[reviewer])
        (d / "review-notes.md").write_text("Synthetic review limitations.")
        for sheet in range(1, (len(inputs[reviewer]) + 3) // 4 + 1):
            (d / f"sheet-{sheet:02d}.png").write_bytes(b"synthetic sheet")
    return tmp_path, rows


def test_complete_neutral_reviews_bind_original_capture_and_every_review_file(packets):
    p = publisher()
    directory, rows = packets
    reviews, files, status = p.validate_reviews(directory, rows)
    assert len(reviews) == 32 and reviews["id-0"]["passed"] is False
    assert status["status"] == "complete" and status["reviewer_kind"] == "assistant"
    assert any("reviewer-2/review-notes.md" in name for name in files)
    assert len([name for name in files if "/sheet-" in name]) == 9
    assert len([name for name in files if "/images/image-" in name]) == 31


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "wrong_index",
        "wrong_identity",
        "nonbool",
        "invented_requirement",
        "missing_image_pass",
        "omitted_missing_requirements",
        "changed_copy",
        "changed_original",
        "changed_capture_hash",
        "original_path",
        "changed_protocol",
        "missing_notes",
        "missing_sheet",
    ],
)
def test_neutral_review_tampering_and_incomplete_packets_fail_closed(packets, mutation):
    p = publisher()
    directory, rows = packets
    folder = directory / "visual-review"
    path = folder / "reviewer-0/decisions.jsonl"
    decisions = p.jsonl(path)
    if mutation == "missing":
        decisions.pop()
    elif mutation == "duplicate":
        decisions.append(decisions[0])
    elif mutation == "wrong_index":
        decisions[0]["index"] = 1
    elif mutation == "wrong_identity":
        key = p.read(folder / "unblinding-key.json")
        key[0]["model"] = "other"
        save(folder / "unblinding-key.json", key)
    elif mutation == "nonbool":
        decisions[1]["passed"] = 1
    elif mutation == "invented_requirement":
        decisions[1]["failed_requirements"] = ["invented"]
    elif mutation == "missing_image_pass":
        decisions[0].update(passed=True, failed_requirements=[])
    elif mutation == "omitted_missing_requirements":
        decisions[0]["failed_requirements"] = []
    elif mutation == "changed_copy":
        (folder / "images/image-02.png").write_bytes(b"changed")
    elif mutation == "changed_original":
        Path(rows[1]["png_path"]).write_bytes(b"changed")
    elif mutation == "changed_capture_hash":
        rows[1]["benchmark_metadata"]["png_sha256"] = "0" * 64
    elif mutation == "original_path":
        f = folder / "reviewer-0/inputs.json"
        data = p.read(f)
        data[1]["png_path"] = rows[3]["png_path"]
        save(f, data)
    elif mutation == "changed_protocol":
        f = folder / "protocol.json"
        data = p.read(f)
        data["manifest_sha256"] = "0" * 64
        save(f, data)
    elif mutation == "missing_notes":
        (folder / "reviewer-1/review-notes.md").unlink()
    elif mutation == "missing_sheet":
        (folder / "reviewer-2/sheet-01.png").unlink()
    jsonl(path, decisions)
    with pytest.raises(ValueError):
        p.validate_reviews(directory, rows)


def publication_data(packets):
    p = publisher()
    directory, rows = packets
    reviews, files, status = p.validate_reviews(directory, rows)
    for i, row in enumerate(rows):
        row.update(
            input_tokens=10,
            output_tokens=20,
            latency_s=2.0,
            tool_calls=1,
            error=None if i else "synthetic failure",
            prompt_version="frozen",
            scores={
                "scene_quality": {
                    "status": "ok" if i else "missing_image",
                    "quality_failure": False if i else True,
                    "prompt_checks": [],
                    "constraint_coverage": 1.0 if i else None,
                }
            },
        )
    # Deliberate judge/reviewer disagreement; conjunction must remain false.
    rows[1]["scores"]["scene_quality"]["quality_failure"] = True
    return {
        "rows": rows,
        "reviews": reviews,
        "review_status": status,
        "files": files,
        "hashes": {name: sha(path) for name, path in files.items()},
        "prompts": [
            {"id": f"p-{i}", "prompt": f"prompt {i}", "requirements": ["visible grip"]}
            for i in range(16)
        ],
        "configuration_hash": "frozen-config",
        "summary": {},
    }


def test_default_publication_is_offline_and_does_not_create_receipt(monkeypatch, packets):
    p = publisher()
    data = publication_data(packets)
    monkeypatch.setattr(p, "validate", lambda _: data)
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **_: pytest.fail("network attempted"))
    )
    result = p.publish(packets[0])
    assert result["network_calls"] == 0 and result["attempts"] == 32 and result["paired_rows"] == 16
    assert not (packets[0] / "visualization.json").exists()


def test_tables_account_for_all_attempts_and_receipt_prevents_duplicate_upload(
    monkeypatch, packets
):
    p = publisher()
    data = publication_data(packets)
    calls = []
    logged = {}
    monkeypatch.setattr(p, "validate", lambda _: data)
    monkeypatch.setattr(p, "project_path", lambda: "synthetic/project")

    class Table:
        def __init__(self, columns):
            self.columns, self.rows = columns, []

        def add_data(self, *values):
            assert len(values) == len(self.columns)
            self.rows.append(dict(zip(self.columns, values, strict=True)))

    class Run:
        url = "synthetic://run"

        def __init__(self):
            self.summary = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def log(self, items):
            logged.update(items)

        def log_artifact(self, artifact):
            return SimpleNamespace(name="v2:v0", wait=lambda: None)

    def init(**kwargs):
        json.dumps(kwargs["config"], allow_nan=False)
        calls.append(kwargs)
        return Run()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Table=Table,
            Image=lambda path: path,
            init=init,
            Artifact=lambda *_, **__: SimpleNamespace(add_file=lambda *_, **__: None),
        ),
    )
    receipt = p.publish(packets[0], execute=True)
    assert receipt["status"] == "published" and receipt["fine_tuning_gain_claim"] is False
    assert len(logged["paired_images"].rows) == 16 and len(logged["all_32_attempts"].rows) == 32
    attempts = {r["generation_id"]: r for r in logged["all_32_attempts"].rows}
    assert (
        attempts["id-0"]["combined_pass"] is False
        and attempts["id-0"]["generation_error"] == "synthetic failure"
    )
    assert attempts["id-1"]["assistant_pass"] is True and attempts["id-1"]["combined_pass"] is False
    assert sum(r["attempts"] for r in logged["model_summary"].rows) == 32
    assert p.publish(packets[0], execute=True) == receipt and len(calls) == 1
    files = data["files"]
    Path(next(iter(files.values()))).write_text("changed evidence")
    with pytest.raises(ValueError):
        p.publish(packets[0], execute=True)
    assert len(calls) == 1


def test_uncertain_receipt_blocks_retry_before_sdk_import(monkeypatch, packets):
    p = publisher()
    data = publication_data(packets)
    monkeypatch.setattr(p, "validate", lambda _: data)
    save(
        packets[0] / "visualization.json",
        {"status": "publishing", "evidence_sha256": data["hashes"], "run_id": "uncertain"},
    )
    with pytest.raises(RuntimeError, match="reconcile"):
        p.publish(packets[0], execute=True)


@pytest.fixture
def attempts(tmp_path):
    from bufogen.evaluation import benchmark
    from bufogen.evaluation.scorers.scene_quality import failure_result
    from bufogen.generation import BufoGeneration

    p = publisher()
    prompts = [
        {
            "id": f"p-{i}",
            "prompt": f"prompt {i}",
            "family": f"family-{i}",
            "split": "development",
            "requirements": ["visible grip"],
        }
        for i in range(16)
    ]
    config = {
        "prompts": prompts,
        "arms": [{"name": a, "model": m} for a, m in zip(p.ARMS, p.MODELS)],
        "scorer": {"threshold": 0.65},
        "prompt_version": "frozen",
    }
    settings = {
        "configuration": config,
        "configuration_hash": "frozen-hash",
        "run_id": "test-run",
        "comparison_fingerprint": "test-fingerprint",
    }
    originals = []
    scored = []
    for i in range(32):
        prompt = prompts[i // 2]
        row = BufoGeneration(
            id=f"id-{i}",
            prompt_id=prompt["id"],
            prompt=prompt["prompt"],
            backend=p.ARMS[i % 2],
            model=p.MODELS[i % 2],
            group_id=f"group-{i}",
            svg="",
            valid=False,
            latency_s=2.0,
            error="synthetic failed request",
            prompt_version="frozen",
            benchmark_metadata={
                "configuration_hash": "frozen-hash",
                "run_id": "test-run",
                "status": "generation_error",
                **{k: prompt[k] for k in ("requirements", "family", "split")},
            },
        )
        originals.append(row.to_dict())
        row.scores = {
            "scene_quality": failure_result("missing_image")
            | {
                "generation_id": row.id,
                "scorer_configuration": config["scorer"],
                "requirements": prompt["requirements"],
            }
        }
        scored.append(row.to_dict())
    jsonl(tmp_path / "manifest.jsonl", originals)
    jsonl(tmp_path / "scored.jsonl", scored)
    save(
        tmp_path / "summary.json",
        benchmark._summary([BufoGeneration(**row) for row in scored], settings),
    )
    save(
        tmp_path / "weave-publication.json",
        {
            "feedback": {row["id"]: {"call_id": f"fallback-{i}"} for i, row in enumerate(scored)},
            "evaluations": {
                arm: {"ref": f"weave:///synthetic/{arm}", "scored_rows": 16} for arm in p.ARMS
            },
            "errors": {},
        },
    )
    return tmp_path, settings


def test_all_failed_attempts_keep_native_fallback_feedback_coverage(attempts):
    p = publisher()
    directory, settings = attempts
    rows, summary, _ = p.validate_attempts(directory, settings)
    assert (
        len(rows) == 32 and sum(arm["generation_errors"] for arm in summary["arms"].values()) == 32
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "omit_attempt",
        "duplicate_attempt",
        "alter_generation",
        "incomplete_feedback",
        "duplicate_feedback",
        "wrong_known_call",
        "incomplete_evaluation",
        "false_failure_pass",
        "wrong_scorer",
        "alter_summary",
    ],
)
def test_generation_score_and_native_weave_corruption_rejected(attempts, mutation):
    p = publisher()
    directory, settings = attempts
    path = directory / "scored.jsonl"
    rows = p.jsonl(path)
    pub = p.read(directory / "weave-publication.json")
    if mutation == "omit_attempt":
        rows.pop()
    elif mutation == "duplicate_attempt":
        rows[-1] = rows[0]
    elif mutation == "alter_generation":
        rows[0]["latency_s"] = 999
    elif mutation == "incomplete_feedback":
        pub["feedback"].pop("id-0")
    elif mutation == "duplicate_feedback":
        pub["feedback"]["id-1"] = pub["feedback"]["id-0"]
    elif mutation == "wrong_known_call":
        originals = p.jsonl(directory / "manifest.jsonl")
        originals[0]["weave_call_id"] = "original-call"
        jsonl(directory / "manifest.jsonl", originals)
        rows[0]["weave_call_id"] = "original-call"
    elif mutation == "incomplete_evaluation":
        pub["evaluations"][p.ARMS[0]]["scored_rows"] = 15
    elif mutation == "false_failure_pass":
        rows[0]["scores"]["scene_quality"]["quality_failure"] = False
    elif mutation == "wrong_scorer":
        rows[0]["scores"]["scene_quality"]["scorer_configuration"] = {}
    elif mutation == "alter_summary":
        save(directory / "summary.json", {})
    jsonl(path, rows)
    save(directory / "weave-publication.json", pub)
    with pytest.raises(ValueError):
        p.validate_attempts(directory, settings)
