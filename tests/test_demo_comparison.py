"""Contract tests for offline, failure-inclusive comparison and safe publication."""

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bufogen.workflows import demo_comparison as demo

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(tmp_path):
    prompts = [
        dict(
            id=f"p{i}",
            prompt=f"Draw {i}",
            requirements=["Visible"],
            family=f"f{i}",
            split="validation",
        )
        for i in range(2)
    ]
    judge = dict(rubric_hash="rubric", judge_model="gpt-6-astra", samples=3)
    refs = tmp_path / "refs.json"
    write(refs, {"rows": {p["id"]: {"png_sha256": "reference"} for p in prompts}})
    arms = []
    for name in ["base", "sft"]:
        directory = tmp_path / name
        model = (
            "Qwen/Qwen3.8-27B" if name == "base" else "wandb-artifact:///entity/project/model:v1"
        )
        arm = dict(
            name=name,
            model=model,
            base_model="Qwen/Qwen3.8-27B",
            tuning_method="" if name == "base" else "sft",
            sampling={"max_tokens": 20000},
        )
        config = dict(
            prompts=prompts,
            arms=[arm],
            prompt_version="v1",
            runtime={"instructions_hash": "same"},
            source_hashes={"agent.py": "same"},
            policy={"max_tokens": 20000},
        )
        write(
            directory / "benchmark.json",
            {"configuration": config, "configuration_hash": "hash", "run_id": name},
        )
        write(
            directory / "astra/scorer.json",
            {
                "judge": judge,
                "references": {
                    "path": str(refs),
                    "sha256": hashlib.sha256(refs.read_bytes()).hexdigest(),
                },
                "self_reference": False,
            },
        )
        rows = []
        for i, prompt in enumerate(prompts):
            png = directory / f"{i}.png"
            png.write_bytes(PNG)
            rows.append(
                dict(
                    id=f"{name}-{i}",
                    prompt_id=prompt["id"],
                    prompt=prompt["prompt"],
                    backend=name,
                    model=model,
                    group_id="g",
                    svg="",
                    valid=True,
                    latency_s=10 + i,
                    png_path=str(png),
                    prompt_version="v1",
                    benchmark_metadata=dict(
                        run_id=name,
                        configuration_hash="hash",
                        requirements=prompt["requirements"],
                        family=prompt["family"],
                        split="validation",
                        status="completed",
                        png_sha256=hashlib.sha256(PNG).hexdigest(),
                    ),
                    scores={
                        "astra_judge": dict(
                            status="ok",
                            strict_pass=True,
                            scorer_configuration=judge,
                            generation_id=f"{name}-{i}",
                            reference={"png_sha256": "reference", "is_self": False},
                            requirement_coverage=1.0,
                        )
                    },
                )
            )
        (directory / "astra/scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        arms.append(
            {
                "id": name,
                "source": str(directory),
                "scored_dir": "astra",
                "backend": name,
                "model_id": model,
                "role": "matched",
                "label": name,
            }
        )
    manifest = {"version": 1, "expected_prompts": 2, "arms": arms}
    path = tmp_path / "manifest.json"
    write(path, manifest)
    return path


def mutate_rows(manifest, arm, mutation):
    config = json.loads(manifest.read_text())["arms"][arm]
    path = Path(config["source"]) / "astra/scored.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_failure_rows_stay_in_denominator_and_missing_png_is_visible(tmp_path):
    manifest = fixture(tmp_path)

    def fail(rows):
        rows[1]["valid"] = False
        rows[1]["error"] = "No drawing"
        rows[1]["png_path"] = None
        rows[1]["scores"]["astra_judge"].update(status="missing_image", strict_pass=False)

    mutate_rows(manifest, 0, fail)
    # A lost image does not rewrite a historical saved verdict.
    (tmp_path / "sft/0.png").unlink()
    result = demo.build_comparison(manifest, tmp_path / "out")
    assert result["arms"][0]["summary"]["attempts"] == 2
    assert result["arms"][0]["summary"]["strict_pass_rate_all_attempts"] == 0.5
    assert result["arms"][0]["summary"]["generation_errors"] == 1
    assert result["arms"][1]["summary"]["missing_png_files"] == 1
    assert result["arms"][1]["summary"]["strict_passes"] == 2
    page = (tmp_path / "out/index.html").read_text()
    assert "Missing PNG" in page and "No drawing" in page and "data:image/png;base64," in page
    assert len(result["paired_intervals"]) == 1
    assert result["paired_intervals"][0]["interval"]["difference"] == -0.5


@pytest.mark.parametrize(
    "field",
    [
        "runtime",
        "policy",
        "prompt",
        "rubric",
        "reference",
        "duplicate",
        "missing",
        "model",
        "heldout",
    ],
)
def test_refuses_incompatible_or_incomplete_evidence(tmp_path, field):
    manifest = fixture(tmp_path)
    if field in ["runtime", "policy"]:
        path = tmp_path / "sft/benchmark.json"
        data = json.loads(path.read_text())
        data["configuration"][field]["changed"] = True
        write(path, data)
    else:

        def change(rows):
            if field == "prompt":
                rows[0]["prompt"] = "different"
            if field == "rubric":
                rows[0]["scores"]["astra_judge"]["scorer_configuration"]["rubric_hash"] = (
                    "different"
                )
            if field == "reference":
                rows[0]["scores"]["astra_judge"]["reference"]["png_sha256"] = "different"
            if field == "duplicate":
                rows.append(rows[0])
            if field == "missing":
                rows.pop()
            if field == "model":
                rows[0]["model"] = "alias:latest"
            if field == "heldout":
                rows[0]["benchmark_metadata"]["split"] = "heldout"

        mutate_rows(manifest, 1, change)
    with pytest.raises(ValueError):
        demo.build_comparison(manifest, tmp_path / "out")
    assert not (tmp_path / "out/index.html").exists()


def test_historical_reference_discloses_differences_and_has_no_paired_interval(tmp_path):
    manifest = fixture(tmp_path)
    config = json.loads(manifest.read_text())
    config["arms"][1].update(
        role="historical_reference",
        caveat="Different historical timeout and token budget; not runtime matched.",
    )
    write(manifest, config)
    path = tmp_path / "sft/benchmark.json"
    data = json.loads(path.read_text())
    data["configuration"]["policy"]["max_tokens"] = 6000
    write(path, data)
    result = demo.build_comparison(manifest, tmp_path / "out")
    assert result["paired_intervals"] == []
    assert result["arms"][1]["runtime_differences"]
    assert "not runtime matched" in (tmp_path / "out/index.html").read_text()


class Remote:
    """Small in-memory W&B boundary: the actual exporter and receipt code run unchanged."""

    def __init__(self, fail=False):
        self.fail = fail
        self.saved = {}
        self.summary = {}
        self.url = "https://wandb.ai/e/p/runs/test"
        self.settings = SimpleNamespace(mode="online")
        self.plot = SimpleNamespace(bar=lambda *a, **k: {"chart": "bar"})

    def init(self, **kwargs):
        self.saved["init"] = kwargs
        return self

    def Table(self, columns, data=None):
        return SimpleNamespace(columns=columns, data=data or [], add_data=lambda *args: None)

    def Image(self, path):
        return str(path)

    def Artifact(self, name, type):
        return SimpleNamespace(add_file=lambda *a, **k: None, name=name + ":v0", wait=lambda: None)

    def log(self, payload):
        self.saved["payload"] = payload

    def log_artifact(self, artifact):
        if self.fail:
            raise RuntimeError("network unavailable")
        return artifact

    def finish(self, **kwargs):
        pass


def test_publish_receipt_retry_and_idempotency(tmp_path):
    manifest = fixture(tmp_path)
    output = tmp_path / "out"
    demo.build_comparison(manifest, output)
    with pytest.raises(RuntimeError, match="network unavailable"):
        demo.publish_comparison(output, entity="e", project="p", wandb_module=Remote(fail=True))
    failed = json.loads((output / "publication.json").read_text())
    assert failed["status"] == "failed"
    remote = Remote()
    receipt = demo.publish_comparison(output, entity="e", project="p", wandb_module=remote)
    assert receipt["status"] == "published"
    assert receipt["run_id"] == failed["run_id"]
    assert "comparison_summary" in remote.saved["payload"]
    assert "matched_images" in remote.saved["payload"]
    assert remote.summary["arms/base/attempts"] == 2
    assert (
        demo.publish_comparison(output, entity="e", project="p", wandb_module=Remote(fail=True))
        == receipt
    )
    with pytest.raises(ValueError, match="destination"):
        demo.publish_comparison(output, entity="other", project="p", wandb_module=Remote())


def test_evidence_tamper_refused_and_success_receipt_preserved(tmp_path):
    output = tmp_path / "out"
    demo.build_comparison(fixture(tmp_path), output)
    demo.publish_comparison(output, entity="e", project="p", wandb_module=Remote())
    receipt = (output / "publication.json").read_bytes()
    report = json.loads((output / "comparison.json").read_text())
    report["arms"][0]["summary"]["strict_passes"] = 100
    write(output / "comparison.json", report)
    with pytest.raises(ValueError):
        demo.publish_comparison(output, entity="e", project="p", wandb_module=Remote())
    assert (output / "publication.json").read_bytes() == receipt


def test_terminal_generation_error_with_no_prompt_version_is_retained(tmp_path):
    manifest = fixture(tmp_path)

    def fail(rows):
        rows[0].update(
            valid=False,
            error="InternalServerError: generation failed",
            prompt_version=None,
            png_path=None,
        )
        rows[0]["benchmark_metadata"]["status"] = "generation_error"
        rows[0]["scores"]["astra_judge"].update(status="missing_image", strict_pass=False)

    mutate_rows(manifest, 0, fail)
    result = demo.build_comparison(manifest, tmp_path / "out")
    assert result["arms"][0]["summary"]["generation_errors"] == 1
    assert result["arms"][0]["summary"]["strict_pass_rate_all_attempts"] == 0.5


def test_evaluation_reference_becomes_clickable_original_object_link(tmp_path):
    manifest = fixture(tmp_path)
    write(
        tmp_path / "base/astra/weave-publication.json",
        {"evaluations": {"base": {"ref": "weave:///entity/project/object/eval-name:abc"}}},
    )
    result = demo.build_comparison(manifest, tmp_path / "out")
    assert (
        result["arms"][0]["evaluation_url"]
        == "https://wandb.ai/entity/project/weave/objects/eval-name/versions/abc"
    )


def test_calibration_status_is_evidence_not_a_changed_rubric(tmp_path):
    manifest = fixture(tmp_path)
    path = tmp_path / "sft/astra/scorer.json"
    scorer = json.loads(path.read_text())
    scorer["judge"]["calibration_status"] = "calibrated"
    write(path, scorer)
    mutate_rows(
        manifest,
        1,
        lambda rows: [
            r["scores"]["astra_judge"]["scorer_configuration"].update(
                calibration_status="calibrated"
            )
            for r in rows
        ],
    )
    result = demo.build_comparison(manifest, tmp_path / "out")
    assert len(result["paired_intervals"]) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_policy", {"timeout_s": 42}),
        ("provider", "different"),
        ("base_url", "https://different.invalid"),
    ],
)
def test_refuses_per_arm_runtime_or_transport_changes(tmp_path, field, value):
    manifest = fixture(tmp_path)
    path = tmp_path / "sft/benchmark.json"
    config = json.loads(path.read_text())
    config["configuration"]["arms"][0][field] = value
    write(path, config)
    with pytest.raises(ValueError, match="Runtime"):
        demo.build_comparison(manifest, tmp_path / "out")


def test_publication_rejects_arbitrary_files_from_local_inventory(tmp_path):
    output = tmp_path / "out"
    demo.build_comparison(fixture(tmp_path), output)
    private = tmp_path / "private.json"
    private.write_text("private-data-not-evidence")
    files = json.loads((output / "files.json").read_text())
    files["../private.json"] = hashlib.sha256(private.read_bytes()).hexdigest()
    write(output / "files.json", files)
    remote = Remote()
    with pytest.raises(ValueError, match="inventory"):
        demo.publish_comparison(output, entity="e", project="p", wandb_module=remote)
    assert not remote.saved


def test_epoch_label_comes_from_exact_checkpoint_lineage_not_arm_name(tmp_path):
    manifest = fixture(tmp_path)
    config = json.loads(manifest.read_text())
    config["arms"][1]["label"] = "SFT 99 epochs"
    lineage = tmp_path / "lineage.json"
    write(
        lineage,
        [
            {
                "arm_name": "sft",
                "artifact": config["arms"][1]["model_id"],
                "sft_config": {"epochs": 3},
                "forge_job_id": "job",
                "forge_dataset_id": "dataset",
            }
        ],
    )
    config["arms"][1]["lineage_path"] = str(lineage)
    write(manifest, config)
    result = demo.build_comparison(manifest, tmp_path / "out")
    assert result["arms"][1]["training_metadata"]["sft_config"]["epochs"] == 3
    assert result["arms"][1]["arm_label"] == "SFT · 3 epochs · v1"
    assert "99 epochs" not in (tmp_path / "out/index.html").read_text()
    data = json.loads(lineage.read_text())
    data[0]["artifact"] = "wandb-artifact:///entity/project/model:v2"
    write(lineage, data)
    with pytest.raises(ValueError, match="lineage"):
        demo.build_comparison(manifest, tmp_path / "different")


def test_artifact_without_tuning_method_still_requires_pinned_checkpoint(tmp_path):
    manifest = fixture(tmp_path)
    path = tmp_path / "sft/benchmark.json"
    data = json.loads(path.read_text())
    data["configuration"]["arms"][0].pop("tuning_method")
    write(path, data)
    result = demo.build_comparison(manifest, tmp_path / "out")
    assert result["arms"][1]["model_label"] == "ft-Qwen/Qwen3.8-27B"
    assert result["arms"][1]["checkpoint"].endswith(":v1")
    data["configuration"]["arms"][0]["model"] = "wandb-artifact:///entity/project/model:latest"
    write(path, data)
    config = json.loads(manifest.read_text())
    config["arms"][1]["model_id"] = data["configuration"]["arms"][0]["model"]
    write(manifest, config)
    with pytest.raises(ValueError, match="immutable"):
        demo.build_comparison(manifest, tmp_path / "other")


def test_publication_rejects_evidence_symlink_outside_output(tmp_path):
    output = tmp_path / "out"
    demo.build_comparison(fixture(tmp_path), output)
    outside = tmp_path / "external-page.html"
    outside.write_bytes((output / "index.html").read_bytes())
    (output / "index.html").unlink()
    (output / "index.html").symlink_to(outside)
    with pytest.raises(ValueError, match="inventory"):
        demo.publish_comparison(output, entity="e", project="p", wandb_module=Remote())
