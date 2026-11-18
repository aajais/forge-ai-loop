"""Reviewed Forge imports must preserve real tools and reject unbound evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
import yaml
from test_training_contract import review
from test_trajectories import captured_row

from bufogen.evaluation.benchmark import comparison_fingerprint, digest
from bufogen.integrations.forge import ForgeClient, ForgeError
from bufogen.settings import ROOT
from bufogen.workflows.forge_loop import AmbiguousOperation


def module():
    path = ROOT / "docs/forge-iteration2/train_reviewed.py"
    assert path.exists(), "iteration2 reviewed training workflow must exist"
    spec = importlib.util.spec_from_file_location("iteration2_training", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The evidence fixture is a frozen collection, independent of later UI edits.
    mod.prompt_version = lambda: "bufo-v6-79c9c91ca9fa"
    return mod


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "iteration2"
    teacher = root / "teacher"
    teacher.mkdir(parents=True)
    rows, prompts, reviews = [], [], []
    for i in range(104):
        row = captured_row()
        row.id, row.prompt_id, row.group_id = f"gen-{i}", f"prompt-{i}", f"group-{i}"
        row.prompt = f"Draw Bufo scenario {i}"
        row.prompt_version = "bufo-v6-79c9c91ca9fa"
        row.trajectory["provenance"]["prompt_version"] = row.prompt_version
        row.model, row.backend = "gpt-6.1-sol", "closed_quality_bar"
        split = "train" if i < 80 else "validation"
        family = f"train-{i // 8}" if i < 80 else f"val-{(i - 80) // 6}"
        prompt = {"id": row.prompt_id, "prompt": row.prompt, "family": family,
                  "split": split, "requirements": ["Bufo touches prop"], "curriculum_metadata": {"version": "bufo-curriculum-v2", "kind": "graded_request"}}
        prompts.append(prompt)
        image = teacher / f"{i}.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nfixture-image")
        row.png_path = str(image)
        trajectory = row.trajectory
        trajectory["provenance"].update(model=row.model, source_models=[row.model],
                                         prompt_id=row.prompt_id, group_id=row.group_id)
        trajectory["messages"][1]["content"] = row.prompt
        trajectory["tools"].append({"type": "function", "function": {
            "name": "plan_scene", "description": "Plan", "parameters": {
                "type": "object", "properties": {"layout": {"type": "string"}},
                "required": ["layout"], "additionalProperties": False}}})
        trajectory["messages"][2:2] = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "plan-1", "type": "function", "function": {
                    "name": "plan_scene", "arguments": json.dumps({"layout": "shared contact"})}}]},
            {"role": "tool", "tool_call_id": "plan-1", "content": json.dumps({
                "version": "bufo-scene-plan-v1", "verification": "planned_geometry_only"})},
        ]
        row.benchmark_metadata = {"family": family, "split": split,
                                  "requirements": prompt["requirements"], "turns": [],
                                  "run_id": "run-1", "png_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
        reviews.append(review(row, reviewer_kind="assistant", partition=split,
            prompt_family=family, source_ref=str(image),
            png_sha256=row.benchmark_metadata["png_sha256"], visual_quality_pass=True))
        rows.append(row.to_dict())
    config = {"prompt_version": "bufo-v6-79c9c91ca9fa", "prompts": prompts, "arms": [{"name": "closed_quality_bar", "model": "gpt-6.1-sol"}],
              "policy": {"max_turns": 8}, "source_hashes": {"src/bufogen/agent.py": "frozen"}}
    settings = {"configuration": config, "configuration_hash": digest(config),
                "comparison_fingerprint": comparison_fingerprint(config), "run_id": "run-1"}
    for row in rows:
        row["benchmark_metadata"]["configuration_hash"] = settings["configuration_hash"]
    (teacher / "benchmark.json").write_text(json.dumps(settings))
    write_jsonl(teacher / "manifest.jsonl", rows)
    review_path = root / "reviews/a/decisions.jsonl"
    write_jsonl(review_path, reviews)
    source = root / "prompts.yaml"
    source.write_text(yaml.safe_dump(prompts + [{"id": "future-test", "prompt": "Unseen scene",
        "split": "heldout", "family": "unseen-family", "requirements": ["Bufo"]}]))
    curriculum = {"schema_version": 1, "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "counts": {"train": 80, "validation": 24, "heldout": 1},
        "excluded_prior_ids": [], "excluded_prior_families": [], "corrections": []}
    (root / "curriculum-manifest.json").write_text(json.dumps(curriculum))
    return root, rows, reviews, review_path


def test_prepare_preserves_actual_plan_tools_and_review_decisions(evidence):
    root, rows, reviews, path = evidence
    reviews[0].update(decision="reject", visual_quality_pass=False, evidence="Floating prop")
    write_jsonl(path, reviews)
    mod = module()
    receipt = mod.prepare(root)
    assert receipt["entry_counts"] == {"train": 79, "validation": 24}
    assert receipt["review_counts"] == {"accept": 103, "reject": 1}
    examples = [json.loads(x) for x in (root / "training/reviewed-examples.jsonl").read_text().splitlines()]
    assert len(examples) == 103
    original = {r["id"]: r for r in rows}
    for example in examples:
        assert example["messages"] == original[example["source_generation_id"]]["trajectory"]["messages"]
    assert "gen-0" in (root / "training/excluded.json").read_text()
    assert mod.prepare(root) == receipt


@pytest.mark.parametrize("change", ["missing", "duplicate", "automatic", "wrong_image", "wrong_family", "wrong_partition"])
def test_invalid_review_cannot_create_training_export(evidence, change):
    root, rows, reviews, path = evidence
    if change == "missing": reviews.pop()
    elif change == "duplicate": reviews[-1] = reviews[0]
    elif change == "automatic": reviews[0]["reviewer_kind"] = "automatic"
    elif change == "wrong_image": reviews[0]["png_sha256"] = "0" * 64
    elif change == "wrong_family": reviews[0]["prompt_family"] = "other"
    else: reviews[0]["partition"] = "validation"
    write_jsonl(path, reviews)
    with pytest.raises(ValueError): module().prepare(root)
    assert not (root / "training/reviewed-examples.jsonl").exists()


@pytest.mark.parametrize("change", ["missing", "late", "failed"])
def test_accepted_trajectory_requires_completed_plan_before_render(evidence, change):
    root, rows, _, _ = evidence
    messages = rows[0]["trajectory"]["messages"]
    if change == "missing": del messages[2:4]
    elif change == "late": messages[2:6] = messages[4:6] + messages[2:4]
    else: messages[3]["content"] = json.dumps({"error": "invalid plan"})
    write_jsonl(root / "teacher/manifest.jsonl", rows)
    with pytest.raises(ValueError, match="plan_scene"): module().prepare(root)


def test_curriculum_leakage_rejected_before_export(evidence):
    root, _, _, _ = evidence
    path = root / "curriculum-manifest.json"
    curriculum = json.loads(path.read_text())
    source = Path(curriculum["source"])
    prompts = yaml.safe_load(source.read_text())
    prompts[-1]["family"] = "train-0"
    source.write_text(yaml.safe_dump(prompts))
    curriculum["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(curriculum))
    with pytest.raises(ValueError, match="family|held.out|leakage"): module().prepare(root)


def test_changed_source_cannot_replace_frozen_receipt(evidence):
    root, _, reviews, path = evidence
    mod = module()
    mod.prepare(root)
    original = (root / "training/source-receipt.json").read_bytes()
    reviews[0]["evidence"] = "different review"
    write_jsonl(path, reviews)
    with pytest.raises(ValueError, match="frozen|changed"): mod.prepare(root)
    assert (root / "training/source-receipt.json").read_bytes() == original


def project(api):
    return {"entity": api.entity, "project_alias": api.project_alias, "project": "bufo", "project_id": "p1",
            "base_url": api.base_url, "training_base_url": api.training_base_url}


def test_uncertain_job_never_repeated_by_training_cli_workflow(evidence):
    root, _, _, _ = evidence
    calls = []
    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/datasets"):
            return httpx.Response(201, json={"id": "d1"})
        if request.url.path.endswith("/entries"):
            return httpx.Response(200, json={"dry_run": request.url.params["dry_run"] == "true",
                "entry_counts": {"total": 104, "train": 80, "val": 24}, "revision": 2, "group_conflicts": []})
        body = json.loads(request.content)
        assert body["model_config"]["config"] == {"epochs": 1, "batch_size": 2,
            "learning_rate": 1e-4, "assistant_turns": "all"}
        raise httpx.ReadTimeout("interrupted", request=request)
    with ForgeClient(api_key="test", entity="team", project_alias="bufo", transport=httpx.MockTransport(handler)) as api:
        mod = module()
        mod.prepare(root)
        with pytest.raises(ForgeError): mod.train_or_status("train", root, api, project(api), candidates=["e1"])
        count = len(calls)
        with pytest.raises(AmbiguousOperation): mod.train_or_status("train", root, api, project(api), candidates=["e1"])
    assert len(calls) == count
    state = json.loads((root / "training/e1/forge-state.json").read_text())
    assert state["pending_operation"]["name"] == "create_finetune"


def test_rejected_missing_image_is_preserved_but_never_exported(evidence):
    root, rows, reviews, path = evidence
    rows[0].update(png_path=None, valid=False, error="generation_failed")
    reviews[0].update(decision="reject", visual_quality_pass=False, png_sha256=None,
                      source_ref=str(root / "teacher/manifest.jsonl"))
    write_jsonl(root / "teacher/manifest.jsonl", rows)
    write_jsonl(path, reviews)
    receipt = module().prepare(root)
    assert receipt["entry_counts"]["train"] == 79
    assert receipt["images"][0]["png_path"] is None


def test_correction_lineage_exported_in_metadata_and_provenance(evidence):
    root, _, _, _ = evidence
    curriculum_path = root / "curriculum-manifest.json"
    curriculum = json.loads(curriculum_path.read_text())
    source = Path(curriculum["source"])
    prompts = yaml.safe_load(source.read_text())
    lineage = {"version": "bufo-curriculum-v2", "kind": "student_correction",
               "source_generation_id": "old-student-gen", "source_model": "google/gemma-4-26B-A4B-it",
               "source_svg_sha256": "1" * 64, "review_status": "unreviewed_automatic_critique"}
    prompts[0]["curriculum_metadata"] = lineage
    source.write_text(yaml.safe_dump(prompts))
    curriculum["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    curriculum_path.write_text(json.dumps(curriculum))
    settings_path = root / "teacher/benchmark.json"
    settings = json.loads(settings_path.read_text())
    settings["configuration"]["prompts"] = prompts[:-1]
    settings["configuration_hash"] = digest(settings["configuration"])
    settings["comparison_fingerprint"] = comparison_fingerprint(settings["configuration"])
    settings_path.write_text(json.dumps(settings))
    manifest = root / "teacher/manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row["benchmark_metadata"]["configuration_hash"] = settings["configuration_hash"]
    write_jsonl(manifest, rows)
    module().prepare(root)
    examples = [json.loads(line) for line in (root / "training/reviewed-examples.jsonl").read_text().splitlines()]
    example = next(e for e in examples if e["prompt_id"] == "prompt-0")
    assert example["metadata"]["curriculum_metadata"] == lineage
    assert example["provenance"]["curriculum_metadata"] == lineage
    assert example["source_generation_id"] == "gen-0"
    assert example["training_review"]["source_generation_id"] == "gen-0"


def test_two_independent_recipes_status_and_repeat_train_keep_original_job_ids(evidence):
    root, _, _, _ = evidence
    creates, imports, gets = [], [], []
    jobs = {}
    def handler(request):
        if request.url.path.endswith("/datasets"):
            imports.append(json.loads(request.content))
            return httpx.Response(201, json={"id": f"d{len(imports)}"})
        if request.url.path.endswith("/entries"):
            return httpx.Response(200, json={"dry_run": request.url.params["dry_run"] == "true",
                "entry_counts": {"total": 104, "train": 80, "val": 24}, "revision": 2, "group_conflicts": []})
        if request.method == "POST":
            body = json.loads(request.content)
            creates.append(body)
            job = {"id": f"job-{len(creates)}", "status": "queued", **{k: body[k] for k in ("dataset_id", "model_name", "base_model")}}
            jobs[job["id"]] = job
            return httpx.Response(202, json=job)
        gets.append(request)
        job = dict(jobs[request.url.path.rsplit("/", 1)[1]])
        job.update(status="deployed", model_artifact_ref=f"wandb-artifact:///team/bufo/{job['model_name']}:v1")
        return httpx.Response(200, json=job)
    mod = module()
    with ForgeClient(api_key="test", entity="team", project_alias="bufo", transport=httpx.MockTransport(handler)) as api:
        first = mod.train_or_status("train", root, api, project(api))
        polled = mod.train_or_status("status", root, api, project(api))
        repeated = mod.train_or_status("train", root, api, project(api))
    assert len(creates) == len(imports) == 2
    assert len(gets) == 4
    assert [c["model_config"]["config"]["epochs"] for c in creates] == [1, 3]
    for arm in ("e1", "e3"):
        assert first[arm]["finetune_id"] == polled[arm]["finetune_id"] == repeated[arm]["finetune_id"]
        assert repeated[arm]["status"] == "deployed"
        assert len(list((root / "training" / arm / "history").glob("*.json"))) == 3
    assert first["e1"]["source_sha256"] == first["e3"]["source_sha256"]
    assert first["e1"]["upload_sha256"] == first["e3"]["upload_sha256"]


def test_status_without_saved_job_never_calls_remote(evidence):
    root, _, _, _ = evidence
    calls = []
    with ForgeClient(api_key="test", entity="team", project_alias="bufo", transport=httpx.MockTransport(lambda r: calls.append(r))) as api:
        with pytest.raises(ValueError, match="saved workflow"):
            module().train_or_status("status", root, api, project(api))
    assert not calls


def test_new_export_binds_current_prompt_but_existing_job_keeps_frozen_version(evidence):
    root, _, _, _ = evidence
    mod = module()
    mod.prompt_version = lambda: "different-current-version"
    with pytest.raises(ValueError, match="harness|version"):
        mod.prepare(root)
    mod.prompt_version = lambda: "bufo-v6-79c9c91ca9fa"
    receipt = mod.prepare(root)
    mod.prompt_version = lambda: "later-ui-version"
    assert mod.prepare(root) == receipt


def test_opt_in_tls12_transport_keeps_certificate_and_hostname_verification(monkeypatch):
    import ssl
    mod = module()
    seen = []
    class InspectTransport:
        def __init__(self, **kwargs): seen.append(kwargs)
    monkeypatch.setattr(mod.httpx, "HTTPTransport", InspectTransport)
    transport, metadata = mod.transport_for_cli(True)
    assert isinstance(transport, InspectTransport)
    context = seen[0]["verify"]
    assert context.minimum_version == context.maximum_version == ssl.TLSVersion.TLSv1_2
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert metadata["mode"] == "tls12"
    assert metadata["certificate_verification"] is True
    default, metadata = mod.transport_for_cli(False)
    assert default is None and metadata["mode"] == "default"
    assert len(seen) == 1


def test_cli_exposes_tls12_opt_in_without_changing_default():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / "docs/forge-iteration2/train_reviewed.py"), "--help"],
                            capture_output=True, text=True, check=True)
    assert "--tls12" in result.stdout


def test_transport_choice_is_receipted_even_when_upload_is_ambiguous(evidence):
    root, _, _, _ = evidence
    def handler(request):
        raise httpx.ReadTimeout("interrupted", request=request)
    metadata = {"mode": "tls12", "certificate_verification": True, "hostname_check": True}
    with ForgeClient(api_key="test", entity="team", project_alias="bufo", transport=httpx.MockTransport(handler)) as api:
        with pytest.raises(ForgeError):
            module().train_or_status("train", root, api, project(api), candidates=["e1"],
                                     transport_metadata=metadata)
    receipts = list((root / "training/transport-history").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["transport"] == metadata
    assert receipt["operation"] == "train"
    assert receipt["candidates"] == ["e1"]
    assert "training_config" not in receipt
