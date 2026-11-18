import json

import httpx
import pytest

from bufogen.integrations.forge import ForgeClient, ForgeError, reviewed_entries
from bufogen.workflows.forge_loop import (
    AmbiguousOperation,
    import_reviewed_dataset,
    initialize_workflow,
    resume_training,
    submit_training,
)

MODEL = "google/gemma-4-26B-A4B-it"


def example(partition="train", family="cloud"):
    from bufogen.curation.curate import select_examples
    from test_training_contract import review
    from test_trajectories import captured_row

    row = captured_row()
    row.group_id = family
    row.trajectory["provenance"]["group_id"] = family
    row.training_review = review(row, partition=partition, prompt_family=family)
    return select_examples([row], heldout_ids=set(), allowed_partitions={"train", "validation"})[0]


def client(handler):
    return ForgeClient(
        api_key="secret-test-key",
        entity="team",
        project_alias="bufo",
        transport=httpx.MockTransport(handler),
    )


def test_authorization_error_sanitized_and_actionable():
    def handler(request):
        assert request.headers["Wandb-Entity"] == "team"
        return httpx.Response(403, json={"message": "Bearer secret-test-key is forbidden"})

    with client(handler) as api, pytest.raises(ForgeError) as error:
        api.list_projects()
    assert error.value.status_code == 403
    assert "secret-test-key" not in str(error.value)
    assert "permission" in str(error.value).lower()


def test_disabled_model_cannot_pass_preflight():
    def handler(request):
        if request.url.path.endswith("/projects"):
            return httpx.Response(
                200, json=[{"id": "project-1", "alias": "bufo", "entity": "team"}]
            )
        return httpx.Response(200, json={"models": [{"name": MODEL, "enabled": False}]})

    with client(handler) as api, pytest.raises(ValueError, match="eligible"):
        api.preflight(MODEL)


def test_reviewed_mapping_preserves_actual_tools_and_rejects_leakage():
    train, val = example(), example("validation", "boats")
    entries = reviewed_entries([train, val])
    assert [row["split"] for row in entries] == ["train", "val"]
    assert entries[0]["messages"] == train["messages"]
    assert entries[0]["tools"] == train["tools"]
    assert entries[0]["metadata"]["training_review"] == train["training_review"]
    assert set(entries[0]) <= {
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "row_id",
        "group_id",
        "split",
        "metadata",
        "provenance",
    }
    # Use separate group/prompt identifiers when testing family leakage.
    val["source_group_id"] = "val-group"
    val["prompt_family"] = val["training_review"]["prompt_family"] = "cloud"
    with pytest.raises(ValueError, match="family"):
        reviewed_entries([train, val])


@pytest.mark.parametrize("change", ["review", "heldout", "orphan", "final"])
def test_malformed_or_unreviewed_rows_fail_before_http(change):
    row = example()
    if change == "review":
        row["training_review"]["decision"] = "reject"
    elif change == "heldout":
        row["partition"] = row["training_review"]["partition"] = "test"
    elif change == "orphan":
        row["messages"][3]["tool_call_id"] = "invented"
    else:
        row["messages"].pop()
    with pytest.raises(ValueError):
        reviewed_entries([row])


def prepared(tmp_path, api):
    source = tmp_path / "reviewed.jsonl"
    source.write_text(json.dumps(example()) + "\n")
    path = tmp_path / "state.json"
    initialize_workflow(
        path,
        api,
        source_path=source,
        dataset_name="reviewed",
        model_name="bufo-sft",
        project_id="project-1",
    )
    return path, source


def test_import_dry_run_then_commit_and_resume_existing_job(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/datasets"):
            return httpx.Response(201, json={"id": "dataset-1"})
        if request.url.path.endswith("/entries"):
            dry = request.url.params.get("dry_run") == "true"
            return httpx.Response(
                200,
                json={
                    "dry_run": dry,
                    "created": 1,
                    "revision": 1,
                    "entry_counts": {"total": 1, "train": 1, "val": 0},
                    "group_conflicts": [],
                },
            )
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["model_config"]["config"]["assistant_turns"] == "all"
            return httpx.Response(202, json={"id": "job-1", "status": "queued"})
        return httpx.Response(
            200,
            json={
                "id": "job-1",
                "status": "deployed",
                "model_artifact_ref": "wandb-artifact:///team/p/real:v1",
            },
        )

    with client(handler) as api:
        path, source = prepared(tmp_path, api)
        imported = import_reviewed_dataset(path, api)
        assert imported["dataset_id"] == "dataset-1"
        submit_training(path, api)
        state = resume_training(path, api)
        assert state["model_artifact_ref"] == "wandb-artifact:///team/p/real:v1"
        import_reviewed_dataset(path, api)
        submit_training(path, api)
    posts = [r for r in requests if r.method == "POST"]
    assert len(posts) == 4
    assert posts[1].url.params["dry_run"] == "true"
    assert posts[2].url.params["dry_run"] == "false"
    source.write_text(source.read_text() + "\n")
    with client(handler) as api, pytest.raises(ValueError, match="source.*changed"):
        submit_training(path, api)


def test_transport_interruption_never_repeats_uncertain_write(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("secret-test-key", request=request)

    with client(handler) as api:
        path, _ = prepared(tmp_path, api)
        with pytest.raises(ForgeError) as error:
            import_reviewed_dataset(path, api)
        assert "secret-test-key" not in str(error.value)
        with pytest.raises(AmbiguousOperation):
            import_reviewed_dataset(path, api)
    assert len(calls) == 1
    assert json.loads(path.read_text())["pending_operation"]["name"] == "create_dataset"


def test_deployed_without_artifact_is_not_promoted(tmp_path):
    with client(lambda r: httpx.Response(200, json={"id": "job-1", "status": "deployed"})) as api:
        path, _ = prepared(tmp_path, api)
        state = json.loads(path.read_text())
        state["finetune_id"] = "job-1"
        path.write_text(json.dumps(state))
        result = resume_training(path, api)
    assert result["status"] == "deployed_missing_artifact"
    assert result["model_artifact_ref"] is None


def test_preflight_checks_explicit_project_identity_and_only_gets():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/projects"):
            return httpx.Response(
                200,
                json={
                    "projects": [
                        {"id": "project-1", "alias": "bufo", "entity": "team", "project": "p"}
                    ]
                },
            )
        return httpx.Response(200, json={"models": [MODEL]})

    with client(handler) as api:
        result = api.preflight(project="p", project_id="project-1")
        assert result["catalog_eligible"] and not result["job_acceptance_verified"]
        with pytest.raises(ValueError, match="entity/name"):
            api.preflight(project="wrong")
    assert all(r.method == "GET" for r in requests)


def test_invalid_source_does_not_create_state_or_call_forge(tmp_path):
    calls = []
    source = tmp_path / "bad.jsonl"
    row = example()
    row["training_review"] = None
    source.write_text(json.dumps(row))
    with client(lambda r: calls.append(r)) as api, pytest.raises(ValueError, match="review"):
        initialize_workflow(
            tmp_path / "state.json",
            api,
            source_path=source,
            dataset_name="d",
            model_name="m",
            project_id="p",
        )
    assert not (tmp_path / "state.json").exists()
    assert not calls


def test_dry_run_rejection_keeps_dataset_id_and_never_uploads(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/datasets"):
            return httpx.Response(201, json={"id": "dataset-1"})
        return httpx.Response(422, json={"message": "invalid message"})

    with client(handler) as api:
        path, _ = prepared(tmp_path, api)
        for _ in range(2):
            with pytest.raises(ForgeError, match="422"):
                import_reviewed_dataset(path, api)
    state = json.loads(path.read_text())
    assert state["dataset_id"] == "dataset-1"
    assert not state["entries_imported"]
    assert len([r for r in requests if r.url.path.endswith("/datasets")]) == 1
    assert all(r.url.params["dry_run"] == "true" for r in requests[1:])


def test_ambiguous_job_creation_requires_verified_remote_id(tmp_path):
    from bufogen.workflows.forge_loop import reconcile_remote_id

    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            raise httpx.ReadTimeout("interrupted", request=request)
        return httpx.Response(
            200,
            json={
                "id": "recovered-job",
                "status": "training",
                "dataset_id": "dataset-1",
                "model_name": "bufo-sft",
                "base_model": MODEL,
                "model_id": "underlying-model",
            },
        )

    with client(handler) as api:
        path, _ = prepared(tmp_path, api)
        state = json.loads(path.read_text())
        state.update(dataset_id="dataset-1", entries_imported=True)
        path.write_text(json.dumps(state))
        with pytest.raises(ForgeError):
            submit_training(path, api)
        with pytest.raises(AmbiguousOperation):
            submit_training(path, api)
        recovered = reconcile_remote_id(path, api, remote_id="recovered-job")
        assert recovered["finetune_id"] == "recovered-job"
        assert recovered["training_model_identity"]["model_id"] == "underlying-model"
        resume_training(path, api)
    assert len([r for r in requests if r.method == "POST"]) == 1


def test_active_file_lock_prevents_remote_mutation(tmp_path):
    import fcntl

    requests = []
    with client(lambda r: requests.append(r)) as api:
        path, _ = prepared(tmp_path, api)
        with path.with_suffix(".json.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(RuntimeError, match="lock"):
                import_reviewed_dataset(path, api)
    assert not requests


def test_server_error_is_uncertain_and_cannot_duplicate_dataset(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503, json={"message": "unavailable"})

    with client(handler) as api:
        path, _ = prepared(tmp_path, api)
        with pytest.raises(ForgeError) as error:
            import_reviewed_dataset(path, api)
        assert error.value.uncertain
        with pytest.raises(AmbiguousOperation):
            import_reviewed_dataset(path, api)
    assert len(requests) == 1


def test_documented_nested_error_retains_actionable_type_and_redacts_credentials():
    with (
        client(
            lambda r: httpx.Response(
                400,
                json={
                    "error": {
                        "type": "model_not_enabled",
                        "message": "Enable Gemma; token=secret-test-key",
                    }
                },
            )
        ) as api,
        pytest.raises(ForgeError) as error,
    ):
        api.list_projects()
    assert "model_not_enabled" in str(error.value)
    assert "Enable Gemma" in str(error.value)
    assert "secret-test-key" not in str(error.value)


def test_remote_training_failure_type_is_retained_without_retry(tmp_path):
    with client(
        lambda r: httpx.Response(
            200,
            json={
                "id": "job-1",
                "status": "failed",
                "training_failure_type": "RemoteTrainingFailed",
            },
        )
    ) as api:
        path, _ = prepared(tmp_path, api)
        state = json.loads(path.read_text())
        state["finetune_id"] = "job-1"
        path.write_text(json.dumps(state))
        result = resume_training(path, api)
    assert result["training_failure_type"] == "RemoteTrainingFailed"
    assert result["status"] == "failed"


def test_custom_metadata_cannot_overwrite_validated_source_evidence():
    source = example()
    authoritative = (
        "source_model",
        "source_generation_id",
        "source_group_id",
        "partition",
        "prompt_family",
        "prompt_id",
        "prompt_version",
        "training_review",
    )
    source["metadata"] = {key: "untrusted override" for key in authoritative}
    source["metadata"]["custom_note"] = "preserve this unrelated annotation"
    metadata = reviewed_entries([source])[0]["metadata"]
    assert {key: metadata[key] for key in authoritative} == {
        key: source[key] for key in authoritative
    }
    assert metadata["custom_note"] == "preserve this unrelated annotation"


def test_pricing_endpoints_put_exact_bodies_and_reject_bad_prices():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"id": "x", **json.loads(request.content)})

    with client(handler) as api:
        api.set_finetune_pricing("ft-1", input_usd_per_mtok=0.4, output_usd_per_mtok=3.0)
        api.set_provider_model_pricing(
            "openai", "m-1", input_usd_per_mtok=2.0, output_usd_per_mtok=10.0
        )
        with pytest.raises(ValueError):
            api.set_finetune_pricing("ft-1", input_usd_per_mtok=0, output_usd_per_mtok=1.0)
    assert seen == [
        (
            "PUT",
            "/api/distillation/v1/projects/bufo/finetunes/ft-1/pricing",
            {"input_usd_per_mtok": 0.4, "output_usd_per_mtok": 3.0},
        ),
        (
            "PUT",
            "/api/distillation/v1/providers/openai/models/m-1",
            {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 10.0},
        ),
    ]
