"""Offline durability and identity checks for the optional managed LR runner."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest

from bufogen.integrations.forge import ForgeClient, ForgeError
from bufogen.workflows.forge_loop import AmbiguousOperation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/forge-gap-work"))
spec = importlib.util.spec_from_file_location(
    "gap_lr_training", ROOT / "docs/forge-gap-work/train_lr_candidates.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    original = tmp_path / "original"
    for suffix in (
        "training/reviewed-examples.jsonl",
        "training/source-receipt.json",
        "training/e1/forge-state.json",
        "training/e1/wandb-metrics.json",
    ):
        dest = original / suffix
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runner.ORIGINAL / suffix, dest)
    monkeypatch.setenv("WANDB_API_KEY", "offline-test-only")
    return tmp_path / "new", original


def mock_factory(calls, *, fail_create=False):
    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(
                200, json={"id": request.url.path.rsplit("/", 1)[-1], "status": "queued"}
            )
        if request.url.path.endswith("/datasets"):
            return httpx.Response(201, json={"id": "dataset-" + str(len(calls))})
        if request.url.path.endswith("/entries"):
            return httpx.Response(
                200,
                json={
                    "dry_run": request.url.params.get("dry_run") == "true",
                    "entry_counts": {"total": 147, "train": 124, "val": 23},
                    "group_conflicts": [],
                    "revision": 2,
                },
            )
        assert request.url.path.endswith("/finetunes")
        if fail_create:
            raise httpx.ReadTimeout("ambiguous remote acknowledgement", request=request)
        return httpx.Response(202, json={"id": "job-" + str(len(calls)), "status": "queued"})

    def factory(**kwargs):
        return ForgeClient(**kwargs, transport=httpx.MockTransport(handler))

    return factory


def test_prepare_is_local_exact_and_idempotent(workspace):
    root, original = workspace
    first = runner.run(
        "prepare",
        root=root,
        original=original,
        client_factory=lambda **kw: pytest.fail("prepare contacted remote service"),
    )
    assert first == runner.run("prepare", root=root, original=original)
    assert (root / "training/reviewed-examples.jsonl").read_bytes() == (
        original / "training/reviewed-examples.jsonl"
    ).read_bytes()
    assert first["entry_counts"] == {"train": 124, "val": 23}
    assert {c["training_config"]["learning_rate"] for c in first["candidates"].values()} == {
        2e-5,
        1e-4,
    }
    assert not list(root.glob("training/*/forge-state.json"))


@pytest.mark.parametrize("change", ["source", "recipe", "plan"])
def test_changed_inputs_fail_before_remote(workspace, monkeypatch, change):
    root, original = workspace
    runner.prepare(root, original=original)
    if change == "source":
        with (original / "training/reviewed-examples.jsonl").open("a") as stream:
            stream.write("\n")
    elif change == "recipe":
        monkeypatch.setitem(runner.RECIPES, "lr2e-5", 3e-5)
    else:
        p = root / "plan.json"
        value = json.loads(p.read_text())
        value["purpose"] = "tampered"
        p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.run(
            "submit",
            root=root,
            original=original,
            client_factory=lambda **kw: pytest.fail("HTTP before validation"),
        )


def test_submit_twice_creates_each_job_once_and_poll_is_get_only(workspace):
    root, original = workspace
    calls = []
    factory = mock_factory(calls)
    before = (original / "training/e1/forge-state.json").read_bytes()
    states = runner.run("submit", root=root, original=original, client_factory=factory)
    assert len(states) == 2
    jobs = [r for r in calls if r.method == "POST" and r.url.path.endswith("/finetunes")]
    assert len(jobs) == 2
    assert {json.loads(r.content)["model_config"]["config"]["learning_rate"] for r in jobs} == {
        2e-5,
        1e-4,
    }
    assert all(json.loads(r.content)["base_model"] == runner.STUDENT_MODEL for r in jobs)
    count = len(calls)
    runner.run("submit", root=root, original=original, client_factory=factory)
    assert all(r.method == "GET" for r in calls[count:])
    count = len(calls)
    runner.run("poll", root=root, original=original, client_factory=factory)
    assert len(calls[count:]) == 2 and all(r.method == "GET" for r in calls[count:])
    assert (original / "training/e1/forge-state.json").read_bytes() == before
    for name in runner.RECIPES:
        history = list((root / "training" / name / "history").glob("*.json"))
        assert len(history) == 5


def test_ambiguous_job_submission_does_not_replay(workspace):
    root, original = workspace
    calls = []
    factory = mock_factory(calls, fail_create=True)
    with pytest.raises(ForgeError):
        runner.run("submit", root=root, original=original, names=["lr2e-5"], client_factory=factory)
    count = len(calls)
    with pytest.raises(AmbiguousOperation):
        runner.run("submit", root=root, original=original, names=["lr2e-5"], client_factory=factory)
    assert len(calls) == count
    state = runner.read(root / "training/lr2e-5/forge-state.json")
    assert state["pending_operation"]["name"] == "create_finetune"
    assert len(list((root / "training/lr2e-5/history").glob("*.json"))) == 3


def test_saved_job_rate_tampering_rejected_before_get(workspace):
    root, original = workspace
    calls = []
    factory = mock_factory(calls)
    runner.run("submit", root=root, original=original, names=["lr2e-5"], client_factory=factory)
    p = root / "training/lr2e-5/forge-state.json"
    state = runner.read(p)
    state["training_config"]["learning_rate"] = 0.1
    p.write_text(json.dumps(state))
    count = len(calls)
    with pytest.raises(ValueError, match="recipe changed"):
        runner.run("poll", root=root, original=original, names=["lr2e-5"], client_factory=factory)
    assert len(calls) == count


def test_poll_cannot_initialize_missing_candidate(workspace):
    root, original = workspace
    runner.prepare(root, original=original)
    with pytest.raises(ValueError, match="without a saved workflow"):
        runner.run(
            "poll",
            root=root,
            original=original,
            client_factory=lambda **kw: pytest.fail("poll created client"),
        )
    assert not list(root.glob("training/*/forge-state.json"))
