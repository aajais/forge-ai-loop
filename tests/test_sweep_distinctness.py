import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import sweep_distinctness as sd  # noqa: E402


def row(backend, pid, tokens, svg="<svg><rect id='x'/></svg>"):
    return {"backend": backend, "prompt_id": pid, "output_tokens": tokens, "svg": svg}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def base_rows(rng, backend="open_base", n=40):
    return [row(backend, f"p{i}", 1500 + rng.randint(-300, 300)) for i in range(n)]


def build(tmp_path, shifts):
    rng = random.Random(7)
    export = tmp_path / "export.jsonl"
    svg = '<svg><g id="sky"/><g id="sky"/><g id="wall"/></svg>'
    call = {"function": {"arguments": json.dumps({"svg": svg})}}
    export.write_text(
        json.dumps({"messages": [{"role": "assistant", "tool_calls": [call]}]}) + "\n"
    )
    sweep = tmp_path / "sweep"
    for arm, shift in shifts.items():
        base = base_rows(rng)
        student = [
            row(arm, r["prompt_id"], r["output_tokens"] + shift + rng.randint(-20, 20))
            for r in base
        ]
        write(sweep / arm / "astra-v5-k3" / "scored.jsonl", base + student)
    control = tmp_path / "control"
    write(control / "scored.jsonl", base_rows(rng))
    sol = tmp_path / "sol"
    write(
        sol / "scored.jsonl",
        [
            row("closed_quality_bar", f"p{i}", 1, '<svg><g id="sky"/><g id="zzz"/></svg>')
            for i in range(5)
        ],
    )
    return dict(
        sweep_dir=sweep,
        arms=list(shifts),
        control_run=control,
        sol_run=sol,
        export=export,
        replicates=2000,
    )


def test_shifted_arm_is_distinct_and_identical_distribution_is_not_separable(tmp_path):
    report = sd.analyze(**build(tmp_path, {"shifted": -400, "same": 0}))
    assert report["schema_version"] == 1
    assert report["verdict"] == {"shifted": "distinct", "same": "not_separable"}
    assert report["student_vs_base"]["shifted_vs_open_base"]["output_tokens"]["p"] < 0.01
    assert report["student_vs_base"]["shifted_vs_open_base"]["output_tokens"]["mean_diff"] < -300
    assert all(len(v) == 64 for v in report["inputs"].values())


def test_fingerprint_share_counts_row_ids_in_the_training_top_set(tmp_path):
    report = sd.analyze(**build(tmp_path, {"shifted": -400}))
    fp = report["sol_training_id_fingerprint"]
    assert fp["top_ids"] == ["sky", "wall"]
    assert fp["export_svgs"] == 1
    assert fp["sol_validation_rows"]["mean_share"] == 0.5  # sky in, zzz out
    assert fp["shifted_student"]["mean_share"] == 0.0  # ids are 'x' only


def test_noisy_control_blocks_a_distinct_verdict(tmp_path):
    args = build(tmp_path, {"shifted": -400})
    rng = random.Random(3)
    # a control run that is itself far from the sweep's base: the null is not quiet
    write(
        args["control_run"] / "scored.jsonl",
        [row("open_base", f"p{i}", 3500 + rng.randint(-50, 50)) for i in range(40)],
    )
    assert sd.analyze(**args)["verdict"] == {"shifted": "not_separable"}


def test_arm_with_another_base_is_compared_in_run_and_keeps_gemma_controls(tmp_path):
    args = build(tmp_path, {"shifted": -400})
    rng = random.Random(11)
    qwen_base = [row("qwen_base", f"p{i}", 4000 + rng.randint(-300, 300)) for i in range(40)]
    qwen = [row("sft_q", r["prompt_id"], r["output_tokens"] - 900) for r in qwen_base]
    write(args["sweep_dir"] / "sft_q" / "astra-v5-k3" / "scored.jsonl", qwen_base + qwen)
    args["arms"] = ["shifted", "sft_q"]
    report = sd.analyze(**args, base_arms={"sft_q": "qwen_base"})
    assert report["base_arms"] == {"shifted": "open_base", "sft_q": "qwen_base"}
    assert "sft_q_vs_qwen_base" in report["student_vs_base"]
    assert all("qwen_base" not in name for name in report["base_vs_base_controls"])
    assert report["verdict"] == {"shifted": "distinct", "sft_q": "distinct"}
