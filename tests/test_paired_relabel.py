import importlib.util
from pathlib import Path

from test_paired_across_runs import row, write_dir

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/paired_relabel.py"
spec = importlib.util.spec_from_file_location("paired_relabel", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_same_arm_name_on_both_sides_is_paired_by_relabelling(tmp_path):
    fams = ["a", "a", "b", "b"]
    left = write_dir(tmp_path, "l", [row(f"p{i}", "x", f, True, i) for i, f in enumerate(fams)])
    right = write_dir(tmp_path, "r", [row(f"p{i}", "x", f, i < 2, i) for i, f in enumerate(fams)])
    report = mod.paired_relabel(left, "x", right, "x", "x_right")
    assert report["paired"]["difference"] == 0.5 and report["left"]["arm"] == "x"
