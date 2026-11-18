"""Re-key a labelled review packet with another judge run's verdicts.

The packet's unblinding key records the verdicts of the judge that was live when the packet
was built. To test a revised rubric against labels already in hand, this writes a second key
with the same items and the strict_pass / vs_reference verdicts read from the scored rows of
the revised run (e.g. <run>/astra-v2/scored.jsonl). review_agreement.py then compares the
reviewer's decisions against that key. The output stays in the key directory.
"""

import argparse
import json
from pathlib import Path

from bufogen.curation.review_packet import _inside


def rekey(key_path: Path, scored_paths: list[Path], out: Path) -> dict:
    key_path, out = Path(key_path), Path(out)
    if out.parent.resolve() != key_path.parent.resolve():
        raise ValueError("the re-keyed file must be written beside the original key")
    verdicts, versions = {}, set()
    for scored in scored_paths:
        for line in Path(scored).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            judge = row.get("scores", {}).get("astra_judge", {})
            if judge.get("status") not in ("ok", "missing_image"):
                continue
            versions.add(judge.get("rubric_version"))
            verdicts[row["id"]] = {
                "judge_strict_pass": bool(judge.get("strict_pass")),
                "judge_vs_reference": judge.get("vs_reference"),
            }
    if len(versions) != 1:
        raise ValueError(f"scored rows must share one rubric version, got {sorted(versions)}")
    key = json.loads(key_path.read_text())
    missing = [k["generation_id"] for k in key if k["generation_id"] not in verdicts]
    if missing:
        raise ValueError(f"{len(missing)} packet generations have no verdict in the scored rows")
    version = versions.pop()
    rekeyed = [k | verdicts[k["generation_id"]] | {"rubric_version": version} for k in key]
    out.write_text(json.dumps(rekeyed, indent=2) + "\n")
    changed = sum(
        a["judge_strict_pass"] != b["judge_strict_pass"] for a, b in zip(key, rekeyed, strict=True)
    )
    return {"n": len(rekeyed), "strict_verdicts_changed": changed, "out": str(out)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True, help="original unblinding-key.json")
    parser.add_argument("--scored", type=Path, action="append", required=True, help="scored.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--packet", type=Path, help="packet dir; the output must not be inside it")
    args = parser.parse_args(argv)
    if args.packet is not None and _inside(args.out, args.packet):
        raise SystemExit("output must not be inside the packet directory")
    print(json.dumps(rekey(args.key, args.scored, args.out), indent=2))


if __name__ == "__main__":
    main()
