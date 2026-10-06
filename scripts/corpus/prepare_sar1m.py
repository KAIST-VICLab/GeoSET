#!/usr/bin/env python
"""SAR-1M stage-2 pairs: the entries of paired.json (SAR samples with an EO counterpart).

The paired subset consists of two naming conventions: M4-SAR tiles '<base>_<k>.jpg'
(scene group <base>) and SEN1-2 patches 'ROIs..._s1_<scene>_p<patch>.png' (scene group
'ROIs..._s1_<scene>'). Rows keep the order of paired.json.

    python scripts/corpus/prepare_sar1m.py --root $CORPUS_ROOT/sar1m --out work/pairs/sar1m.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

SEN12 = re.compile(r"^(ROIs\w*?_s1_\d+)_p(\d+)\.png$")
M4SAR = re.compile(r"^(\d+)_(\d+)\.jpg$")


def classify(name: str) -> tuple[str, str]:
    """SAR file name -> (subset, scene group)."""
    if m := M4SAR.match(name):
        return "m4sar", m.group(1)
    if m := SEN12.match(name):
        return "sen12", m.group(1)
    raise ValueError(f"unrecognised SAR-1M paired file name: {name!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="SAR-1M root holding SAR/, OPT/ and paired.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    entries = json.loads((args.root / "paired.json").read_text())
    rows, n = [], Counter()
    for e in entries:
        name = os.path.basename(e["sar"])
        sub, group = classify(name)
        n[sub] += 1
        rows.append((f"sar1m/{sub}/{os.path.splitext(name)[0]}", f"sar1m/{sub}/{group}", e["sar"], e["optical"]))
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("duplicate SAR-1M sample ids")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("sample_id\tscene_id\tsar\teo\n")
        fh.writelines("\t".join(r) + "\n" for r in rows)
    print(f"{len(rows):,} pairs (m4sar {n['m4sar']:,}, sen12 {n['sen12']:,}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
