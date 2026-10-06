#!/usr/bin/env python
"""Deterministic pair lists for GUSO, SARLO-80 and 3MOS (sorted paths relative to the source root).

  guso     <region>/<split>/<continent>/<tile>_SAR.tif -> <tile>_OPT.tif
           (SAR-only tiles under unpaired_temp/ are skipped); scene = <region>/<continent>/<site>
  sarlo80  train_x/<chunk>/<shard>/<key>.sar.npy -> optic/<chunk>/<shard>/<key>.optic.png
           (incomplete .npy files are skipped); scene = umbra_pass in <key>.meta.json
  3mos     <group>/sar/sar_<id>.jpg -> <group>/opt/<terrain>/opt_<id>.jpg, written as 3mos_mr
           (SEN/SEN1_region1-7, ALOS) and 3mos_hr (GF3, Radarsat, RCM); scene = <group>/sar

    python scripts/corpus/list_pairs.py --source guso --root $CORPUS_ROOT/guso --out-dir work/pairs
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

MOS3_STREAMS = {"3mos_mr": tuple(f"SEN/SEN1_region{i}" for i in range(1, 8)) + ("ALOS",),
                "3mos_hr": ("GF3", "Radarsat", "RCM")}


def walk(root: Path, suffix: str) -> list[str]:
    out = []
    for d, _, files in os.walk(root):
        rel = os.path.relpath(d, root)
        out += [f if rel == "." else f"{rel}/{f}" for f in files if f.endswith(suffix)]
    return sorted(out)


def npy_complete(path: Path) -> bool:
    """True when the file holds the full array its header declares (reads the header only)."""
    with open(path, "rb", buffering=0) as fh:
        version = npy_format.read_magic(fh)
        read = npy_format.read_array_header_1_0 if version == (1, 0) else npy_format.read_array_header_2_0
        shape, _, dtype = read(fh)
        offset = fh.tell()
    return path.stat().st_size == offset + int(np.prod(shape)) * dtype.itemsize


def guso(root: Path) -> dict[str, list[tuple]]:
    rows = []
    for sar in walk(root, "_SAR.tif"):
        if "unpaired_temp/" in sar:
            continue
        p = sar.split("/")
        site = p[-1].split("_SAR")[0].rsplit("_", 1)[0]
        rows.append((sar, f"{p[0]}/{p[2]}/{site}", sar, sar.replace("_SAR.tif", "_OPT.tif")))
    return {"guso": rows}


def sarlo80(root: Path) -> dict[str, list[tuple]]:
    rows = []
    for rel in walk(root / "train_x", ".sar.npy"):
        sar = f"train_x/{rel}"
        if not npy_complete(root / sar):
            continue
        key = sar[: -len(".sar.npy")]
        scene = json.loads((root / f"{key}.meta.json").read_text())["umbra_pass"]
        rows.append((sar, scene, sar, f"optic/{rel[: -len('.sar.npy')]}.optic.png"))
    return {"sarlo80": rows}


def mos3(root: Path) -> dict[str, list[tuple]]:
    files = walk(root, ".jpg")
    opt = {}
    for o in files:
        p = o.split("/")
        if "opt" in p:
            k = p.index("opt")
            opt[("/".join(p[:k]), p[-1][len("opt_"):-len(".jpg")])] = o
    out = {s: [] for s in MOS3_STREAMS}
    for sar in files:
        p = sar.split("/")
        if len(p) < 3 or p[-2] != "sar":
            continue
        group = "/".join(p[:-2])
        eo = opt.get((group, p[-1][len("sar_"):-len(".jpg")]))
        stream = next((s for s, groups in MOS3_STREAMS.items() if group in groups), None)
        if eo is not None and stream is not None:
            out[stream].append((sar, f"{group}/sar", sar, eo))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=("guso", "sarlo80", "3mos"))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    lists = {"guso": guso, "sarlo80": sarlo80, "3mos": mos3}[args.source](args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stream, rows in lists.items():
        with open(args.out_dir / f"{stream}.tsv", "w") as fh:
            fh.write("sample_id\tscene_id\tsar\teo\n")
            fh.writelines("\t".join(r) + "\n" for r in rows)
        print(f"{stream}: {len(rows):,} pairs -> {args.out_dir / f'{stream}.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
