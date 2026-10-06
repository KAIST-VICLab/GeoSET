#!/usr/bin/env python
"""TerraMesh stage-2 eligibility from the dataset metadata.

A Major TOM sample (S1RTC + S2RGB) is eligible when cloud_cover == 0 and the
S1/S2 acquisition times differ by at most one day. Samples are named
<split>/<zarr member name>, which is how the keep table addresses them.

    python scripts/corpus/prepare_terramesh.py --root $CORPUS_ROOT/terramesh --out work/pairs/terramesh.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

MAX_DAYS = 1.0


def eligible(root: Path, split: str) -> tuple[list[tuple[str, str, str, str]], int]:
    t = pq.read_table(root / f"{split}_metadata.parquet",
                      columns=["tar", "zarr", "sample_id", "cloud_cover", "S1_time", "S2_time"])
    t = t.filter(pc.starts_with(t["tar"], "majortom_"))
    s1 = t["S1_time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    s2 = t["S2_time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    ok = (t["cloud_cover"].to_numpy() == 0.0) & (np.abs((s1 - s2) / 86400.0) <= MAX_DAYS)
    tar, zarr, sid = (t[c].to_numpy(zero_copy_only=False) for c in ("tar", "zarr", "sample_id"))
    rows = [(f"{split}/{z}", s.rsplit("_", 2)[0], f"{split}/S1RTC/{a}", f"{split}/S2RGB/{a}")
            for a, z, s in zip(tar[ok], zarr[ok], sid[ok])]
    return rows, t.num_rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="TerraMesh root holding {train,val}_metadata.parquet")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = []
    for split in ("train", "val"):
        r, n = eligible(args.root, split)
        print(f"{split}: {len(r):,} eligible of {n:,} Major TOM samples")
        rows += r
    rows.sort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("sample_id\tscene_id\tsar\teo\n")
        fh.writelines("\t".join(r) + "\n" for r in rows)
    print(f"{len(rows):,} eligible -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
