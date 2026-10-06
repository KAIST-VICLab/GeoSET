#!/usr/bin/env python
"""Apply the stage-2 filtering rules to per-sample scores and write the keep tables.

Inputs per stream: <pairs>/<stream>.tsv (prepare_terramesh.py, prepare_sar1m.py,
list_pairs.py) and <scores>/<stream>.tsv (score_eo_iqa.py); the SARLO-80 registration
rules read --registration. Writes <out>/keep_v1/<stream>.tsv.gz and <out>/MANIFEST.json.

    python scripts/corpus/build_keep_tables.py --pairs work/pairs --scores work/scores \
        --registration work/sarlo80_registration.tsv --out work/corpus
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

STREAMS = ("guso", "terramesh", "sarlo80", "sar1m", "3mos_mr", "3mos_hr")
REASONS = ("featureless", "cloud", "misregistration", "compression", "missing_tile", "no_data")

TILE_PX = {"guso": 512, "terramesh": 264, "sarlo80": 1024, "sar1m": 256, "3mos_mr": 256, "3mos_hr": 256}
CROPS_PER_TILE = {s: (px // 256) ** 2 for s, px in TILE_PX.items()}
REDUNDANCY = {"guso": 1, "terramesh": 1, "sarlo80": 1, "sar1m": 1, "3mos_mr": 4, "3mos_hr": 4}

FEATURELESS_FLAT = 0.25
NO_DATA_FLAT = 0.5
NO_DATA_STREAMS = {"3mos_mr"}
CLOUD_FRACTION = 0.3
CLOUD_STREAMS = {"sar1m", "3mos_mr", "3mos_hr"}
MISREGISTRATION_PX = 2.0
BLOCK_THRESHOLD = {"guso": 0.2404, "terramesh": 0.3021, "sarlo80": 0.1475,
                   "sar1m": 0.6145, "3mos_mr": 0.4069, "3mos_hr": 0.4544}
BLOCK_POPULATION = 0.01  # the compression rule applies only where median(block) exceeds this


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def decide(stream: str, scores: list[dict], registration: dict | None) -> list[list[str]]:
    """Reasons per scored sample, in REASONS order (empty list = keep)."""
    block = np.array([float(r["block"]) for r in scores])
    lapvar = np.array([float(r["lapvar"]) for r in scores])
    compression = float(np.median(block)) > BLOCK_POPULATION
    lap_median = float(np.median(lapvar))
    out = []
    for r, b, lv in zip(scores, block, lapvar):
        flat, cloud = float(r["flat"]), float(r["cloudish"])
        reg = registration.get(r["sample_id"]) if registration is not None else None
        hit = {
            "featureless": flat > FEATURELESS_FLAT,
            "cloud": stream in CLOUD_STREAMS and cloud > CLOUD_FRACTION,
            "misregistration": reg is not None and float(reg["resid_px"]) > MISREGISTRATION_PX,
            "compression": compression and b > BLOCK_THRESHOLD[stream] and lv > lap_median,
            "missing_tile": reg is not None and int(reg["missing_tiles"]) > 0,
            "no_data": stream in NO_DATA_STREAMS and flat > NO_DATA_FLAT,
        }
        out.append([k for k in REASONS if hit[k]])
    return out


def mix_weights(tiles: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Crop equivalents and integer mix weights (parts of 10,000) from per-stream tile counts."""
    crops = {s: tiles[s] * CROPS_PER_TILE[s] // REDUNDANCY[s] for s in STREAMS}
    total = sum(crops.values())
    w = {s: round(10000 * c / total) for s, c in crops.items()}
    w[max(w, key=w.get)] += 10000 - sum(w.values())
    return crops, w


def write_table(path: Path, rows: list[tuple[str, str, int, str]]) -> str:
    """Deterministic gzip (mtime 0, no name). Returns the sha256 of the uncompressed table."""
    text = "sample_id\tscene_id\tkeep\treasons\n" + "".join(f"{a}\t{b}\t{k}\t{c}\n" for a, b, k, c in rows)
    data = text.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=fh, mtime=0) as gz:
        gz.write(data)
    return hashlib.sha256(data).hexdigest()


def build_stream(stream: str, pairs: list[dict], scores: list[dict], registration: dict | None,
                 out: Path) -> dict:
    by_id = {r["sample_id"]: r for r in scores}
    pairs = [p for p in pairs if p["sample_id"] in by_id]
    scored = [by_id[p["sample_id"]] for p in pairs]
    reasons = decide(stream, scored, registration)
    rows = [(p["sample_id"], p["scene_id"], 0 if rs else 1, ",".join(rs)) for p, rs in zip(pairs, reasons)]
    sha = write_table(out / "keep_v1" / f"{stream}.tsv.gz", rows)
    counts = {k: sum(k in rs for rs in reasons) for k in REASONS}
    kept = sum(not rs for rs in reasons)
    return {"file": f"keep_v1/{stream}.tsv.gz", "tsv_sha256": sha, "eligible": len(rows), "kept": kept,
            "removed": len(rows) - kept, "reasons": {k: v for k, v in counts.items() if v}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True, help="directory of <stream>.tsv pair lists")
    ap.add_argument("--scores", type=Path, required=True, help="directory of <stream>.tsv scores")
    ap.add_argument("--registration", type=Path,
                    help="SARLO-80 TSV: sample_id, resid_px (largest control-point residual of the "
                         "SAR-to-optical transform, 0.8 m pixels), missing_tiles (optical tiles not retrieved); "
                         "without it the misregistration and missing_tile rules are not applied")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--streams", nargs="+", default=list(STREAMS), choices=STREAMS)
    args = ap.parse_args(argv)

    if "sarlo80" in args.streams and args.registration is None:
        print("sarlo80: no --registration, so the misregistration and missing_tile rules are not applied")
    registration = ({r["sample_id"]: r for r in read_tsv(args.registration)}
                    if args.registration else None)

    manifest = {"version": "keep_v1", "columns": ["sample_id", "scene_id", "keep", "reasons"],
                "reasons": list(REASONS), "streams": {}}
    for s in args.streams:
        e = build_stream(s, read_tsv(args.pairs / f"{s}.tsv"), read_tsv(args.scores / f"{s}.tsv"),
                         registration if s == "sarlo80" else None, args.out)
        e |= {"tile_px": TILE_PX[s], "crops_per_tile": CROPS_PER_TILE[s], "redundancy": REDUNDANCY[s]}
        manifest["streams"][s] = e
        print(f"{s:10s} eligible {e['eligible']:>9,}  kept {e['kept']:>9,}  removed {e['removed']:>7,}  {e['reasons']}")

    if set(args.streams) == set(STREAMS):
        st = manifest["streams"]
        crops, w = mix_weights({s: st[s]["kept"] for s in STREAMS})
        for s in STREAMS:
            st[s] |= {"crop_equivalents": crops[s], "weight": w[s]}
        manifest["total"] = {k: sum(st[s][k] for s in STREAMS)
                             for k in ("eligible", "kept", "removed", "crop_equivalents")}
        manifest["total"]["reasons"] = {k: sum(st[s]["reasons"].get(k, 0) for s in STREAMS) for k in REASONS}
        print("stage-2 mix weights (parts of 10,000):", w)
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
