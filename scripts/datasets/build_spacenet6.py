#!/usr/bin/env python3
"""Render the SpaceNet6 split (UTM-easting blocks, 450 m guard band) into the 256-px chips used for training
and evaluation.

    python scripts/datasets/build_spacenet6.py --data-root $DATA_ROOT

Reads ``splits/spacenet6/{train,test}.txt`` and the extracted SpaceNet6 training archive under
``$DATA_ROOT/spacenet6/train/AOI_11_Rotterdam``, and writes
``$DATA_ROOT/spacenet6/chips/{trainA,trainB,testA,testB}/<stem>.png`` (A = SAR, B = EO):

* EO: PS-RGB (900 x 900, 8-bit) resized to 256 x 256, bicubic.
* SAR: the HH, HV and VV bands of SAR-Intensity, ``10 * log10(max(x, 1e-3))``, mapped linearly from the
  per-band window ``[lo, hi]`` of ``splits/spacenet6/sar_stretch.json`` to 0-255 (clipped), then resized to
  256 x 256, bilinear.

The window holds the 1st/99th percentiles over the valid (non-zero) pixels of 200 training tiles;
``--verify-stretch`` recomputes it from the training tiles and checks it against the file.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

SPLITS = Path(__file__).resolve().parents[2] / "splits" / "spacenet6"
RES = 256


def sar_to_rgb(a: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``[4, H, W]`` quad-pol intensity (HH, HV, VH, VV) -> ``[H, W, 3]`` uint8 HH/HV/VV."""
    x = a[[0, 1, 3]]
    db = 10.0 * np.log10(np.maximum(x, 1e-3))
    out = (db - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def tile_paths(root: Path, rel: str) -> tuple[Path, Path, str]:
    """EO GeoTIFF, SAR GeoTIFF and chip stem for one split-list line."""
    eo = Path(rel)
    return root / eo, root / rel.replace("PS-RGB", "SAR-Intensity"), eo.stem.replace("_PS-RGB", "")


def compute_stretch(root: Path, train: list[str], n_tiles: int = 200, seed: int = 20260821,
                    p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-band (HH, HV, VV) percentiles of ``10 * log10(x)`` over the valid pixels of sampled training tiles."""
    pick = np.random.default_rng(seed).choice(len(train), size=min(n_tiles, len(train)), replace=False)
    vals = [[], [], []]
    for i in pick:
        with rasterio.open(tile_paths(root, train[i])[1]) as s:
            a = s.read()[[0, 1, 3]].astype(np.float32)
        valid = ~(a <= 0).all(axis=0)
        db = 10.0 * np.log10(np.maximum(a, 1e-3))
        for b in range(3):
            vals[b].append(db[b][valid].ravel()[::37])
    lo = np.array([np.percentile(np.concatenate(v), p_lo) for v in vals], np.float32)
    hi = np.array([np.percentile(np.concatenate(v), p_hi) for v in vals], np.float32)
    return lo, hi


def render(root: Path, out: Path, split: str, rel: str, lo: np.ndarray, hi: np.ndarray) -> str | None:
    eo_path, sar_path, stem = tile_paths(root, rel)
    if not (eo_path.is_file() and sar_path.is_file()):
        return rel
    with rasterio.open(eo_path) as s:
        eo = s.read().transpose(1, 2, 0)
    with rasterio.open(sar_path) as s:
        sar = s.read().astype(np.float32)
    Image.fromarray(eo).resize((RES, RES), Image.Resampling.BICUBIC).save(out / f"{split}B" / f"{stem}.png")
    Image.fromarray(sar_to_rgb(sar, lo, hi)).resize((RES, RES), Image.Resampling.BILINEAR).save(
        out / f"{split}A" / f"{stem}.png")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path, help="$DATA_ROOT (holds spacenet6/train/...)")
    ap.add_argument("--splits", default=SPLITS, type=Path, help="directory with train.txt, test.txt, sar_stretch.json")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--verify-stretch", action="store_true",
                    help="recompute the SAR window from the training tiles and compare it with sar_stretch.json")
    a = ap.parse_args()

    root = a.data_root / "spacenet6"
    out = root / "chips"
    lists = {s: (a.splits / f"{s}.txt").read_text().split() for s in ("train", "test")}
    st = json.loads((a.splits / "sar_stretch.json").read_text())
    lo, hi = np.array(st["lo"], np.float32), np.array(st["hi"], np.float32)
    if a.verify_stretch:
        lo2, hi2 = compute_stretch(root, lists["train"], st["n_tiles"], st["seed"], st["p_lo"], st["p_hi"])
        same = np.array_equal(lo, lo2) and np.array_equal(hi, hi2)
        print(f"recomputed window lo={lo2.tolist()} hi={hi2.tolist()}: {'matches' if same else 'DIFFERS FROM'} "
              f"{a.splits / 'sar_stretch.json'}")
        if not same:
            return 1

    for d in ("trainA", "trainB", "testA", "testB"):
        (out / d).mkdir(parents=True, exist_ok=True)
    missing = []
    with ThreadPoolExecutor(a.workers) as ex:
        for split, rels in lists.items():
            miss = [m for m in ex.map(lambda r, s=split: render(root, out, s, r, lo, hi), rels) if m]
            print(f"{split}: {len(rels) - len(miss)}/{len(rels)} tiles written to {out}")
            missing += miss
    if missing:
        print(f"{len(missing)} tiles missing under {root}, e.g. {missing[0]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
