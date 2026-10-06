#!/usr/bin/env python
"""Pixel-only scores of the EO target used by the keep-table rules.

Each EO image of a pair list (prepare_terramesh.py, prepare_sar1m.py, list_pairs.py) is read
in [-1, 1] and scored on one 256 x 256 crop: the whole image for 256-px sources, otherwise one
position drawn from a generator seeded with the sample id. Luminance is 0.299 R + 0.587 G + 0.114 B
of (EO + 1) / 2.

  block     mean |horizontal luminance step| on columns 8k+7 minus the mean off them, over the latter
  lapvar    variance of the 3x3 Laplacian of luminance
  flat      fraction of 16 x 16 luminance cells with variance below 1e-5
  cloudish  fraction of pixels with v = max(R,G,B) > 0.4 and (max - min) / (v + 1) * 2 < 0.15

    python scripts/corpus/score_eo_iqa.py --root $CORPUS_ROOT/sar1m --pairs work/pairs/sar1m.tsv \
        --out work/scores/sar1m.tsv
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

RES = 256


@torch.no_grad()
def eo_scores(eo: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample block, lapvar, flat and cloudish of an EO batch [B, 3, 256, 256] in [-1, 1]."""
    w = torch.tensor([0.299, 0.587, 0.114], device=eo.device).view(1, 3, 1, 1)
    g = ((eo + 1) / 2 * w).sum(1, keepdim=True)
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=g.device).view(1, 1, 3, 3)
    lapvar = F.conv2d(g, k, padding=1).var((1, 2, 3))
    d = (g[..., 1:] - g[..., :-1]).abs()
    idx = torch.arange(d.shape[-1], device=g.device)
    on = d[..., (idx % 8) == 7].mean((1, 2, 3))
    off = d[..., (idx % 8) != 7].mean((1, 2, 3))
    block = (on - off) / off.clamp_min(1e-8)
    cells = g.unfold(2, 16, 16).unfold(3, 16, 16).reshape(g.shape[0], -1, 16 * 16)
    flat = (cells.var(-1) < 1e-5).float().mean(1)
    v = eo.amax(1)
    sat = (eo.amax(1) - eo.amin(1)) / (v + 1.0 + 1e-6) * 2.0
    cloudish = ((v > 0.4) & (sat < 0.15)).float().mean((1, 2))
    return {"block": block, "lapvar": lapvar, "flat": flat, "cloudish": cloudish}


def crop(eo: np.ndarray, sample_id: str) -> torch.Tensor:
    h, w = eo.shape[-2:]
    if (h, w) != (RES, RES):
        rng = random.Random(sample_id)
        i, j = rng.randrange(0, h - RES + 1), rng.randrange(0, w - RES + 1)
        eo = eo[:, i:i + RES, j:j + RES]
    return torch.from_numpy(eo.copy()).clamp(-1, 1)


def read_rgb(path: Path) -> np.ndarray | None:
    try:
        a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    except Exception:
        return None
    return a.transpose(2, 0, 1) / 127.5 - 1.0


class EOStream(IterableDataset):
    """Yields (row index, EO crop). Work units (one file, or one TerraMesh tar) are split over workers."""

    def __init__(self, root: Path, rows: list[dict]):
        self.root, self.rows = root, rows
        units: dict = {}
        for i, r in enumerate(rows):
            units.setdefault(r["eo"] if r["eo"].endswith(".tar") else i, []).append(i)
        self.units = list(units.values())

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for unit in self.units[wid::nw]:
            eo_path = self.rows[unit[0]]["eo"]
            if eo_path.endswith(".tar"):
                yield from self._tar(self.root / eo_path, unit)
            else:
                eo = read_rgb(self.root / eo_path)
                if eo is not None:
                    yield unit[0], crop(eo, self.rows[unit[0]]["sample_id"])

    def _tar(self, path: Path, unit: list[int]):
        """TerraMesh S2RGB shard: zarr members, `bands` int16 [1, 3, 264, 264] holding 8-bit RGB."""
        import zarr
        want = {self.rows[i]["sample_id"].split("/", 1)[1]: i for i in unit}
        with tarfile.open(path) as tf:
            for m in tf:
                i = want.get(m.name)
                if i is None:
                    continue
                with zipfile.ZipFile(io.BytesIO(tf.extractfile(m).read())) as zf:
                    g = zarr.open({n: zf.read(n) for n in zf.namelist()}, mode="r")
                eo = np.asarray(g["bands"][0], np.float32) / 127.5 - 1.0
                yield i, crop(eo, self.rows[i]["sample_id"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="source root the pair list is relative to")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    with open(args.pairs, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    res: dict[int, tuple] = {}
    idx, eos = [], []

    def flush():
        s = eo_scores(torch.stack(eos).to(args.device))
        s = {k: v.float().cpu().tolist() for k, v in s.items()}
        for n, i in enumerate(idx):
            res[i] = (round(s["block"][n], 6), round(s["lapvar"][n], 6), round(s["flat"][n], 6),
                      round(s["cloudish"][n], 4))
        idx.clear()
        eos.clear()

    for i, eo in DataLoader(EOStream(args.root, rows), batch_size=None, num_workers=args.workers):
        idx.append(i)
        eos.append(eo)
        if len(idx) == args.batch:
            flush()
            if len(res) % (args.batch * 200) == 0:
                print(f"{len(res):,} / {len(rows):,}", flush=True)
    if idx:
        flush()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("sample_id\tblock\tlapvar\tflat\tcloudish\n")
        for i, r in enumerate(rows):
            if i in res:
                fh.write(r["sample_id"] + "\t" + "\t".join(repr(x) for x in res[i]) + "\n")
    print(f"scored {len(res):,} of {len(rows):,} pairs ({len(rows) - len(res):,} unreadable) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
