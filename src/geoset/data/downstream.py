"""Downstream benchmarks: QXS-SAROPT, SAR2Opt, SAR2EO and SpaceNet6.

``root`` is ``$DATA_ROOT``; dataset ``d`` lives in ``root/d`` and its frozen splits in
``splits/<d>/<split>.txt`` (one EO path per line, relative to ``root/d``). SpaceNet6 pairs point at the
256-px chips written by ``scripts/datasets/build_spacenet6.py`` under ``root/spacenet6/chips``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from ..speckle import speckle_display
from .corpus import _worker, crop_pair, load_eo

DATASETS = ("qxs-saropt", "sar2opt", "sar2eo", "spacenet6")
TEST_RES = {"qxs-saropt": 256, "sar2opt": 512, "sar2eo": 256, "spacenet6": 256}
SPLITS = Path(__file__).resolve().parents[3] / "splits"


def _pair(dataset: str, split: str, rel: str) -> tuple[str, str]:
    """``(SAR, EO)`` paths relative to the dataset root for one split-list line."""
    if dataset == "qxs-saropt":
        return rel.replace("opt_256_oc_0.2/", "sar_256_oc_0.2/"), rel
    if dataset == "sar2opt":
        folder, name = rel.split("/")
        return f"{folder[:-1]}A/{name}", rel
    if dataset == "sar2eo":
        return rel.replace("/EO/", "/SAR/"), rel
    stem = Path(rel).stem.replace("_PS-RGB", "")
    return f"chips/{split}A/{stem}.png", f"chips/{split}B/{stem}.png"


def list_pairs(dataset: str, split: str, root, splits=SPLITS) -> list[tuple[Path, Path]]:
    """``[(sar_path, eo_path)]`` for ``splits/<dataset>/<split>.txt``, in list order."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    base = Path(root) / dataset
    lines = (Path(splits) / dataset / f"{split}.txt").read_text().splitlines()
    return [tuple(base / p for p in _pair(dataset, split, rel)) for rel in lines if rel.strip()]


def load_sar(path) -> np.ndarray:
    """SAR image as float32 ``[1, H, W]`` in [-1, 1] (8-bit luminance)."""
    return np.asarray(Image.open(path).convert("L"), np.float32)[None] / 127.5 - 1.0


def center_crop(a: np.ndarray, res: int) -> np.ndarray:
    """Centered ``res`` x ``res`` window of ``[..., H, W]`` at offset ``((H - res) // 2, (W - res) // 2)``."""
    h, w = a.shape[-2:]
    if h < res or w < res:
        raise ValueError(f"image {h}x{w} is smaller than the {res}-px crop")
    i, j = (h - res) // 2, (w - res) // 2
    return a[..., i:i + res, j:j + res]


class DownstreamTrain(IterableDataset):
    """Infinite fine-tuning stream: ``(speckled SAR [1, 256, 256], EO [3, 256, 256], 0)``.

    One random 256-px crop is shared by SAR and EO; display-domain speckle is applied to the SAR.
    """

    def __init__(self, dataset: str, root, seed: int, splits=SPLITS):
        self.pairs = sorted(list_pairs(dataset, "train", root, splits))
        if not self.pairs:
            raise RuntimeError(f"{dataset}: empty training split")
        self.names = [dataset]
        self.seed = seed

    def __iter__(self):
        wid, nw = _worker()
        rng = random.Random(self.seed + 19 * wid)
        gen = torch.Generator().manual_seed(self.seed + 37 * wid)
        share, n = self.pairs[wid::nw], 0
        while True:
            order = list(share)
            rng.shuffle(order)
            for sar_path, eo_path in order:
                try:
                    sar, eo = load_sar(sar_path), load_eo(eo_path)
                except Exception:
                    continue
                cr = crop_pair(sar, eo, rng)
                if cr is None:
                    continue
                sar, eo = cr
                n += 1
                yield speckle_display(torch.from_numpy(sar.copy()), gen), torch.from_numpy(eo.copy()), 0
            if not n:
                raise RuntimeError(f"{self.names[0]}: DataLoader worker {wid} could not read any pair")


class DownstreamTest:
    """One deterministic pass over a test split: ``(name, SAR [1, r, r], EO [3, r, r])``.

    Items are shuffled with ``seed`` (the order used for the paper's results), optionally restricted to
    the members of another split list (``subset``, e.g. ``"test_first4000"``), and center-cropped to
    ``res`` (default: the evaluation size of the dataset). No speckle is applied.
    """

    def __init__(self, dataset: str, root, seed: int = 20260812, res: int | None = None,
                 subset: str | None = None, splits=SPLITS):
        self.items = [(Path(s).name, s, e) for s, e in sorted(list_pairs(dataset, "test", root, splits))]
        random.Random(seed).shuffle(self.items)
        if subset:
            keep = {Path(e).stem for _, e in list_pairs(dataset, subset, root, splits)}
            self.items = [it for it in self.items if Path(it[2]).stem in keep]
        self.res = TEST_RES[dataset] if res is None else int(res)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        for name, sar_path, eo_path in self.items:
            sar, eo = load_sar(sar_path), load_eo(eo_path)
            if sar.shape[-2:] != eo.shape[-2:]:
                raise ValueError(f"size mismatch for {name}: SAR {sar.shape[-2:]} vs EO {eo.shape[-2:]}")
            sar, eo = center_crop(sar, self.res), center_crop(eo, self.res)
            yield name, torch.from_numpy(sar.copy()), torch.from_numpy(eo.copy())
