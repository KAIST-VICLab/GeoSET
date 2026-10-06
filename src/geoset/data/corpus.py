"""Pretraining corpus streams: SAR-only for stage 1, SAR-EO pairs for stage 2.

``root`` is the corpus root, one directory per source with the files as distributed:

    terramesh/{train,val}/{S1RTC,S2RGB}/*.tar
    guso/<region>/<split>/<continent>/*_{SAR,OPT}.tif
    sarlo80/train_x/<chunk>/<shard>/<key>.sar.npy   (+ sarlo80/optic/<chunk>/<shard>/<key>.optic.png for stage 2)
    sar1m/{SAR,OPT}/*, sar1m/paired.json
    3mos/<group>/{sar/sar_<id>.jpg, opt/<terrain>/opt_<id>.jpg}

Every stream is infinite and sharded across DataLoader workers. Crops are taken at ``RES`` and never
resized; speckle is applied to the SAR side only; D4 is applied by the trainer.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import random
import re
import tarfile
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from ..speckle import speckle, speckle_display

RES = 256
KEEP_TABLES = Path(__file__).resolve().parents[3] / "corpus" / "keep_v1"
STAGE1_WEIGHTS = {"terramesh": 6444, "guso": 1853, "sarlo80": 1106, "sar1m": 575, "3mos_hr": 5, "3mos_mr": 17}
STAGE2_WEIGHTS = {"guso": 3872, "terramesh": 2880, "sarlo80": 2062, "sar1m": 1139, "3mos_mr": 36, "3mos_hr": 11}
MOS3_GROUPS = {
    "3mos_mr": tuple(f"SEN/SEN1_region{i}" for i in range(1, 8)) + ("ALOS",),
    "3mos_hr": ("GF3", "Radarsat", "RCM"),
}
MOS3_SHARD_PREFIX = "extracted/"  # 3MOS worker-shard keys are "extracted/<path>", as in the released training runs
DB_LO, DB_HI = -30.0, 5.0  # TerraMesh S1RTC window, in dB
_M4SAR = re.compile(r"\d+_\d+\.jpg")
_SEN12 = re.compile(r"ROIs\w*?_s1_\d+_p\d+\.png")


# ----------------------------------------------------------------------------- shared helpers
def stable_hash(s: str) -> int:
    """Process-independent 64-bit hash used to shard items across workers."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def d4(x: torch.Tensor, k: int) -> torch.Tensor:
    """Dihedral map ``k`` in 0..7: ``k % 4`` quarter turns, then a left-right flip if ``k >= 4``."""
    y = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if k >= 4 else y


def collate_by_channels(batch):
    """Group ``(sar, target, source)`` items by SAR channel count: ``{n: (sar, target, source)}``."""
    groups = {}
    for item in batch:
        groups.setdefault(item[0].shape[0], []).append(item)
    return {n: (torch.stack([i[0] for i in v]), torch.stack([i[1] for i in v]),
                torch.tensor([i[2] for i in v], dtype=torch.long)) for n, v in groups.items()}


def load_eo(path) -> np.ndarray:
    """EO image as float32 ``[3, H, W]`` in [-1, 1]."""
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    return a.transpose(2, 0, 1) / 127.5 - 1.0


def crop_pair(sar: np.ndarray, eo: np.ndarray, rng: random.Random):
    """One random ``RES`` crop at the same position in both arrays; None if the grids differ or are too small."""
    if sar.shape[-2:] != eo.shape[-2:]:
        return None
    h, w = sar.shape[-2:]
    if h < RES or w < RES:
        return None
    i = rng.randrange(0, h - RES + 1)
    j = rng.randrange(0, w - RES + 1)
    return sar[..., i:i + RES, j:j + RES], eo[..., i:i + RES, j:j + RES]


def load_keep(keep_dir=KEEP_TABLES) -> dict[str, dict[str, str]]:
    """``{stream: {sample_id: scene_id}}`` over the kept rows of ``<keep_dir>/<stream>.tsv.gz``."""
    out = {}
    for name in STAGE2_WEIGHTS:
        with gzip.open(Path(keep_dir) / f"{name}.tsv.gz", "rt", encoding="utf-8") as f:
            if next(f).rstrip("\n").split("\t") != ["sample_id", "scene_id", "keep", "reasons"]:
                raise ValueError(f"{keep_dir}/{name}.tsv.gz: unexpected header")
            rows = (line.rstrip("\n").split("\t") for line in f)
            out[name] = {r[0]: r[1] for r in rows if r[2] == "1"}
    return out


def _worker() -> tuple[int, int]:
    info = torch.utils.data.get_worker_info()
    return (info.id, info.num_workers) if info else (0, 1)


def _listing(root: Path, suffix: str, sub: str = "") -> list[str]:
    """Sorted POSIX paths, relative to ``root``, of the files under ``root/sub`` ending in ``suffix``."""
    out = []
    for d, _, files in os.walk(root / sub, followlinks=True):
        rel = Path(d).relative_to(root).as_posix()
        out += [f if rel == "." else f"{rel}/{f}" for f in files if f.endswith(suffix)]
    if not out:
        raise RuntimeError(f"no *{suffix} files under {root / sub}")
    return sorted(out)


def _sar1m_rows(root: Path) -> list[tuple[str, Path, Path]]:
    """``(sample_id, sar_path, eo_path)`` for every entry of SAR-1M's ``paired.json``, in file order."""
    rows = []
    for e in json.loads((root / "paired.json").read_text()):
        name = os.path.basename(e["sar"])
        if _M4SAR.fullmatch(name):
            sub = "m4sar"
        elif _SEN12.fullmatch(name):
            sub = "sen12"
        else:
            raise ValueError(f"unrecognised SAR-1M file name {name!r}")
        rows.append((f"sar1m/{sub}/{os.path.splitext(name)[0]}", root / e["sar"], root / e["optical"]))
    return rows


def _terramesh_shards(root: Path) -> list[tuple[str, str]]:
    shards = [(split, p.name) for split in ("train", "val") for p in sorted((root / split / "S1RTC").glob("*.tar"))]
    if not shards:
        raise RuntimeError(f"no S1RTC shards under {root}")
    return shards


def _bands(buf: bytes, zarr) -> np.ndarray:
    """``bands[0]`` of a zipped zarr sample, as float32."""
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        z = zarr.open({n: zf.read(n) for n in zf.namelist()}, mode="r")
    return np.asarray(z["bands"][0], np.float32)


def _db_to_unit(x: torch.Tensor) -> torch.Tensor:
    return ((x - DB_LO) / (DB_HI - DB_LO) * 2 - 1).clamp(-1, 1)


def _empty(name: str, wid: int):
    return RuntimeError(f"{name}: DataLoader worker {wid} could not read any sample")


def _mix(parts, weights, seed):
    its = [iter(p) for p in parts]
    rng = random.Random(seed + 43 * _worker()[0])
    idx = [i for i, w in enumerate(weights) for _ in range(w)]
    while True:
        i = rng.choice(idx)
        a, b = next(its[i])
        yield a, b, i


# ----------------------------------------------------------------------------- stage 1: SAR only
class _SARFiles(IterableDataset):
    """Shuffled passes over one source's files; yields ``(speckled, clean)`` SAR crops in [-1, 1].

    Subclasses set ``seed_mult``, the worker-id multipliers of the Python RNG and torch generator seeds.
    """

    shard_prefix = ""

    def __init__(self, items, seed: int):
        self.items, self.seed = items, seed

    def shard(self, item) -> int:
        return stable_hash(self.shard_prefix + item)

    def __iter__(self):
        wid, nw = _worker()
        rng = random.Random(self.seed + self.seed_mult[0] * wid)
        gen = torch.Generator().manual_seed(self.seed + self.seed_mult[1] * wid)
        items, n = self.items[:], 0
        while True:
            rng.shuffle(items)
            for it in items:
                if nw > 1 and self.shard(it) % nw != wid:
                    continue
                out = self.load(it, rng, gen)
                if out is not None:
                    n += 1
                    yield out
            if not n:
                raise _empty(type(self).__name__, wid)


class GusoSAR(_SARFiles):
    seed_mult = (19, 37)

    def __init__(self, root, seed: int):
        self.root = Path(root)
        super().__init__(_listing(self.root, "_SAR.tif"), seed)

    def load(self, f, rng, gen):
        try:
            a = np.asarray(Image.open(self.root / f).convert("L"), np.float32)
        except Exception:
            return None
        if a.shape[0] < RES or a.shape[1] < RES:
            return None
        i = rng.randrange(0, a.shape[0] - RES + 1)
        j = rng.randrange(0, a.shape[1] - RES + 1)
        clean = torch.from_numpy(a[i:i + RES, j:j + RES].copy() / 127.5 - 1.0)[None]
        return speckle_display(clean, gen), clean


class SarloSAR(_SARFiles):
    seed_mult = (13, 29)

    def __init__(self, root, seed: int):
        self.root = Path(root)
        super().__init__(_listing(self.root, ".sar.npy", "train_x"), seed)

    def shard(self, f) -> int:
        return zlib.crc32(f.encode())

    def load(self, f, rng, gen):
        try:
            amp = np.abs(np.load(self.root / f)).astype(np.float32)
        except Exception:
            return None
        i = rng.randrange(0, amp.shape[0] - RES)
        j = rng.randrange(0, amp.shape[1] - RES)
        tile = torch.from_numpy(amp[i:i + RES, j:j + RES])[None]
        p99 = torch.quantile(tile, 0.99).clamp_min(1e-6)  # stretch from the clean crop
        xa = speckle(tile, "amplitude", gen)
        return (xa / p99 * 2 - 1).clamp(-1, 1), (tile / p99 * 2 - 1).clamp(-1, 1)


class Sar1mSAR(_SARFiles):
    seed_mult = (1, 7)

    def __init__(self, root, seed: int):
        super().__init__([(sid, sar) for sid, sar, _ in _sar1m_rows(Path(root))], seed)

    def shard(self, item) -> int:
        return stable_hash(item[0])

    def load(self, item, rng, gen):
        return _display_tile(item[1], gen)


class Mos3SAR(_SARFiles):
    seed_mult = (23, 41)
    shard_prefix = MOS3_SHARD_PREFIX

    def __init__(self, root, stream: str, seed: int):
        self.root = Path(root)
        groups = MOS3_GROUPS[stream]
        super().__init__([f for f in _listing(self.root, ".jpg")
                          if any(f.startswith(f"{g}/sar/") for g in groups)], seed)

    def load(self, f, rng, gen):
        return _display_tile(self.root / f, gen)


def _display_tile(path, gen):
    """A 256-px 8-bit SAR tile and its display-speckled copy; None if unreadable or not 256 px."""
    try:
        a = np.asarray(Image.open(path).convert("L"), np.float32)
    except Exception:
        return None
    if a.shape != (RES, RES):
        return None
    clean = torch.from_numpy(a / 127.5 - 1.0)[None]
    return speckle_display(clean, gen), clean


class TerraMeshSAR(IterableDataset):
    """All S1RTC samples, walked shard by shard; yields 2-channel (VV, VH) ``(speckled, clean)`` crops."""

    def __init__(self, root, seed: int):
        self.root, self.seed = Path(root), seed
        self.shards = _terramesh_shards(self.root)

    def __iter__(self):
        import zarr

        wid, nw = _worker()
        rng = random.Random(self.seed + 17 * wid)
        gen = torch.Generator().manual_seed(self.seed + 31 * wid)
        shards, n = [s for k, s in enumerate(self.shards) if k % nw == wid], 0
        while True:
            rng.shuffle(shards)
            for split, name in shards:
                try:
                    tar = tarfile.open(self.root / split / "S1RTC" / name)
                except Exception:
                    continue
                with tar:
                    for m in tar:
                        try:
                            s1 = _bands(tar.extractfile(m).read(), zarr)  # [2, 264, 264] dB
                        except Exception:
                            continue
                        i = rng.randrange(0, s1.shape[1] - RES)
                        j = rng.randrange(0, s1.shape[2] - RES)
                        db = torch.from_numpy(s1[:, i:i + RES, j:j + RES].copy())
                        xa = speckle(db, "db", gen)
                        n += 1
                        yield _db_to_unit(xa), _db_to_unit(db)
            if not n:
                raise _empty("TerraMeshSAR", wid)


class SARCorpus(IterableDataset):
    """Stage-1 stream: ``(speckled SAR, clean SAR, source index)`` drawn per ``STAGE1_WEIGHTS``."""

    names = ("terramesh", "guso", "sarlo80", "sar1m", "3mos_hr", "3mos_mr")

    def __init__(self, root, seed: int):
        root = Path(root)
        self.parts = [
            TerraMeshSAR(root / "terramesh", seed),
            GusoSAR(root / "guso", seed + 1),
            SarloSAR(root / "sarlo80", seed + 2),
            Sar1mSAR(root / "sar1m", seed + 3),
            Mos3SAR(root / "3mos", "3mos_hr", seed + 4),
            Mos3SAR(root / "3mos", "3mos_mr", seed + 5),
        ]
        self.weights = tuple(STAGE1_WEIGHTS[n] for n in self.names)
        self.seed = seed

    def __iter__(self):
        return _mix(self.parts, self.weights, self.seed)


# ----------------------------------------------------------------------------- stage 2: SAR-EO pairs
class _Pairs(IterableDataset):
    """Hierarchical scene -> tile sampling over one source's kept pairs; yields ``(speckled SAR, EO)``.

    Each pass visits every scene of the worker's share once, in random order, and draws one tile from
    it. Subclasses set ``seed_mult`` as in ``_SARFiles``.
    """

    shard_prefix = ""

    def __init__(self, items, keep: dict[str, str], seed: int, name: str):
        self.by_scene = {}
        for it in items:
            scene = keep.get(self.key(it))
            if scene is not None:
                self.by_scene.setdefault(scene, []).append(it)
        if not self.by_scene:
            raise RuntimeError(f"{name}: no kept pair")
        self.seed = seed

    def key(self, item) -> str:
        return item

    def __iter__(self):
        wid, nw = _worker()
        rng = random.Random(self.seed + self.seed_mult[0] * wid)
        gen = torch.Generator().manual_seed(self.seed + self.seed_mult[1] * wid)
        if nw == 1 or len(self.by_scene) < nw:
            share = list(self.by_scene)
        else:
            share = [s for s in self.by_scene if stable_hash(self.shard_prefix + s) % nw == wid]
        n = 0
        while True:
            scenes = share[:]
            rng.shuffle(scenes)
            for s in scenes:
                out = self.load(rng.choice(self.by_scene[s]), rng, gen)
                if out is not None:
                    n += 1
                    yield out
            if not n:
                raise _empty(type(self).__name__, wid)


class GusoPairs(_Pairs):
    seed_mult = (19, 37)

    def __init__(self, root, keep: dict[str, str], seed: int):
        self.root = Path(root)
        super().__init__(sorted(keep), keep, seed, "guso")

    def load(self, f, rng, gen):
        try:
            eo = load_eo(self.root / f.replace("_SAR.tif", "_OPT.tif"))
            im = Image.open(self.root / f)
            if im.mode == "F":  # float tiles in [0, 1]
                a = np.asarray(im, np.float32) * 2.0 - 1.0
            else:
                a = np.asarray(im.convert("L"), np.float32) / 127.5 - 1.0
        except Exception:
            return None
        cr = crop_pair(a[None], eo, rng)
        if cr is None:
            return None
        sar, eo = cr
        return speckle_display(torch.from_numpy(sar.copy()), gen), torch.from_numpy(eo.copy())


class SarloPairs(_Pairs):
    seed_mult = (13, 29)

    def __init__(self, root, keep: dict[str, str], seed: int):
        self.root = Path(root)
        super().__init__(sorted(keep), keep, seed, "sarlo80")

    def load(self, f, rng, gen):
        rel = Path(f).relative_to("train_x")
        try:
            eo = load_eo(self.root / "optic" / rel.with_name(rel.name.replace(".sar.npy", ".optic.png")))
            amp = np.abs(np.load(self.root / f)).astype(np.float32)[None]
        except Exception:
            return None
        cr = crop_pair(amp, eo, rng)
        if cr is None:
            return None
        tile, eo = cr
        tile = torch.from_numpy(tile.copy())
        p99 = torch.quantile(tile, 0.99).clamp_min(1e-6)  # stretch from the clean crop
        xa = speckle(tile, "amplitude", gen)
        return (xa / p99 * 2 - 1).clamp(-1, 1), torch.from_numpy(eo.copy())


class Sar1mPairs(_Pairs):
    seed_mult = (7, 11)

    def __init__(self, root, keep: dict[str, str], seed: int):
        super().__init__(_sar1m_rows(Path(root)), keep, seed, "sar1m")

    def key(self, item) -> str:
        return item[0]

    def load(self, item, rng, gen):
        return _display_pair(item[1], item[2], rng, gen)


class Mos3Pairs(_Pairs):
    seed_mult = (23, 41)
    shard_prefix = MOS3_SHARD_PREFIX

    def __init__(self, root, keep: dict[str, str], seed: int, name: str):
        self.root = Path(root)
        opt = {}
        for f in _listing(self.root, ".jpg"):
            parts = f.split("/")
            if "opt" in parts:
                k = parts.index("opt")
                opt["/".join(parts[:k]), parts[-1].replace("opt_", "").replace(".jpg", "")] = f
        pairs = []
        for f in sorted(keep):
            parts = f.split("/")
            eo = opt.get(("/".join(parts[:-2]), parts[-1].replace("sar_", "").replace(".jpg", "")))
            if eo is not None:
                pairs.append((f, eo))
        super().__init__(pairs, keep, seed, name)

    def key(self, item) -> str:
        return item[0]

    def load(self, item, rng, gen):
        return _display_pair(self.root / item[0], self.root / item[1], rng, gen)


def _display_pair(sar_path, eo_path, rng, gen):
    try:
        eo = load_eo(eo_path)
        a = np.asarray(Image.open(sar_path).convert("L"), np.float32)
    except Exception:
        return None
    cr = crop_pair(a[None] / 127.5 - 1.0, eo, rng)
    if cr is None:
        return None
    sar, eo = cr
    return speckle_display(torch.from_numpy(sar.copy()), gen), torch.from_numpy(eo.copy())


class TerraMeshPairs(IterableDataset):
    """Kept S1RTC/S2RGB pairs, addressed by zarr member name and walked shard by shard."""

    def __init__(self, root, keep: dict[str, str], seed: int):
        if not keep:
            raise RuntimeError("terramesh: no kept pair")
        self.root, self.keep, self.seed = Path(root), keep, seed
        self.shards = _terramesh_shards(self.root)

    def __iter__(self):
        import zarr

        wid, nw = _worker()
        rng = random.Random(self.seed + 17 * wid)
        gen = torch.Generator().manual_seed(self.seed + 31 * wid)
        shards, n = [s for k, s in enumerate(self.shards) if k % nw == wid], 0
        while True:
            rng.shuffle(shards)
            for split, name in shards:
                try:
                    sar_tar = tarfile.open(self.root / split / "S1RTC" / name)
                    eo_tar = tarfile.open(self.root / split / "S2RGB" / name)
                except Exception:
                    continue
                with sar_tar, eo_tar:
                    eo_members = {m.name: m for m in eo_tar}
                    for m in sar_tar:
                        if f"{split}/{m.name}" not in self.keep or m.name not in eo_members:
                            continue
                        try:
                            s1 = _bands(sar_tar.extractfile(m).read(), zarr)               # [2, 264, 264] dB
                            s2 = _bands(eo_tar.extractfile(eo_members[m.name]).read(), zarr)  # [3, 264, 264] 0-255
                        except Exception:
                            continue
                        cr = crop_pair(s1, s2 / 127.5 - 1.0, rng)
                        if cr is None:
                            continue
                        db, eo = cr
                        xa = speckle(torch.from_numpy(db.copy()), "db", gen)
                        n += 1
                        yield _db_to_unit(xa), torch.from_numpy(eo.copy()).clamp(-1, 1)
            if not n:
                raise _empty("TerraMeshPairs", wid)


class PairedCorpus(IterableDataset):
    """Stage-2 stream: ``(speckled SAR, EO, source index)`` drawn per ``STAGE2_WEIGHTS`` from the kept pairs."""

    names = ("guso", "terramesh", "sarlo80", "sar1m", "3mos_mr", "3mos_hr")

    def __init__(self, root, seed: int, keep_dir=KEEP_TABLES):
        root, keep = Path(root), load_keep(keep_dir)
        self.parts = [
            GusoPairs(root / "guso", keep["guso"], seed),
            TerraMeshPairs(root / "terramesh", keep["terramesh"], seed + 1),
            SarloPairs(root / "sarlo80", keep["sarlo80"], seed + 2),
            Sar1mPairs(root / "sar1m", keep["sar1m"], seed + 3),
            Mos3Pairs(root / "3mos", keep["3mos_mr"], seed + 4, "3mos_mr"),
            Mos3Pairs(root / "3mos", keep["3mos_hr"], seed + 5, "3mos_hr"),
        ]
        self.weights = tuple(STAGE2_WEIGHTS[n] for n in self.names)
        self.seed = seed

    def __iter__(self):
        return _mix(self.parts, self.weights, self.seed)
