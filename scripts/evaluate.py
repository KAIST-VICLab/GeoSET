#!/usr/bin/env python3
"""Score translated EO images on a GeoSET test split with the paper's seven metrics.

  FID    pytorch-fid InceptionV3 pool3 features (2048-d), predictions vs reference EO images.
  DISTS  pyiqa.
  KID    unbiased MMD^2 with the kernel (x.y / 2048 + 1)^3 on the same Inception features,
         averaged over 50 random subsets of min(500, n) images (numpy seed 0).
  DINO   cosine similarity of DINOv3-SAT ViT-L/16 patch tokens at 224 px, averaged over patches.
  LPIPS  VGG, images mapped from [0, 1] to [-1, 1].
  SSIM   torchmetrics, data_range 1.
  PSNR   torchmetrics, data_range 1.

DISTS, DINO, LPIPS, SSIM and PSNR are computed per image and averaged. Predictions are matched
to the test items by file stem (<stem>.png; the <stem>_fake_B.png names of the upstream
pix2pix/CycleGAN test.py are also accepted) and must have the evaluation size of the dataset
(SAR2Opt 512, otherwise 256). The ground truth is centre-cropped to that size and grayscale
ground truth is replicated to RGB. The FID/KID reference is the whole test split for the `test`
list and the ground truth of the scored items for the SAR2EO `test_first4000` list.

  python scripts/evaluate.py --dataset qxs-saropt --pred-dir outputs/qxs-saropt \\
      --data-root $DATA_ROOT --dino-repo dinov3 \\
      --dino-weights dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SPLITS = Path(__file__).resolve().parents[1] / "splits"
SIZE = {"qxs-saropt": 256, "sar2opt": 512, "sar2eo": 256, "spacenet6": 256}
DEFAULT_SPLIT = {"sar2eo": "test_first4000"}
SUFFIXES = {".png", ".jpg", ".jpeg"}
COLUMNS = (("fid", "FID", 1), ("dists", "DISTS", 3), ("kid", "KID", 4), ("dino", "DINO", 3),
           ("lpips", "LPIPS", 3), ("ssim", "SSIM", 3), ("psnr", "PSNR", 2))
DINO_MEAN, DINO_STD = (0.430, 0.411, 0.296), (0.213, 0.156, 0.143)
DINO_HASH = "eadcf0ff"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(SIZE))
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="folder of predicted EO images, one per test item")
    ap.add_argument("--data-root", type=Path, default=os.environ.get("DATA_ROOT"),
                    required="DATA_ROOT" not in os.environ,
                    help="dataset root (default: $DATA_ROOT)")
    ap.add_argument("--split", choices=("test", "test_first4000"), default=None,
                    help="test list in splits/<dataset>/ (default: test_first4000 for sar2eo, "
                         "test otherwise)")
    ap.add_argument("--dino-repo", type=Path, default=None,
                    help="local clone of facebookresearch/dinov3")
    ap.add_argument("--dino-weights", type=Path, default=None,
                    help=f"dinov3_vitl16_pretrain_sat493m-{DINO_HASH}.pth (DINO is skipped "
                         "without it)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <pred-dir>.json)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    return ap.parse_args()


def gt_path(root: Path, dataset: str, split: str, line: str) -> Path:
    """Ground-truth EO image of one split-list line."""
    if dataset == "spacenet6":  # the list names the raw GeoTIFF; the GT is its 256-px chip
        return root / dataset / "chips" / f"{split}B" / (Path(line).stem.replace("_PS-RGB", "") + ".png")
    return root / dataset / line


def load01(path: Path, size: int, exact: bool = False) -> torch.Tensor:
    """[3, size, size] float32 in [0, 1]: the centre crop of the image (exact: the whole image)."""
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w = a.shape[:2]
    if (h, w) != (size, size) and (exact or h < size or w < size):
        raise SystemExit(f"{path} is {w}x{h}; expected {'' if exact else 'at least '}{size}x{size}")
    top, left = (h - size) // 2, (w - size) // 2
    return torch.from_numpy(a.transpose(2, 0, 1))[:, top:top + size, left:left + size]


def batches(paths: list[Path], size: int, batch: int, device: str, exact: bool = False):
    for i in range(0, len(paths), batch):
        yield torch.stack([load01(p, size, exact) for p in paths[i:i + batch]]).to(device)


def match(pred_dir: Path, gts: dict[str, Path]) -> list[tuple[Path, Path]]:
    """(prediction, ground truth) pairs matched by stem, sorted by prediction path."""
    found: dict[str, Path] = {}
    for p in pred_dir.iterdir():
        stem = p.stem[:-len("_fake_B")] if p.stem.endswith("_fake_B") else p.stem
        if p.suffix.lower() in SUFFIXES and stem in gts:
            if stem in found:
                raise SystemExit(f"two predictions for {stem}: {found[stem].name}, {p.name}")
            found[stem] = p
    return sorted((p, gts[stem]) for stem, p in found.items())


def frechet(fa: np.ndarray, fb: np.ndarray) -> float:
    """FID between prediction features fa and reference features fb."""
    from pytorch_fid.fid_score import calculate_frechet_distance
    fa, fb = fa.astype(np.float64), fb.astype(np.float64)
    return float(calculate_frechet_distance(fb.mean(axis=0), np.cov(fb, rowvar=False),
                                            fa.mean(axis=0), np.cov(fa, rowvar=False)))


def kid(fa: np.ndarray, fb: np.ndarray, subset_size: int = 500, n_subsets: int = 50) -> float:
    """Unbiased MMD^2 with the kernel (x.y / d + 1)^3, averaged over random subsets."""
    fa, fb = fa.astype(np.float64), fb.astype(np.float64)
    m, d = min(subset_size, len(fa), len(fb)), fa.shape[1]

    def k(a, b):
        return (a @ b.T / d + 1.0) ** 3

    rng = np.random.default_rng(0)
    vals = np.empty(n_subsets)
    for i in range(n_subsets):
        x = fa[rng.choice(len(fa), m, replace=False)]
        y = fb[rng.choice(len(fb), m, replace=False)]
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
        vals[i] = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                   + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                   - 2.0 * kxy.mean())
    return float(vals.mean())


class DinoSat:
    """DINOv3-SAT ViT-L/16 patch tokens: bf16 backbone, satellite normalisation, 224-px input."""

    def __init__(self, repo: Path, weights: Path, device: str):
        if not weights.name.endswith(f"-{DINO_HASH}.pth"):  # the builder reads the variant from it
            raise SystemExit(f"--dino-weights must keep its released file name (*-{DINO_HASH}.pth)")
        sys.path.insert(0, str(repo.absolute()))
        from dinov3.hub.backbones import dinov3_vitl16
        net = dinov3_vitl16(pretrained=False, weights=str(weights.absolute()))
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net = net.to(device=device, dtype=torch.bfloat16).eval()
        self.mean = torch.tensor(DINO_MEAN).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(DINO_STD).view(1, 3, 1, 1).to(device)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.clamp(0.0, 1.0), size=(224, 224), mode="bilinear",
                          align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return self.net.forward_features(x.to(torch.bfloat16))["x_norm_patchtokens"].float()


class Scorer:
    """The metric networks, loaded once."""

    def __init__(self, device: str, dino: DinoSat | None):
        import lpips
        import pyiqa
        from pytorch_fid.inception import InceptionV3
        self.device, self.dino = device, dino
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        self.dists = pyiqa.create_metric("dists", device=device)
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    def features(self, x: torch.Tensor) -> np.ndarray:
        return self.inception(x)[0].squeeze(3).squeeze(2).cpu().numpy()

    @torch.no_grad()
    def pairs(self, pairs: list[tuple[Path, Path]], size: int, batch: int) -> dict[str, np.ndarray]:
        """Per-pair PSNR, SSIM, LPIPS, DISTS and DINO, and the predictions' Inception features."""
        from torchmetrics.functional.image import (peak_signal_noise_ratio,
                                                   structural_similarity_index_measure)
        per = {k: [] for k in ("psnr", "ssim", "lpips", "dists", "dino", "feat")}
        preds, gts = [p for p, _ in pairs], [g for _, g in pairs]
        for x, y in zip(batches(preds, size, batch, self.device, exact=True),
                        batches(gts, size, batch, self.device)):
            for j in range(len(x)):
                xj, yj = x[j:j + 1], y[j:j + 1]
                per["psnr"].append(float(peak_signal_noise_ratio(xj, yj, data_range=1.0)))
                per["ssim"].append(float(structural_similarity_index_measure(xj, yj, data_range=1.0)))
            per["lpips"] += self.lpips(x * 2 - 1, y * 2 - 1).flatten().cpu().tolist()
            per["dists"] += self.dists(x, y).flatten().cpu().tolist()
            per["feat"].append(self.features(x))
            if self.dino is not None:
                cos = F.cosine_similarity(self.dino.tokens(x), self.dino.tokens(y), dim=-1)
                per["dino"] += cos.mean(dim=1).cpu().tolist()
        per["feat"] = np.concatenate(per["feat"])
        return {k: np.asarray(v) for k, v in per.items()}

    @torch.no_grad()
    def reference(self, paths: list[Path], size: int, batch: int) -> np.ndarray:
        """Inception features of the FID/KID reference images."""
        return np.concatenate([self.features(y) for y in batches(paths, size, batch, self.device)])


def summarize(per: dict[str, np.ndarray], ref: np.ndarray) -> dict:
    """Means of the per-image metrics, plus FID and KID against the reference features."""
    out = {k: float(np.mean(per[k])) if len(per[k]) else None
           for k in ("psnr", "ssim", "lpips", "dists", "dino")}
    out.update(fid=frechet(per["feat"], ref), kid=kid(per["feat"], ref))
    return out


def main() -> int:
    args = parse_args()
    split = args.split or DEFAULT_SPLIT.get(args.dataset, "test")
    size = SIZE[args.dataset]
    list_file = SPLITS / args.dataset / f"{split}.txt"
    if not list_file.is_file():
        raise SystemExit(f"{args.dataset} has no {split} list ({list_file})")
    lines = [s.strip() for s in list_file.read_text().splitlines() if s.strip()]
    gts = {p.stem: p for p in (gt_path(args.data_root, args.dataset, split, s) for s in lines)}

    pairs = match(args.pred_dir, gts)
    if not pairs:
        raise SystemExit(f"no image in {args.pred_dir} is named after an item of {list_file}")
    if len(pairs) < len(gts):
        print(f"warning: {len(pairs)} of {len(gts)} test items have a prediction", file=sys.stderr)
    refs = sorted(gts.values()) if split == "test" else [g for _, g in pairs]

    dino = None
    if args.dino_repo and args.dino_weights:
        dino = DinoSat(args.dino_repo, args.dino_weights, args.device)
    else:
        print("note: --dino-repo/--dino-weights not given, DINO is skipped", file=sys.stderr)

    scorer = Scorer(args.device, dino)
    metrics = summarize(scorer.pairs(pairs, size, args.batch), scorer.reference(refs, size, args.batch))
    result = {"dataset": args.dataset, "split": split, "pred_dir": str(args.pred_dir),
              "n": len(pairs), "n_gt": len(gts), "size": size,
              "metrics": {k: metrics[k] for k, _, _ in COLUMNS}}
    pred_dir = args.pred_dir.absolute()
    out = args.out or pred_dir.with_name(pred_dir.name + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"{args.dataset} ({split})  n={len(pairs)}/{len(gts)}  {size}x{size}  -> {out}")
    print("  ".join(f"{label:>7}" for _, label, _ in COLUMNS))
    print("  ".join("      -" if metrics[k] is None else f"{metrics[k]:7.{nd}f}"
                    for k, _, nd in COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
