#!/usr/bin/env python3
"""Translate SAR images to EO images with GeoSET.

A benchmark test split (outputs in the order, batching and seeding of the paper's results):

  python scripts/translate.py --dataset qxs-saropt --data-root $DATA_ROOT --out-dir outputs/qxs-saropt
  python scripts/translate.py --dataset sar2opt --data-root $DATA_ROOT --lora sar2opt --out-dir outputs/sar2opt

A folder of SAR images (every side a multiple of 16):

  python scripts/translate.py --input-dir my_sar --out-dir my_eo

Each output is <out-dir>/<stem>.png. Batch b starts from noise drawn with seed --seed + b, so a batch whose
outputs all exist is skipped and an interrupted run can be restarted with the same command.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoset import (  # noqa: E402
    AutoencoderFlux2, FlowBridgeScheduler, GeoSETPipeline, GeoSETSAREncoder, GeoSETTransformer2DModel,
)
from geoset.data import DATASETS, DownstreamTest, center_crop, load_sar  # noqa: E402
from geoset.pipeline_geoset import HUB_REPO, LORA_ADAPTERS, sar_to_tensor  # noqa: E402

# Batch sizes of the paper's outputs (generalist, fine-tuned); the noise is drawn per batch.
BATCH = {"qxs-saropt": (64, 32), "sar2opt": (16, 16), "sar2eo": (32, 32), "spacenet6": (32, 32)}
DEFAULT_SPLIT = {"sar2eo": "test_first4000"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_pipeline(args) -> GeoSETPipeline:
    root = Path(args.checkpoint)
    if not root.is_dir():
        parts = ["sar_encoder", "vae"] + ([] if args.transformer else ["transformer"])
        patterns = [f"generalist/{p}/*" for p in parts]
        if args.lora in LORA_ADAPTERS:
            patterns.append(f"lora/{args.lora}/*")
        root = Path(snapshot_download(args.checkpoint, allow_patterns=patterns))
    transformer = (GeoSETTransformer2DModel.from_pretrained(args.transformer) if args.transformer
                   else GeoSETTransformer2DModel.from_pretrained(root, subfolder="generalist/transformer"))
    pipe = GeoSETPipeline(
        transformer=transformer,
        sar_encoder=GeoSETSAREncoder.from_pretrained(root, subfolder="generalist/sar_encoder"),
        vae=AutoencoderFlux2.from_pretrained(root, subfolder="generalist/vae"),
        scheduler=FlowBridgeScheduler(),
    ).to(args.device)
    if args.lora:
        adapter = root / "lora" / args.lora if args.lora in LORA_ADAPTERS else Path(args.lora)
        print(f"merged LoRA adapter {adapter}: {pipe.load_lora(adapter)}")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def list_items(args) -> tuple[list[tuple[str, Path]], int]:
    """``[(output stem, SAR path)]`` in generation order, and the center-crop size (0 = none)."""
    if args.input_dir:
        paths = sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in IMAGE_EXT)
        return [(p.stem, p) for p in paths], 0
    split = args.split or DEFAULT_SPLIT.get(args.dataset, "test")
    test = DownstreamTest(args.dataset, args.data_root, subset=None if split == "test" else split)
    return [(Path(eo).stem, Path(sar)) for _, sar, eo in test.items], test.res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=HUB_REPO,
                    help="Hugging Face repo id, or a local directory with the same layout (generalist/, lora/)")
    ap.add_argument("--transformer", help="transformer directory replacing generalist/transformer "
                                          "(e.g. a full fine-tune saved with save_pretrained)")
    ap.add_argument("--lora", help=f"adapter to merge: one of {', '.join(LORA_ADAPTERS)}, or an adapter directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", help="folder of SAR images")
    src.add_argument("--dataset", choices=DATASETS, help="benchmark whose test split is translated")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"), help="benchmark root (default: $DATA_ROOT)")
    ap.add_argument("--split", help="split list (default: test; test_first4000 for sar2eo)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, help="default: the paper's setting per dataset, 16 for --input-dir")
    ap.add_argument("--seed", type=int, default=20260812, help="noise seed of batch 0 (batch b uses seed + b)")
    ap.add_argument("--steps", type=int, default=50, help="number of function evaluations")
    ap.add_argument("--guidance", type=float, default=2.0, help="classifier-free guidance scale")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.dataset and not args.data_root:
        ap.error("--dataset needs --data-root or $DATA_ROOT")

    items, crop = list_items(args)
    bs = args.batch_size or (BATCH[args.dataset][bool(args.lora or args.transformer)] if args.dataset else 16)
    batches = [items[i:i + bs] for i in range(0, len(items), bs)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [(b, chunk) for b, chunk in enumerate(batches)
            if not all((out / f"{stem}.png").exists() for stem, _ in chunk)]
    n_todo = sum(len(chunk) for _, chunk in todo)
    print(f"{len(items)} images, {len(batches)} batches of {bs}, {n_todo} to translate -> {out}")
    if not todo:
        return

    pipe = load_pipeline(args)
    device = torch.device(args.device)
    t0, done = time.time(), 0
    for b, chunk in todo:
        if crop:
            sar = torch.stack([torch.from_numpy(center_crop(load_sar(p), crop).copy()) for _, p in chunk])
        else:
            sar = torch.stack([sar_to_tensor(Image.open(p)) for _, p in chunk])
        generator = torch.Generator(device=device).manual_seed(args.seed + b)
        images = pipe(sar, num_inference_steps=args.steps, guidance_scale=args.guidance, generator=generator).images
        for (stem, _), image in zip(chunk, images):
            image.save(out / f"{stem}.png")
        done += len(chunk)
        print(f"  {done}/{n_todo}  {done / (time.time() - t0):.2f} img/s", flush=True)


if __name__ == "__main__":
    main()
