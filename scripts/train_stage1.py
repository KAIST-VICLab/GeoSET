#!/usr/bin/env python3
"""Stage 1: speckle-robust SAR encoder.

The FLUX.2 encoder trunk with channel-specific stems learns to reconstruct the clean SAR image from its
speckle-perturbed version through the frozen FLUX.2 decoder (L1 loss). Paper settings, 4 GPUs:

  accelerate launch --num_processes 4 --mixed_precision no --dynamo_backend no scripts/train_stage1.py \\
      --corpus-root $CORPUS_ROOT --vae weights/GeoSET/generalist/vae --out runs/stage1

Checkpoints are written to <out>/checkpoints/step_XXXXXXX/ and load with GeoSETSAREncoder.from_pretrained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoset import AutoencoderFlux2, GeoSETSAREncoder  # noqa: E402
from geoset.data import SARCorpus, collate_by_channels, d4  # noqa: E402


class Reconstruction(nn.Module):
    """Trainable SAR encoder followed by the frozen decoder: ``[B, 3|2, H, W]`` -> ``[B, 3|2, H, W]``."""

    def __init__(self, encoder: GeoSETSAREncoder, vae: AutoencoderFlux2):
        super().__init__()
        self.encoder = encoder
        self.vae = [vae]  # frozen; kept out of the module tree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.reconstruct(x, self.vae[0])


def reconstruction_loss(model, groups, g_aug, device) -> torch.Tensor:
    """L1 loss of one micro-batch ``{channels: (speckled, clean, source)}``, averaged over all items."""
    total, n_items = 0.0, 0
    for n, (x_in, x, _) in sorted(groups.items()):
        x_in, x = x_in.to(device, non_blocking=True), x.to(device, non_blocking=True)
        k = int(torch.randint(8, (1,), generator=g_aug).item())
        x_in, x = d4(x_in, k), d4(x, k)
        if n == 1:  # single-channel SAR goes through the three-channel stem
            x_in, x = x_in.repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.l1_loss(model(x_in).float(), x.float())
        total = total + loss * x_in.shape[0]
        n_items += x_in.shape[0]
    return total / max(n_items, 1)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-root", required=True, help="pretraining corpus root (docs/CORPUS.md)")
    ap.add_argument("--vae", required=True, help="AutoencoderFlux2 directory (generalist/vae of the GeoSET release)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500, help="linear warm-up length in scheduler steps")
    ap.add_argument("--micro", type=int, default=36, help="per-GPU micro-batch")
    ap.add_argument("--accum", type=int, default=3, help="gradient accumulation steps")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    acc = Accelerator(gradient_accumulation_steps=args.accum)
    device, out = acc.device, Path(args.out)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    vae = AutoencoderFlux2.from_pretrained(args.vae).to(device).requires_grad_(False)
    model = Reconstruction(GeoSETSAREncoder.from_autoencoder(vae), vae).to(device)
    data = SARCorpus(args.corpus_root, seed=20260805 + 1000 * acc.process_index)
    loader = DataLoader(data, batch_size=args.micro, num_workers=args.workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=6, collate_fn=collate_by_channels)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    model, opt, sched = acc.prepare(model, opt, sched)
    model.train()

    if acc.is_main_process:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        run = {**vars(args), "world": acc.num_processes, "global_batch": args.micro * args.accum * acc.num_processes}
        (out / "run_args.json").write_text(json.dumps(run, indent=2) + "\n")
        acc.print(f"[stage1] {sum(p.numel() for p in trainable):,} trainable parameters, "
                  f"global batch {run['global_batch']}")
    log = (out / "metrics.jsonl").open("a") if acc.is_main_process else None
    g_aug = torch.Generator().manual_seed(20260806 + acc.process_index)
    step, t_win, it = 0, time.time(), iter(loader)
    while step < args.steps:
        groups = next(it)
        with acc.accumulate(model):
            loss = reconstruction_loss(model, groups, g_aug, device)
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if not acc.sync_gradients:
            continue
        step += 1
        if acc.is_main_process and step % args.log_every == 0:
            now = time.time()
            line = {"step": step, "loss": float(loss.detach()), "lr": sched.get_last_lr()[0],
                    "img_per_s": args.log_every * args.micro * args.accum * acc.num_processes / (now - t_win)}
            t_win = now
            log.write(json.dumps(line) + "\n")
            log.flush()
            print(f"[{step}] loss {line['loss']:.4f} lr {line['lr']:.2e} {line['img_per_s']:.1f} img/s", flush=True)
        if step % args.ckpt_every == 0 or step == args.steps:
            acc.wait_for_everyone()
            if acc.is_main_process:
                acc.unwrap_model(model).encoder.save_pretrained(out / "checkpoints" / f"step_{step:07d}")
    if log:
        log.close()


if __name__ == "__main__":
    main()
