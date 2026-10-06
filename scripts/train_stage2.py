#!/usr/bin/env python3
"""Stage 2 (generalist pretraining) and stage 3 (LoRA or full fine-tuning on one benchmark).

The GeoSET transformer learns the flow-matching velocity from Gaussian noise to the EO latent of the frozen FLUX.2
autoencoder, conditioned on the latent of the frozen stage-1 SAR encoder. Paper settings:

  # stage 2, 4 GPUs, initialised from FLUX.2-klein-base-4B
  accelerate launch --num_processes 4 --mixed_precision no --dynamo_backend no scripts/train_stage2.py \\
      --flux2-base flux-2-klein-base-4b.safetensors --corpus-root $CORPUS_ROOT \\
      --sar-encoder weights/GeoSET/generalist/sar_encoder --vae weights/GeoSET/generalist/vae --out runs/stage2

  # stage 3, 1 GPU, from the generalist (--lora or --full-finetune)
  accelerate launch --num_processes 1 --mixed_precision no --dynamo_backend no scripts/train_stage2.py \\
      --downstream sar2opt --data-root $DATA_ROOT --init-from weights/GeoSET/generalist/transformer --lora \\
      --sar-encoder weights/GeoSET/generalist/sar_encoder --vae weights/GeoSET/generalist/vae --out runs/lora_sar2opt

Checkpoints are written to <out>/checkpoints/step_XXXXXXX/: the EMA transformer (loads with
GeoSETTransformer2DModel.from_pretrained) or, with --lora, the EMA adapter (geoset.lora.merge_lora).
<out>/state holds the resumable training state (--resume).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoset import AutoencoderFlux2, GeoSETSAREncoder, GeoSETTransformer2DModel, load_flux2_base  # noqa: E402
from geoset.data import DATASETS, KEEP_TABLES, DownstreamTrain, PairedCorpus, collate_by_channels, d4  # noqa: E402
from geoset.lora import add_lora, base_fingerprint, check_lora_grads, lora_parameters, save_lora  # noqa: E402

DEFAULTS = {  # paper settings per mode
    "pretrain": dict(steps=500_000, lr=1e-4, warmup=1000, micro=32, accum=2, workers=16, ckpt_every=25_000,
                     state_every=2000),
    "lora": dict(steps=20_000, lr=1e-4, warmup=100, micro=16, accum=1, workers=12, ckpt_every=10_000,
                 state_every=5000),
    "full": dict(steps=20_000, lr=2e-5, warmup=100, micro=16, accum=1, workers=12, ckpt_every=10_000,
                 state_every=5000),
}
VAE_CHUNK = 32


class EMA:
    """Exponential moving average with warm-up: decay ``min(decay, (1 + step) / (10 + step))``."""

    def __init__(self, tensors, decay: float):
        self.tensors, self.decay = tensors, float(decay)
        self.shadow = {k: v.detach().clone().float() for k, v in tensors().items()}

    @torch.no_grad()
    def update(self, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        current = self.tensors()
        for k, s in self.shadow.items():
            s.mul_(d).add_(current[k].detach().float(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return dict(self.shadow)

    def load_state_dict(self, state: dict) -> None:
        for k, s in self.shadow.items():
            s.copy_(state[k])


def optimizer_and_schedule(params, lr: float, warmup: int):
    """AdamW and a linear learning-rate warm-up (both are prepared and stepped by accelerate)."""
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warmup)))


def flow_matching_loss(model, sar_encoder, encode_eo, groups, g_aug, g_eps, cond_dropout: float, device):
    """Loss of one micro-batch ``{channels: (sar, eo, source)}``: D4, frozen encoders, noise, condition dropout."""
    z_s, z_e = [], []
    for _, (sar, eo, _) in sorted(groups.items()):
        sar, eo = sar.to(device, non_blocking=True), eo.to(device, non_blocking=True)
        k = int(torch.randint(8, (1,), generator=g_aug).item())
        sar, eo = d4(sar, k), d4(eo, k)
        with torch.no_grad():
            z_s.append(sar_encoder(sar))
            eo = eo.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
            z_e.append(torch.cat([encode_eo(c) for c in eo.split(VAE_CHUNK)]).float())
    z_s, z_e = torch.cat(z_s), torch.cat(z_e)
    b = z_e.shape[0]
    t = torch.sigmoid(torch.randn(b, device=device, generator=g_eps)).clamp(0.0, 1.0 - 1e-3)  # logit-normal(0, 1)
    eps = torch.randn(z_e.shape, device=device, generator=g_eps)
    tb = t.view(-1, 1, 1, 1)
    z_t = (1.0 - tb) * eps + tb * z_e
    target = (z_e - z_t) / (1.0 - tb).clamp_min(1e-3)
    if cond_dropout > 0:  # null condition = zero SAR latent, per sample
        z_s = z_s * (torch.rand(b, 1, 1, 1, device=device, generator=g_eps) >= cond_dropout)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = model(z_t, t, z_s)
    return F.mse_loss(v.float(), target)


def save_checkpoint(path: Path, net, ema: EMA, args) -> None:
    """EMA weights: an adapter directory (--lora) or a GeoSETTransformer2DModel directory."""
    if args.lora:
        save_lora(path, ema.state_dict(), args.lora_rank, args.lora_alpha, base_fingerprint(net))
        return
    with torch.device("meta"):
        model = GeoSETTransformer2DModel.from_config(net.config)
    model.load_state_dict(ema.state_dict(), strict=True, assign=True)
    model.save_pretrained(path, max_shard_size="20GB")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sar-encoder", required=True, help="GeoSETSAREncoder directory (stage-1 checkpoint)")
    ap.add_argument("--vae", required=True, help="AutoencoderFlux2 directory")
    ap.add_argument("--flux2-base", help="stage 2: flux-2-klein-base-4b.safetensors of FLUX.2-klein-base-4B")
    ap.add_argument("--corpus-root", help="stage 2: pretraining corpus root (docs/CORPUS.md)")
    ap.add_argument("--keep-tables", default=str(KEEP_TABLES), help="stage 2: keep-table directory")
    ap.add_argument("--downstream", choices=DATASETS, help="stage 3: benchmark to fine-tune on")
    ap.add_argument("--data-root", help="stage 3: $DATA_ROOT")
    ap.add_argument("--init-from", help="stage 3: GeoSETTransformer2DModel directory (the generalist)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--lora", action="store_true", help="stage 3: train LoRA adapters only")
    mode.add_argument("--full-finetune", action="store_true", help="stage 3: train every transformer weight")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    for name, kind in (("steps", int), ("lr", float), ("warmup", int), ("micro", int), ("accum", int),
                       ("workers", int), ("ckpt-every", int), ("state-every", int)):
        ap.add_argument(f"--{name}", type=kind, help="default depends on the mode (see DEFAULTS)")
    ap.add_argument("--cond-dropout", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/state")
    ap.add_argument("--no-compile", action="store_true", help="run the transformer and the EO encoder eagerly")
    args = ap.parse_args(argv)
    if args.downstream:
        if not (args.data_root and args.init_from and (args.lora or args.full_finetune)):
            ap.error("--downstream needs --data-root, --init-from and one of --lora / --full-finetune")
    elif args.lora or args.full_finetune or args.init_from or not (args.flux2_base and args.corpus_root):
        ap.error("stage 2 needs --flux2-base and --corpus-root; --init-from, --lora and --full-finetune "
                 "need --downstream")
    for k, v in DEFAULTS["lora" if args.lora else "full" if args.full_finetune else "pretrain"].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    acc = Accelerator(gradient_accumulation_steps=args.accum,
                      kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3600))])
    device, rank = acc.device, acc.process_index
    out = Path(args.out)
    state_dir = out / "state"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    resume_step = json.loads((state_dir / "progress.json").read_text())["step"] if args.resume else 0

    sar_encoder = GeoSETSAREncoder.from_pretrained(args.sar_encoder).to(device).eval().requires_grad_(False)
    vae = AutoencoderFlux2.from_pretrained(args.vae, torch_dtype=torch.bfloat16).to(device)
    vae = vae.to(memory_format=torch.channels_last).requires_grad_(False)
    encode_eo = vae.encode if args.no_compile else torch.compile(
        vae.encode, mode="max-autotune-no-cudagraphs", dynamic=False)

    if args.downstream:
        model = GeoSETTransformer2DModel.from_pretrained(args.init_from)
    else:
        model = GeoSETTransformer2DModel()
        report = load_flux2_base(model, args.flux2_base)
        acc.print(f"[init] {report['loaded_tensors']} tensors from {args.flux2_base}, SAR stream cloned from the "
                  f"image stream ({report['loaded_fraction']:.2%} of the checkpoint parameters used)")
    if args.lora:
        targets = add_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        acc.print(f"[lora] rank {args.lora_rank}, alpha {args.lora_alpha:g} on {len(targets)} projections")
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not args.no_compile:
        model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs", dynamic=False)

    seed = args.seed + 1000 * rank + resume_step
    data = (DownstreamTrain(args.downstream, args.data_root, seed) if args.downstream
            else PairedCorpus(args.corpus_root, seed, args.keep_tables))
    loader = DataLoader(data, batch_size=args.micro, num_workers=args.workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=6, collate_fn=collate_by_channels)
    opt, sched = optimizer_and_schedule(trainable, args.lr, args.warmup)
    model, opt, sched = acc.prepare(model, opt, sched)
    net = acc.unwrap_model(model)
    ema = EMA((lambda: lora_parameters(net)) if args.lora else net.state_dict, args.ema)
    acc.register_for_checkpointing(ema)
    step = resume_step
    if args.resume:
        acc.load_state(str(state_dir))

    if acc.is_main_process:
        run = {**vars(args), "world": acc.num_processes, "global_batch": args.micro * args.accum * acc.num_processes}
        (out / "run_args.json").write_text(json.dumps(run, indent=2) + "\n")
        acc.print(f"[train] {sum(p.numel() for p in trainable):,} trainable parameters, "
                  f"global batch {run['global_batch']}, from step {step}")
    log = (out / "metrics.jsonl").open("a") if acc.is_main_process else None
    g_aug = torch.Generator().manual_seed(args.seed + 7 * rank)
    g_eps = torch.Generator(device=device).manual_seed(args.seed + 13 * rank)
    lora_checked, t_win, it = not args.lora, time.time(), iter(loader)
    while step < args.steps:
        groups = next(it)
        with acc.accumulate(model):
            loss = flow_matching_loss(model, sar_encoder, encode_eo, groups, g_aug, g_eps, args.cond_dropout, device)
            acc.backward(loss)
            if not lora_checked:
                acc.print(f"[lora] {check_lora_grads(net)} adapter tensors receive gradients, the base none")
                lora_checked = True
            if acc.sync_gradients:
                grad_norm = acc.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if not acc.sync_gradients:
            continue
        step += 1
        ema.update(step)
        if acc.is_main_process and step % args.log_every == 0:
            now = time.time()
            line = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                    "lr": sched.get_last_lr()[0],
                    "img_per_s": args.log_every * args.micro * args.accum * acc.num_processes / (now - t_win)}
            t_win = now
            log.write(json.dumps(line) + "\n")
            log.flush()
            print(f"[{step}] loss {line['loss']:.4f} grad_norm {line['grad_norm']:.3f} lr {line['lr']:.2e} "
                  f"{line['img_per_s']:.1f} img/s", flush=True)
        if step % args.state_every == 0 or step == args.steps:
            acc.save_state(str(state_dir))
            if acc.is_main_process:
                (state_dir / "progress.json").write_text(json.dumps({"step": step}) + "\n")
        if step % args.ckpt_every == 0 or step == args.steps:
            acc.wait_for_everyone()
            if acc.is_main_process:
                save_checkpoint(out / "checkpoints" / f"step_{step:07d}", net, ema, args)
    if log:
        log.close()


if __name__ == "__main__":
    main()
