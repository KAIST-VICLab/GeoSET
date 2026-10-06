# E3Diff

The E3Diff rows of Tables 2 and 3 (Qin et al., "Efficient end-to-end diffusion model for one-step SAR-to-optical
translation", GRSL 2024), retrained on each dataset: stage 2 of E3Diff turns the stage-1 conditional DDPM
([`../ddpm-sr3`](../ddpm-sr3)) into a one-step generator.

| | |
|---|---|
| Upstream | [DeepSARRS/E3Diff](https://github.com/DeepSARRS/E3Diff) |
| Pinned commit | `38601093ab8f8e4b478144621f20890b100a3b74` |
| Licence | The upstream repository contains no licence file (its bundled `SoftPool/` directory is MIT). Only our changes are distributed here; `fetch.sh` downloads the upstream code. |

## Environment

```bash
export BASELINES_ROOT=/path/to/upstream_code   # fetch.sh clones into $BASELINES_ROOT/e3diff
export DATA_ROOT=/path/to/data
export WORK_DIR=/path/to/work
export CKPT_ROOT=/path/to/checkpoints          # $CKPT_ROOT/{ddpm-sr3,e3diff}/<dataset>/gen.pth
# optional: GPU (default 0), PY (default python)
bash baselines/e3diff/fetch.sh
pip install --no-deps vision-aided-loss==0.1.0 antialiased-cnns==0.3 ftfy==6.3.1 regex==2026.7.19 gdown==6.1.0 \
    beautifulsoup4==4.15.0 soupsieve==2.9.2 "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
```

Stage 2 trains against the CLIP discriminator of `vision-aided-loss` (its ViT-B/32 weights are downloaded on
first use), and the model builds it for testing as well. The other requirements are those of
[`../ddpm-sr3`](../ddpm-sr3). `e3diff.patch` is identical to `ddpm-sr3.patch` (`core/logger.py` keeps an already
set `CUDA_VISIBLE_DEVICES`). The scripts reuse `../ddpm-sr3/e3diff_rgb_main.py` and `../ddpm-sr3/shims/SoftPool.py`
(but not the stage-1 discriminator stub). `configs/<dataset>.json` are the stage-2 training configurations (expanded with `envsubst`, GNU gettext).

## Data

```bash
python baselines/data/diff_data.py e3diff <dataset>
```

The same `$WORK_DIR/diff_data/<dataset>/e3diff` tree as DDPM (SR3).

## Train

```bash
GPU=0 bash baselines/e3diff/train.sh <dataset>     # -> $CKPT_ROOT/e3diff/<dataset>/gen.pth
```

Stage 2 loads the stage-1 weights `$CKPT_ROOT/ddpm-sr3/<dataset>/gen.pth` (trained with
`baselines/ddpm-sr3/train.sh` or downloaded), resumes the iteration counter at 250,000 and trains to iteration
310,000, i.e. 60,000 updates (E3Diff reads only the iteration and epoch counters from the stage-1 `*_opt.pth`,
which `train.sh` writes). Losses: L1 + 5 × LPIPS-VGG + 10 × focal frequency loss + 0.5 × CLIP-discriminator
GAN loss on the one-step DDIM output; Adam, lr 5e-5 for the generator and the discriminator; `--seed 1`.

| dataset | resolution | batch | stage-2 updates | entry point |
|---|---|---|---|---|
| qxs-saropt | 256 | 16 | 60,000 | `../ddpm-sr3/e3diff_rgb_main.py` |
| sar2opt | 512 | 4 | 60,000 | `../ddpm-sr3/e3diff_rgb_main.py` |
| sar2eo | 256 | 16 | 60,000 | `main.py` |
| spacenet6 | 256 | 16 | 60,000 | `../ddpm-sr3/e3diff_rgb_main.py` |

## Test

```bash
GPU=0 bash baselines/e3diff/test.sh <dataset>      # -> $WORK_DIR/results/e3diff/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/e3diff/<dataset>
```

Sampling: one DDIM step, `--seed 1`.

## Released weights

[`JeonghyeokDo/GeoSET`](https://huggingface.co/JeonghyeokDo/GeoSET), `baselines/<dataset>/e3diff/gen.pth`
(iteration-310,000 UNet state dict, one per dataset). Place it at `$CKPT_ROOT/e3diff/<dataset>/gen.pth`:

```bash
hf download JeonghyeokDo/GeoSET --include "baselines/<dataset>/e3diff/*" --local-dir hf
mkdir -p $CKPT_ROOT/e3diff/<dataset> && cp hf/baselines/<dataset>/e3diff/gen.pth $CKPT_ROOT/e3diff/<dataset>/
```
