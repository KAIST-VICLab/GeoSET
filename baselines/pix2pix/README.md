# pix2pix

Isola, Zhu, Zhou and Efros, "Image-to-image translation with conditional adversarial networks", CVPR 2017.
Paired translation with a U-Net generator and an L1 + adversarial objective; A = SAR, B = EO, `--direction AtoB`.

| | |
|---|---|
| Upstream | https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix |
| Pinned commit | `2a7afba2895d52556dd5dfe07e8555ef657ced6f` |
| Licence | BSD, Copyright (c) 2017 Jun-Yan Zhu and Taesung Park; pix2pix: Copyright (c) 2016 Phillip Isola and Jun-Yan Zhu (upstream `LICENSE`) |

## Setup

```bash
export BASELINES_ROOT=/path/to/upstream_code DATA_ROOT=/path/to/data WORK_DIR=/path/to/work CKPT_ROOT=/path/to/checkpoints
bash baselines/pix2pix/fetch.sh     # clones into $BASELINES_ROOT/pix2pix and applies pix2pix.patch
```

Optional: `GPU` (default 0), `PY` (default `python`). Python packages: `torch`, `torchvision`, `numpy`, `pillow`,
`dominate`, `wandb` (imported by `util/visualizer.py`; no account is needed) and `opencv-python`.

`pix2pix.patch` changes:
- `datasets/combine_A_and_B.py`: imports `pathlib.Path`, which the script uses; it writes the side-by-side A|B
  images that pix2pix trains on.

## Data

```bash
python baselines/data/gan_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR BASELINES_ROOT
```

This builds `$WORK_DIR/gan_data/<dataset>/` from the split lists in `splits/` (A = SAR, B = EO):
`AB/{trainA,trainB,testA,testB}` (links), `combined/{train,test}` (A|B images written by the upstream
`combine_A_and_B.py`), `p2phd/` (pix2pixHD layout) and, for `sar2opt`, `test512/` and `combined_test512/`
(centre 512 × 512 crops of the 600 × 600 test tiles). The `sar2eo` test set is the first 4,000 pairs
(`splits/sar2eo/test_first4000.txt`); `spacenet6` needs the chips of `scripts/datasets/build_spacenet6.py`.
pix2pix trains on `combined/train` and is tested on `combined/test` (`sar2opt`: `combined_test512/test`).

## Train

```bash
bash baselines/pix2pix/train.sh <dataset>   # run in $WORK_DIR/pix2pix/train/<dataset>, checkpoint -> $CKPT_ROOT/pix2pix/<dataset>/net_G.pth
```

| Dataset | Load → crop | Batch | Epochs (constant + linear decay) |
|---|---|---:|---|
| `qxs-saropt` | 256 → 256 | 16 | 60 + 60 |
| `sar2opt` | 600 → random 512 | 8 | 100 + 100 |
| `sar2eo` | 256 → 256 | 16 | 14 + 14 |
| `spacenet6` | 256 → 256 | 16 | 250 + 250 |

Upstream defaults otherwise: `unet_256` generator (batch norm, dropout), 3-layer PatchGAN discriminator, vanilla
GAN loss, λ_L1 = 100, Adam (lr 2e-4, β1 = 0.5), random flips. `net_G.pth` is the run's `latest_net_G.pth`.

## Test

```bash
bash baselines/pix2pix/test.sh <dataset>    # -> $WORK_DIR/results/pix2pix/<dataset>/<stem>_fake_B.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/pix2pix/<dataset>
```

`test.sh` exposes `$CKPT_ROOT/pix2pix/<dataset>/net_G.pth` as the upstream `latest_net_G.pth` and runs `test.py`
at the evaluation size (256; `sar2opt`: 512 on the centre crops). Like the upstream default, `test.py` runs
without `--eval`, so dropout and batch statistics are active at test time and repeated runs differ slightly.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/pix2pix/net_G.pth`. Copy it to `$CKPT_ROOT/pix2pix/<dataset>/net_G.pth`
and run `test.sh`.
