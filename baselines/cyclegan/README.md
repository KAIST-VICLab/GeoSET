# CycleGAN

Zhu, Park, Isola and Efros, "Unpaired image-to-image translation using cycle-consistent adversarial networks",
ICCV 2017. A = SAR, B = EO; the output is G_A(SAR).

| | |
|---|---|
| Upstream | https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix |
| Pinned commit | `2a7afba2895d52556dd5dfe07e8555ef657ced6f` |
| Licence | BSD, Copyright (c) 2017 Jun-Yan Zhu and Taesung Park (upstream `LICENSE`) |

## Setup

```bash
export BASELINES_ROOT=/path/to/upstream_code DATA_ROOT=/path/to/data WORK_DIR=/path/to/work CKPT_ROOT=/path/to/checkpoints
bash baselines/cyclegan/fetch.sh    # clones into $BASELINES_ROOT/cyclegan and applies cyclegan.patch
bash baselines/pix2pix/fetch.sh     # gan_prepare.py uses its combine_A_and_B.py
```

Optional: `GPU` (default 0), `PY` (default `python`). Python packages: `torch`, `torchvision`, `numpy`, `pillow`,
`dominate`, `wandb` (imported by `util/visualizer.py`; no account is needed) and `opencv-python`.

`cyclegan.patch` (the same repository and patch as `baselines/pix2pix`) changes:
- `datasets/combine_A_and_B.py`: imports `pathlib.Path`, which the script uses.

## Data

```bash
python baselines/data/gan_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR BASELINES_ROOT
```

This builds `$WORK_DIR/gan_data/<dataset>/` from the split lists in `splits/` (A = SAR, B = EO); see
`baselines/pix2pix/README.md`. CycleGAN trains on `AB/{trainA,trainB}` and is tested on `AB/testA`
(`sar2opt`: `test512/testA`, centre 512 × 512 crops). `spacenet6` needs the chips of
`scripts/datasets/build_spacenet6.py`.

## Train

```bash
bash baselines/cyclegan/train.sh <dataset>   # run in $WORK_DIR/cyclegan/train/<dataset>, checkpoints -> $CKPT_ROOT/cyclegan/<dataset>/net_G_{A,B}.pth
```

| Dataset | Load → crop | Batch | Epochs (constant + linear decay) |
|---|---|---:|---|
| `qxs-saropt` | 256 → 256 | 8 | 100 + 100 |
| `sar2opt` | 600 → random 512 | 4 | 222 + 221 |
| `sar2eo` | 256 → 256 | 8 | 6 + 6 |
| `spacenet6` | 256 → 256 | 8 | 250 + 250 |

Upstream defaults otherwise: `resnet_9blocks` generators (instance norm, no dropout), 3-layer PatchGAN
discriminators, LSGAN loss, λ_A = λ_B = 10, identity loss 0.5, image pool 50, Adam (lr 2e-4, β1 = 0.5), random
flips. `net_G_A.pth` / `net_G_B.pth` are the run's `latest_net_G_A.pth` / `latest_net_G_B.pth`.

## Test

```bash
bash baselines/cyclegan/test.sh <dataset>    # -> $WORK_DIR/results/cyclegan/<dataset>/<stem>_fake_B.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/cyclegan/<dataset>
```

`test.sh` runs the upstream `test.py --model cycle_gan` at the evaluation size (256; `sar2opt`: 512) and keeps
`fake_B` = G_A(SAR). The upstream test script loads both generators, so both files are needed.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/cyclegan/net_G_A.pth` and `net_G_B.pth`. Copy them to
`$CKPT_ROOT/cyclegan/<dataset>/` and run `test.sh`.
