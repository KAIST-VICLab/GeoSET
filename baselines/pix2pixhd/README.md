# pix2pixHD

Wang, Liu, Zhu, Tao, Kautz and Catanzaro, "High-resolution image synthesis and semantic manipulation with
conditional GANs", CVPR 2018. Image-to-image mode (`--label_nc 0 --no_instance`): the SAR image is the input map
A and the EO image the target B.

| | |
|---|---|
| Upstream | https://github.com/NVIDIA/pix2pixHD |
| Pinned commit | `14b3b3c7fff413086e3b58df52096f16b6891172` |
| Licence | BSD, Copyright (C) 2019 NVIDIA Corporation (upstream `LICENSE.txt`, which also carries the pytorch-CycleGAN-and-pix2pix BSD notice) |

## Setup

```bash
export BASELINES_ROOT=/path/to/upstream_code DATA_ROOT=/path/to/data WORK_DIR=/path/to/work CKPT_ROOT=/path/to/checkpoints
bash baselines/pix2pixhd/fetch.sh   # clones into $BASELINES_ROOT/pix2pixhd and applies pix2pixhd.patch
bash baselines/pix2pix/fetch.sh     # gan_prepare.py uses its combine_A_and_B.py
```

Optional: `GPU` (default 0), `PY` (default `python`). Python packages: `torch`, `torchvision`, `numpy`, `pillow`,
`dominate` (and `opencv-python` for `gan_prepare.py`).

`pix2pixhd.patch` changes:
- `data/base_dataset.py`: `transforms.Scale` → `transforms.Resize` (removed from torchvision).
- `train.py`: `fractions.gcd` → `math.gcd` with integer division (removed in Python 3.9).
- `util/visualizer.py`: the `scipy.misc` import is optional (removed in SciPy 1.12; only used with `--tf_log`);
  test images are written as PNG instead of JPEG.

## Data

```bash
python baselines/data/gan_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR BASELINES_ROOT
```

This builds `$WORK_DIR/gan_data/<dataset>/` from the split lists in `splits/`; see `baselines/pix2pix/README.md`.
pix2pixHD reads `p2phd/{train_A,train_B,test_A,test_B}` (links to `AB/`; `sar2opt` tests on `test512/`, the
centre 512 × 512 crops). `spacenet6` needs the chips of `scripts/datasets/build_spacenet6.py`.

## Train

```bash
bash baselines/pix2pixhd/train.sh <dataset>   # run in $WORK_DIR/pix2pixhd/train/<dataset>, checkpoint -> $CKPT_ROOT/pix2pixhd/<dataset>/net_G.pth
```

| Dataset | `--resize_or_crop`, load → crop | Batch | Epochs (`niter` + `niter_decay`) |
|---|---|---:|---|
| `qxs-saropt` | `resize_and_crop`, 256 → 256 | 16 | 60 + 60 |
| `sar2opt` | `crop`, 600 → random 512 | 8 | 100 + 100 |
| `sar2eo` | `resize_and_crop`, 256 → 256 | 16 | 14 + 14 |
| `spacenet6` | `resize_and_crop`, 256 → 256 | 16 | 375 + 375 |

Upstream defaults otherwise: global generator (64 filters, 4 downsampling layers, 9 residual blocks, instance
norm), two multi-scale 3-layer discriminators, LSGAN loss with feature matching (λ = 10) and VGG loss, Adam
(lr 2e-4, β1 = 0.5). `net_G.pth` is the run's `latest_net_G.pth`.

## Test

```bash
bash baselines/pix2pixhd/test.sh <dataset>    # -> $WORK_DIR/results/pix2pixhd/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/pix2pixhd/<dataset>
```

`test.sh` runs the upstream `test.py` at the evaluation size (256; `sar2opt`: `--resize_or_crop crop` at 512 on
the centre crops) with `--how_many` covering the whole test set (the upstream default is 50) and keeps
`<stem>_synthesized_image.png`.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/pix2pixhd/net_G.pth`. Copy it to
`$CKPT_ROOT/pix2pixhd/<dataset>/net_G.pth` and run `test.sh`.
