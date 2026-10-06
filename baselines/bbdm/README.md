# BBDM

Li, Xue, Liu and Lai, "BBDM: Image-to-image translation with Brownian bridge diffusion models", CVPR 2023.
We train the latent model LBBDM-f4 with the SAR image as the condition (A) and the EO image as the target (B).

| | |
|---|---|
| Upstream | https://github.com/xuekt98/BBDM |
| Pinned commit | `02c3b13c9f9dfab0853e32123100680a0640c4ed` |
| Licence | MIT, Copyright (c) 2023 xuekt98 |
| Autoencoder | CompVis latent-diffusion vq-f4 (MIT) |

## Setup

```bash
bash baselines/bbdm/fetch.sh        # clones into $BASELINES_ROOT/bbdm and applies bbdm.patch
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
unzip vq-f4.zip -d $CKPT_ROOT/vq-f4  # $CKPT_ROOT/vq-f4/model.ckpt, needed for training and testing
```

`model.ckpt` sha256: `aacf13951f4b18f5af9b47febdc696cf9559305d6de0821084abeaf342439251`.

`bbdm.patch` changes:
- `model/VQGAN/vqgan.py`, `runners/BaseRunner.py`, `runners/DiffusionBasedModelRunners/BBDMRunner.py`: `torch.load(..., weights_only=False)`, needed with PyTorch >= 2.6.
- `runners/DiffusionBasedModelRunners/BBDMRunner.py`: no `verbose` argument for `ReduceLROnPlateau` (removed from PyTorch).
- `datasets/__init__.py` (new, empty): the repository's `datasets` package takes precedence over the Hugging Face `datasets` package.

## Data

```bash
python baselines/data/ldm_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR
```

This links the split lists into `$WORK_DIR/ldm_data/<dataset>/bbdm/{train,val,test}/{A,B}` (A = SAR, B = EO;
SAR2Opt uses centre 512 × 512 PNG crops, SpaceNet6 the chips of `scripts/datasets/build_spacenet6.py`).
`test` of `sar2eo` holds the first 4,000 test pairs. `val` is only used for the losses and sample grids logged
during training: the first 512 (`qxs-saropt`) or 64 (`sar2opt`) test pairs, the whole test split (`sar2eo`),
the first 64 training pairs (`spacenet6`).

## Training and testing

Environment: `BASELINES_ROOT`, `WORK_DIR`, `CKPT_ROOT`, optional `GPU` (default 0) and `PY` (default `python`).
The configs in `configs/` are expanded with `envsubst` (GNU gettext).

```bash
bash baselines/bbdm/train.sh <dataset>   # run in $WORK_DIR/bbdm/<dataset>/LBBDM-f4, checkpoint -> $CKPT_ROOT/bbdm/<dataset>/last_model.pth
bash baselines/bbdm/test.sh <dataset>    # -> $WORK_DIR/results/bbdm/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/bbdm/<dataset>
```

| Dataset | Image / latent | Batch | Flipped copies | Training length | Test batch |
|---|---|---:|---|---|---:|
| `qxs-saropt` | 256 / 64 | 32 | no | 100 epochs, 50,000 updates | 31 |
| `sar2opt` | 512 / 128 | 8 | yes | 139 epochs, 50,318 updates | 11 |
| `sar2eo` | 256 / 64 | 32 | no | 24 epochs, 51,096 updates | 20 |
| `spacenet6` | 256 / 64 | 32 | yes | 315 epochs, 50,085 updates | 45 |

Training stops at the end of the first epoch past `n_steps: 50000` or after `n_epochs`. All runs use Adam
(lr 1e-4) with `ReduceLROnPlateau`, EMA (0.995, from step 30,000), seed 1234 and fp32. `test.sh` samples with the
EMA weights, 200 Brownian-bridge steps and seed 1234.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/bbdm/last_model.pth`. Copy it to
`$CKPT_ROOT/bbdm/<dataset>/last_model.pth` and run `test.sh`.
