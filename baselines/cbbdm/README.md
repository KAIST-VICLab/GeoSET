# cBBDM

Kim and Chung, "Conditional Brownian bridge diffusion model for VHR SAR to optical image translation",
IEEE Geoscience and Remote Sensing Letters, 2025. We train the latent model cBBDM-f4 with the SAR image as the
condition (A) and the EO image as the target (B).

| | |
|---|---|
| Upstream | https://github.com/egshkim/ConditionalBBDM-for-VHR-SAR-to-Optical |
| Pinned commit | `8ce15934f4d4e3f01efe70d11e2d9b9e0859210c` |
| Licence | MIT, Copyright (c) 2025 egshkim |
| Autoencoder | CompVis latent-diffusion vq-f4 (MIT) |

## Setup

```bash
bash baselines/cbbdm/fetch.sh       # clones into $BASELINES_ROOT/cbbdm and applies cbbdm.patch
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
unzip vq-f4.zip -d $CKPT_ROOT/vq-f4  # $CKPT_ROOT/vq-f4/model.ckpt, needed for training and testing
```

`model.ckpt` sha256: `aacf13951f4b18f5af9b47febdc696cf9559305d6de0821084abeaf342439251`.

`cbbdm.patch` changes:
- `model/VQGAN/vqgan.py`, `runners/BaseRunner.py`, `runners/DiffusionBasedModelRunners/BBDMRunner.py`: `torch.load(..., weights_only=False)`, needed with PyTorch >= 2.6.
- `runners/DiffusionBasedModelRunners/BBDMRunner.py`: no `verbose` argument for `ReduceLROnPlateau` (removed from PyTorch).
- `runners/BaseRunner.py`: `num_workers=8` instead of 0 in the train, validation and test data loaders.

## Data

```bash
python baselines/data/ldm_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR
```

cBBDM reads the same tree as BBDM, `$WORK_DIR/ldm_data/<dataset>/bbdm/{train,val,test}/{A,B}` (A = SAR, B = EO);
see [`../bbdm/README.md`](../bbdm/README.md#data).

## Training and testing

Environment: `BASELINES_ROOT`, `WORK_DIR`, `CKPT_ROOT`, optional `GPU` (default 0) and `PY` (default `python`).
The configs in `configs/` are expanded with `envsubst` (GNU gettext).

```bash
bash baselines/cbbdm/train.sh <dataset>   # run in $WORK_DIR/cbbdm/<dataset>/cBBDM-f4, checkpoint -> $CKPT_ROOT/cbbdm/<dataset>/last_model.pth
bash baselines/cbbdm/test.sh <dataset>    # -> $WORK_DIR/results/cbbdm/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/cbbdm/<dataset>
```

| Dataset | Image / latent | Batch | Flipped copies | Training length | Test batch |
|---|---|---:|---|---|---:|
| `qxs-saropt` | 256 / 64 | 32 | no | 100 epochs, 50,000 updates | 31 |
| `sar2opt` | 512 / 128 | 8 | yes | 139 epochs, 50,318 updates | 11 |
| `sar2eo` | 256 / 64 | 32 | no | 24 epochs, 51,096 updates | 20 |
| `spacenet6` | 256 / 64 | 32 | yes | 315 epochs, 50,085 updates | 45 |

The SAR latent conditions the UNet through the `SpatialRescaler` (concatenation, 6 input channels). Training
stops at the end of the first epoch past `n_steps: 50000` or after `n_epochs`. All runs use Adam (lr 1e-4) with
`ReduceLROnPlateau`, EMA (0.995, from step 30,000), seed 1234 and fp32. `test.sh` samples with the EMA weights,
200 Brownian-bridge steps and seed 1234.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/cbbdm/last_model.pth`. Copy it to
`$CKPT_ROOT/cbbdm/<dataset>/last_model.pth` and run `test.sh`.
