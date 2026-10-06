# ResShift

The ResShift rows of Tables 2 and 3 (Yue, Wang and Loy, "ResShift: Efficient diffusion model for image
super-resolution by residual shifting", NeurIPS 2023), retrained on each dataset as a same-size (`sf = 1`)
SAR-to-EO mapping in the latent space of the frozen ResShift VQ-f4 autoencoder.

| | |
|---|---|
| Upstream | [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift) |
| Pinned commit | `bb03b7d21614cace01787e097c8a6ab6b945227d` |
| Licence | [S-Lab License 1.0](https://github.com/zsyOAOA/ResShift/blob/bb03b7d21614cace01787e097c8a6ab6b945227d/LICENSE) (non-commercial use) |

## Environment

```bash
export BASELINES_ROOT=/path/to/upstream_code   # fetch.sh clones into $BASELINES_ROOT/resshift
export DATA_ROOT=/path/to/data
export WORK_DIR=/path/to/work
export CKPT_ROOT=/path/to/checkpoints          # $CKPT_ROOT/resshift/<dataset>/ema_model.pth
# optional: GPU (default 0), PY (default python)
bash baselines/resshift/fetch.sh
```

Install the upstream requirements (`requirements.txt` at the pinned commit) for your PyTorch;
`configs/<dataset>.yaml` are expanded with `envsubst` (GNU gettext). `fetch.sh` also
downloads the frozen VQ-f4 autoencoder of the ResShift v2.0 release
(`autoencoder_vq_f4.pth`, 221,364,711 bytes, sha256
`e967eaa2c380152811c53e312990d6967a8a957c4f7590f19490ec72110b2889`) to `$BASELINES_ROOT/resshift/weights/`.
LPIPS (AlexNet) downloads the torchvision AlexNet weights on first use. `resshift.patch`:

- `ldm/modules/attention.py`, `ldm/modules/diffusionmodules/model.py`, `models/unet.py`: xformers is not used
  (PyTorch attention);
- `trainer.py`: NaN/Inf are replaced and values clamped before images are written to the training log;
- adds `inference_sar2eo.py`: paired inference over a folder with `ResShiftSampler` (`sf = 1`, AMP, 256-px tiles
  without overlap, seed 12345), one output per input named by its stem.

## Data

```bash
python baselines/data/diff_data.py paired <dataset>
```

writes `$WORK_DIR/diff_data/<dataset>/paired/{train,test}{A,B}` (A = SAR, B = EO; SAR2Opt test images are the
512 × 512 centre crops) and, for SAR2EO, `test_first4000A` (the evaluated first 4,000 test SAR images).

## Train

```bash
GPU=0 bash baselines/resshift/train.sh <dataset>   # -> $CKPT_ROOT/resshift/<dataset>/ema_model.pth
```

All four datasets use `configs/<dataset>.yaml`: `UNetModelSwin` (160 base channels, multipliers [1, 2, 2, 4],
Swin depth 2, embedding 192, window 8, `cond_lq`, `lq_size` 256); 4 diffusion steps (exponential schedule,
power 0.3, `etas_end` 0.99, `min_noise_level` 0.2, kappa 2.0, `predict_type` xstart); 50,000 iterations, batch 16
(two micro-batches of 8), lr 5e-5 with 2,000 warm-up steps and cosine decay to 2e-5, EMA 0.999, AMP, seed 123456;
loss = latent MSE + 4 × LPIPS (AlexNet); random 256 × 256 crops with flips and rotations (SAR2Opt: crops of the
600 × 600 tiles). `ema_model.pth` is `ema_ckpts/ema_model_50000.pth`.

## Test

```bash
GPU=0 bash baselines/resshift/test.sh <dataset>    # -> $WORK_DIR/results/resshift/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/resshift/<dataset>
```

SAR2Opt 512 × 512 test images are processed as four non-overlapping 256 × 256 tiles.

## Released weights

[`JeonghyeokDo/GeoSET`](https://huggingface.co/JeonghyeokDo/GeoSET), `baselines/<dataset>/resshift/ema_model.pth`
(one per dataset). Place it at `$CKPT_ROOT/resshift/<dataset>/ema_model.pth`:

```bash
hf download JeonghyeokDo/GeoSET --include "baselines/<dataset>/resshift/*" --local-dir hf
mkdir -p $CKPT_ROOT/resshift/<dataset> && cp hf/baselines/<dataset>/resshift/ema_model.pth $CKPT_ROOT/resshift/<dataset>/
```
