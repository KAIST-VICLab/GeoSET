# DDPM (SR3)

The DDPM (SR3) rows of Tables 2 and 3 (Saharia et al., "Image super-resolution via iterative refinement",
TPAMI 2022): a conditional pixel-space DDPM, run with the stage-1 configuration of the E3Diff code.
Stage 2 of the same code is the E3Diff row ([`../e3diff`](../e3diff)); it starts from these weights.

| | |
|---|---|
| Upstream | [DeepSARRS/E3Diff](https://github.com/DeepSARRS/E3Diff) |
| Pinned commit | `38601093ab8f8e4b478144621f20890b100a3b74` |
| Licence | The upstream repository contains no licence file (its bundled `SoftPool/` directory is MIT). Only our changes are distributed here; `fetch.sh` downloads the upstream code. |

## Environment

```bash
export BASELINES_ROOT=/path/to/upstream_code   # fetch.sh clones into $BASELINES_ROOT/ddpm-sr3
export DATA_ROOT=/path/to/data                 # datasets in the layout of splits/README.md
export WORK_DIR=/path/to/work                  # derived data, run directories, outputs
export CKPT_ROOT=/path/to/checkpoints          # $CKPT_ROOT/ddpm-sr3/<dataset>/gen.pth
# optional: GPU (default 0), PY (default python)
bash baselines/ddpm-sr3/fetch.sh
```

Python packages used by the upstream code: `torch`, `torchvision`, `opencv-python`, `pillow`, `lmdb`, `lpips`,
`tensorboardX`. The code builds LPIPS-VGG at start-up, which downloads the torchvision VGG-16 weights.

`ddpm-sr3.patch` changes one upstream file:

- `core/logger.py`: an already set `CUDA_VISIBLE_DEVICES` is kept instead of being overwritten from `gpu_ids`.

Files in this directory (used in place, not copied into the clone):

- `e3diff_rgb_main.py`: runs the upstream `main.py` with a 3-channel EO target and a 3-channel condition
  [SAR-PPB, SAR-canny, SAR] (QXS-SAROPT, SAR2Opt, SpaceNet6). SAR2EO has grayscale EO and uses the upstream
  `main.py` directly (1-channel target, [SAR-PPB, SAR-canny] condition).
- `shims/SoftPool.py`: pure-PyTorch `SoftPool2d`, in place of the SoftPool CUDA extension.
- `stub/vision_aided_loss.py`: constant stand-in for the CLIP discriminator, which the code builds but stage 1
  does not use.
- `configs/<dataset>.json`: training configurations (`${WORK_DIR}` paths, expanded with `envsubst` from GNU
  gettext).

## Data

```bash
python baselines/data/diff_data.py e3diff <dataset>
```

writes `$WORK_DIR/diff_data/<dataset>/e3diff/{train,val}/{SAR,EO,SAR-PPB,SAR-canny}`, where `val` is the test
split (SAR2EO: its first 4,000 pairs) and SAR2Opt uses 512 × 512 centre crops for training and testing.
SAR-PPB is the non-iterative FAST-PPB filter (Deledalle et al., 2009; P = 3, W = 10, h = 0.5) of the SAR image
rescaled to 8 bit, and SAR-canny is `cv2.Canny(ppb, 50, 150, L2gradient=True)`; both run on the GPU when one is
available. SpaceNet6 SAR chips have three channels: the filters see their grayscale conversion and the raw-SAR
condition channel is channel 0.

## Train

```bash
GPU=0 bash baselines/ddpm-sr3/train.sh <dataset>   # -> $CKPT_ROOT/ddpm-sr3/<dataset>/gen.pth
```

| dataset | resolution | batch | iterations | entry point |
|---|---|---|---|---|
| qxs-saropt | 256 | 16 | 250,000 | `e3diff_rgb_main.py` |
| sar2opt | 512 | 4 | 250,000 | `e3diff_rgb_main.py` |
| sar2eo | 256 | 16 | 250,000 | `main.py` |
| spacenet6 | 256 | 16 | 250,000 | `e3diff_rgb_main.py` |

Adam (lr 5e-5, constant), linear noise schedule over 1,000 steps (1e-6 to 1e-2), noise-prediction L1 loss,
`--seed 1`. Checkpoints are written every 25,000 iterations (SAR2EO: 10,000); `gen.pth` is the iteration-250,000
UNet state dict.

## Test

```bash
GPU=0 bash baselines/ddpm-sr3/test.sh <dataset>    # -> $WORK_DIR/results/ddpm-sr3/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/ddpm-sr3/<dataset>
```

Sampling: DDIM with 50 steps, `--seed 1`.

## Released weights

[`JeonghyeokDo/GeoSET`](https://huggingface.co/JeonghyeokDo/GeoSET), `baselines/<dataset>/ddpm-sr3/gen.pth`
(one per dataset). Place it at `$CKPT_ROOT/ddpm-sr3/<dataset>/gen.pth`:

```bash
hf download JeonghyeokDo/GeoSET --include "baselines/<dataset>/ddpm-sr3/*" --local-dir hf
mkdir -p $CKPT_ROOT/ddpm-sr3/<dataset> && cp hf/baselines/<dataset>/ddpm-sr3/gen.pth $CKPT_ROOT/ddpm-sr3/<dataset>/
```
