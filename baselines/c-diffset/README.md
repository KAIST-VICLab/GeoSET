# C-DiffSET

Do, Lee, Lee and Kim, "C-DiffSET: Leveraging latent diffusion for SAR-to-EO image translation with
confidence-guided reliable object generation", IEEE Transactions on Circuits and Systems for Video Technology,
2026. This is the confidence-guided training stage; it starts from the SD2.1 FT run of
[`../sd21-ft`](../sd21-ft/README.md) on the same dataset.

| | |
|---|---|
| Upstream | https://github.com/KAIST-VICLab/C-DiffSET |
| Pinned commit | `6fc380215ae97c9d134e8f1434061b652339b45c` |
| Licence | MIT, Copyright (c) 2026 KAIST VICLab |
| Base model | `Manojb/stable-diffusion-2-1-base` (Stable Diffusion 2.1-base), revision `0094d483a120f3f33dafbd187ea4aa60d10de75c` |

## Setup

```bash
bash baselines/c-diffset/fetch.sh    # clones into $BASELINES_ROOT/c-diffset; the upstream code is used unchanged
```

## Data

```bash
python baselines/data/ldm_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR
```

The same list files and test inputs as SD2.1 FT; see [`../sd21-ft/README.md`](../sd21-ft/README.md#data).

## Training and testing

Environment: `BASELINES_ROOT`, `DATA_ROOT`, `WORK_DIR`, `CKPT_ROOT`, optional `GPU` (default 0) and `PY`
(default `python`). The configs in `configs/` are expanded with `envsubst` (GNU gettext). `WORK_DIR` must not
contain the string `train`, because the data feeders take the split from the list path.

```bash
bash baselines/sd21-ft/train.sh <dataset>     # stage 1, see ../sd21-ft
bash baselines/c-diffset/train.sh <dataset>   # run in $WORK_DIR/c-diffset/<dataset>, checkpoint-40000 -> $CKPT_ROOT/c-diffset/<dataset>/
bash baselines/c-diffset/test.sh <dataset>    # -> $WORK_DIR/results/c-diffset/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/c-diffset/<dataset>
```

| Dataset | Initialisation (`accelerator_path`) | Training crop | Batch | Updates (`num_iter`) | Checkpoint used |
|---|---|---|---:|---:|---|
| `qxs-saropt` | `$WORK_DIR/sd21-ft/qxs-saropt/lastest` | 256 | 64 | 50,000 | `checkpoint-40000` |
| `sar2opt` | `$WORK_DIR/sd21-ft/sar2opt/lastest` | random 512 of 600 | 16 | 50,000 | `checkpoint-40000` |
| `sar2eo` | `$WORK_DIR/sd21-ft/sar2eo/checkpoint-40000` | 256 | 64 | 40,000 | `checkpoint-40000` |
| `spacenet6` | `$WORK_DIR/sd21-ft/spacenet6/checkpoint-40000` | 256 | 64 | 40,000 | `checkpoint-40000` |

The UNet output gains a zero-initialised fifth channel, a spatial variance whose inverse is the confidence, and
the UNet is trained with the confidence-guided loss. Optimiser, schedule, precision, seed and augmentation are those of SD2.1 FT. The `qxs-saropt` run saves a
checkpoint every 2,000 updates (about 10 GB each with the optimiser state). `test.sh` samples with DDIM, 50 steps,
fp32, one image at a time; the scheduler uses output channels 0-3.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/c-diffset/{diffusion_pytorch_model.safetensors,config.json}` (a diffusers
`UNet2DConditionModel` folder). Copy them to `$CKPT_ROOT/c-diffset/<dataset>/` and run `test.sh`.
