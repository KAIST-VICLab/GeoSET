# SD2.1 FT

Fine-tuning of Stable Diffusion 2.1-base (Rombach et al., "High-resolution image synthesis with latent diffusion
models", CVPR 2022) for SAR-to-EO translation. The UNet receives the SAR latent concatenated with the noisy EO
latent (8 input channels; the extra input kernel is initialised by duplicating the pretrained one and halving
both copies), predicts the noise with an MSE loss and is conditioned on the fixed prompt `"electro-optical image"`.

| | |
|---|---|
| Code base | https://github.com/KAIST-VICLab/C-DiffSET |
| Pinned commit | `6fc380215ae97c9d134e8f1434061b652339b45c` |
| Licence | MIT, Copyright (c) 2026 KAIST VICLab |
| Base model | `Manojb/stable-diffusion-2-1-base` (Stable Diffusion 2.1-base), revision `0094d483a120f3f33dafbd187ea4aa60d10de75c` |

## Setup

```bash
bash baselines/sd21-ft/fetch.sh                                             # clones into $BASELINES_ROOT/sd21-ft and applies sd21-ft.patch
python baselines/sd21-ft/make_init.py $CKPT_ROOT/sd21_unet8ch_init.safetensors   # 8-channel initialisation (training only)
```

`sd21-ft.patch` adds three files and changes nothing else:
- `main_stage1.py`: the entry point of `main.py`, using `Stage1Trainer`.
- `train_stage1.py`: `Stage1Trainer`, the C-DiffSET trainer with a 4-channel output and the MSE noise loss.
- `test_stage1.py`: inference with a 4-channel output and strict checkpoint loading.

`make_init.py` writes the Stable Diffusion 2.1-base UNet with the 8-channel `conv_in`
(sha256 `e11500d06a2304d0a498b24fe93d90617fbee0247398f13b4ae7b251a48dcfed`).

## Data

```bash
python baselines/data/ldm_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR
```

Training reads the list files `$WORK_DIR/ldm_data/<dataset>/cdiffset/{train,test}.txt` (QXS-SAROPT and SAR2Opt:
the split lists, read from `$DATA_ROOT/<dataset>`; SAR2EO and SpaceNet6: `trainB|testB/<name>.png` in the paired
view `$WORK_DIR/ldm_data/<dataset>/AB`). Testing translates `$WORK_DIR/ldm_data/<dataset>/test_sar`
(SAR2Opt: centre 512 × 512 crops; SAR2EO: the first 4,000 test pairs).

## Training and testing

Environment: `BASELINES_ROOT`, `DATA_ROOT`, `WORK_DIR`, `CKPT_ROOT`, optional `GPU` (default 0) and `PY`
(default `python`). The configs in `configs/` are expanded with `envsubst` (GNU gettext). `WORK_DIR` must not
contain the string `train`, because the data feeders take the split from the list path.

```bash
bash baselines/sd21-ft/train.sh <dataset>   # run in $WORK_DIR/sd21-ft/<dataset>, checkpoint-40000 -> $CKPT_ROOT/sd21-ft/<dataset>/
bash baselines/sd21-ft/test.sh <dataset>    # -> $WORK_DIR/results/sd21-ft/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/sd21-ft/<dataset>
```

| Dataset | Training crop | Batch | Updates (`num_iter`) | Checkpoint used |
|---|---|---:|---:|---|
| `qxs-saropt` | 256 | 64 | 50,000 | `checkpoint-40000` |
| `sar2opt` | random 512 of 600 | 16 | 50,000 | `checkpoint-40000` |
| `sar2eo` | 256 | 64 | 40,000 | `checkpoint-40000` |
| `spacenet6` | 256 | 64 | 40,000 | `checkpoint-40000` |

All runs use AdamW (lr 3e-5, weight decay 0.01) with a cosine schedule and 100 warm-up steps, fp32, seed 2024,
and random horizontal / vertical flips and 90° rotations. `test.sh` samples with DDIM, 50 steps, fp32, one image
at a time. `baselines/c-diffset/train.sh` starts from this run.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/sd21-ft/{diffusion_pytorch_model.safetensors,config.json}` (a diffusers
`UNet2DConditionModel` folder). Copy them to `$CKPT_ROOT/sd21-ft/<dataset>/` and run `test.sh`.
