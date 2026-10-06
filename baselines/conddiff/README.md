# CondDiff

The CondDiff rows of Tables 2 and 3 (Bai, Pu and Xu, "Conditional diffusion for SAR to optical image
translation", GRSL 2023), retrained on each dataset with the authors' code: a pixel-space conditional DDPM whose
UNet receives the SAR image concatenated with the noisy EO image.

| | |
|---|---|
| Upstream | [Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation](https://github.com/Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation) |
| Pinned commit | `8326f54adf3a8d717792cb7abc7d3d2b7dfac565` |
| Licence | The upstream repository contains no licence file; it is derived from [openai/guided-diffusion](https://github.com/openai/guided-diffusion) (MIT). Only our changes are distributed here; `fetch.sh` downloads the upstream code. |

## Environment

```bash
export BASELINES_ROOT=/path/to/upstream_code   # fetch.sh clones into $BASELINES_ROOT/conddiff
export DATA_ROOT=/path/to/data
export WORK_DIR=/path/to/work
export CKPT_ROOT=/path/to/checkpoints          # $CKPT_ROOT/conddiff/<dataset>/ema_final.pt
# optional: GPU (default 0), PY (default python)
bash baselines/conddiff/fetch.sh
```

Python packages: `torch`, `torchvision`, `numpy`, `pillow`, `tqdm`. `fetch.sh` converts the upstream
`scripts/image_sample_realtime.py` from CRLF to LF line endings before applying `conddiff.patch`, which

- adds `_stubs/blobfile.py` and `_stubs/mpi4py/` (local-file `BlobFile` and a single-process `COMM_WORLD`), put
  on `PYTHONPATH` by the scripts so that neither `blobfile` nor MPI is needed;
- `guided_diffusion/dist_util.py`: keeps an already set `CUDA_VISIBLE_DEVICES`;
- `scripts/image_sample_realtime.py`: reads the test pairs from `--test_dir`, pairs SAR and EO images by sorted
  file name, resizes to `--image_size`, writes to `--out_dir/{gen_opt,cond_sar,gt_opt}`, and sizes the sampling
  noise by the actual (possibly smaller) last batch.

## Data

```bash
python baselines/data/diff_data.py conddiff <dataset>
```

writes `$WORK_DIR/diff_data/<dataset>/conddiff/{train,test}/{sar,opt}/<6-digit id>` (the code orders files by
integer name) and `manifest_{train,test}.json` mapping ids to image names. SAR2Opt uses 512 × 512 centre crops
for training and testing; the SAR2EO test set is its first 4,000 pairs.

## Train

```bash
GPU=0 bash baselines/conddiff/train.sh <dataset>   # -> $CKPT_ROOT/conddiff/<dataset>/ema_final.pt
```

| dataset | resolution | batch | updates |
|---|---|---|---|
| qxs-saropt | 256 | 24 | 50,000 |
| sar2opt | 512 | 6 | 50,000 |
| sar2eo | 256 | 24 | 50,000 |
| spacenet6 | 256 | 24 | 50,000 |

`image_train.py` with 128 base channels, 3 residual blocks per level, 2,000 linear diffusion steps, lr 1e-4
annealed linearly to zero over 50,000 steps (`--lr_anneal_steps`), EMA 0.9999, checkpoints every 10,000 steps.
`train.sh` resumes from the newest `model*.pt` in `$WORK_DIR/conddiff/<dataset>`. `ema_final.pt` is
`ema_0.9999_050000.pt`. The upstream scripts set no random seed.

## Test

```bash
GPU=0 bash baselines/conddiff/test.sh <dataset>    # -> $WORK_DIR/results/conddiff/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/conddiff/<dataset>
```

Sampling: ancestral DDPM sampling respaced to 250 steps (`--timestep_respacing 250`, as in the upstream
`sample.sh`), with the EMA weights.

## Released weights

[`JeonghyeokDo/GeoSET`](https://huggingface.co/JeonghyeokDo/GeoSET), `baselines/<dataset>/conddiff/ema_final.pt`
(one per dataset). Place it at `$CKPT_ROOT/conddiff/<dataset>/ema_final.pt`:

```bash
hf download JeonghyeokDo/GeoSET --include "baselines/<dataset>/conddiff/*" --local-dir hf
mkdir -p $CKPT_ROOT/conddiff/<dataset> && cp hf/baselines/<dataset>/conddiff/ema_final.pt $CKPT_ROOT/conddiff/<dataset>/
```
