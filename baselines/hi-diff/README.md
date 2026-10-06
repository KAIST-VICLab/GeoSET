# HI-Diff

The HI-Diff rows of Tables 2 and 3 (Chen et al., "Hierarchical integration diffusion model for realistic image
deblurring", NeurIPS 2023), retrained on each dataset: a Transformer restorer guided by a latent prior, trained in
two stages (stage 1: restorer and latent encoder; stage 2: adds the latent diffusion model, initialised from
stage 1).

| | |
|---|---|
| Upstream | [zhengchen1999/HI-Diff](https://github.com/zhengchen1999/HI-Diff) |
| Pinned commit | `b3bfd167997e27f8edd57681cf70e5031a0e35f2` |
| Licence | [Apache-2.0](https://github.com/zhengchen1999/HI-Diff/blob/b3bfd167997e27f8edd57681cf70e5031a0e35f2/LICENSE) |

## Environment

```bash
export BASELINES_ROOT=/path/to/upstream_code   # fetch.sh clones into $BASELINES_ROOT/hi-diff
export DATA_ROOT=/path/to/data
export WORK_DIR=/path/to/work
export CKPT_ROOT=/path/to/checkpoints          # $CKPT_ROOT/hi-diff/<dataset>/{S1,S2}_net_*_latest.pth
# optional: GPU (default 0), PY (default python)
bash baselines/hi-diff/fetch.sh
```

Python packages: `torch`, `torchvision`, `basicsr==1.4.2`, `einops`, `opencv-python`, `natsort`, `scikit-image`.
With torchvision >= 0.17, `basicsr` 1.4.2 needs `torchvision.transforms.functional_tensor` replaced by
`torchvision.transforms.functional` in `basicsr/data/degradations.py`. `hi-diff.patch` passes
`weights_only=False` to the two `torch.load` calls that read basicsr checkpoints (`hi_diff/utils/base_model.py`,
`train.py`), as required by PyTorch >= 2.6.

`configs/<dataset>_{s1,s2,test}.yml` are the option files (`${VAR}` paths, expanded with `envsubst` from GNU
gettext).
`test_order/<dataset>.txt` lists the test images in the order in which the released outputs were produced (it is
passed as `meta_info_file`; sampling draws one noise tensor per image in this order).

## Data

```bash
python baselines/data/diff_data.py paired <dataset>
```

writes `$WORK_DIR/diff_data/<dataset>/paired/{train,test}{A,B}` (A = SAR, B = EO; SAR2Opt training reads the
600 × 600 tiles, testing the 512 × 512 centre crops), `val{A,B}` (a fixed subset of the test split that is scored
every 5,000 iterations for monitoring only: QXS-SAROPT the 100 lowest image numbers, SAR2Opt the first 64 names,
SAR2EO and SpaceNet6 every k-th name for 100 images) and, for SAR2EO, `meta_train.txt`.

## Train

```bash
GPU=0 bash baselines/hi-diff/train.sh <dataset>    # -> $CKPT_ROOT/hi-diff/<dataset>/{S1,S2}_net_*_latest.pth
```

Every dataset: stage 1 for 25,000 iterations, then stage 2 for 25,000 iterations; batch 8, random 256 × 256
patches with flips and rotations, AdamW (lr 2e-4, weight decay 1e-4), cosine-annealing restarts (periods 8,000
and 17,000), L1 loss (stage 2: plus the L1 loss on the latent prior), 8 diffusion steps (linear schedule 0.1 to
0.99), `manual_seed: 100`. basicsr writes its run directories to `$BASELINES_ROOT/hi-diff/experiments/` and
renames an existing run directory of the same name before starting again.

## Test

```bash
GPU=0 bash baselines/hi-diff/test.sh <dataset>     # -> $WORK_DIR/results/hi-diff/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/hi-diff/<dataset>
```

Uses the stage-2 `net_g`, `net_le_dm` and `net_d` weights, `manual_seed: 100`.

## Released weights

[`JeonghyeokDo/GeoSET`](https://huggingface.co/JeonghyeokDo/GeoSET), `baselines/<dataset>/hi-diff/`:
`S1_net_g_latest.pth`, `S1_net_le_latest.pth` (stage 1) and `S2_net_g_latest.pth`, `S2_net_le_dm_latest.pth`,
`S2_net_d_latest.pth` (stage 2; the three files used by `test.sh`). Place them in `$CKPT_ROOT/hi-diff/<dataset>/`:

```bash
hf download JeonghyeokDo/GeoSET --include "baselines/<dataset>/hi-diff/*" --local-dir hf
mkdir -p $CKPT_ROOT/hi-diff/<dataset> && cp hf/baselines/<dataset>/hi-diff/*.pth $CKPT_ROOT/hi-diff/<dataset>/
```
