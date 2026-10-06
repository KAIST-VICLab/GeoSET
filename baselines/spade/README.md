# SPADE

Park, Liu, Wang and Zhu, "Semantic image synthesis with spatially-adaptive normalization", CVPR 2019. Used as a
paired translator: the SAR image replaces the semantic label map (`--label_nc 0 --no_instance`) and the EO image is
the target.

| | |
|---|---|
| Upstream | https://github.com/NVlabs/SPADE |
| Pinned commit | `fecacc920c1367a038995c45a39c15f6521ca64f` |
| Licence | CC BY-NC-SA 4.0, Copyright (C) 2019 NVIDIA Corporation (upstream `LICENSE.md`): non-commercial use; adaptations, including `spade.patch` and the released weights, are under the same licence |
| Bundled | [vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) at `7553990fb9a917cddd9342e89b6dc12a70573f5b`, MIT, Copyright (c) 2018 Jiayuan Mao |

## Setup

```bash
export BASELINES_ROOT=/path/to/upstream_code DATA_ROOT=/path/to/data WORK_DIR=/path/to/work CKPT_ROOT=/path/to/checkpoints
bash baselines/spade/fetch.sh       # clones into $BASELINES_ROOT/spade and applies spade.patch
bash baselines/pix2pix/fetch.sh     # gan_prepare.py uses its combine_A_and_B.py
```

Optional: `GPU` (default 0), `PY` (default `python`). Python packages: `torch`, `torchvision`, `numpy`, `pillow`,
`dominate`, `dill` (and `opencv-python` for `gan_prepare.py`).

`spade.patch` changes:
- `data/pix2pix_dataset.py`: with `--label_nc 0` the label image is loaded and normalised like an RGB image
  instead of as an integer label map.
- `models/pix2pix_model.py`: with `--label_nc 0` that image is passed to the networks as is (no one-hot encoding).
- `options/base_options.py`, `models/networks/discriminator.py`: with `--label_nc 0` the input map has 3 channels.
- `util/visualizer.py`: the `scipy.misc` import is optional (removed in SciPy 1.12; only used with `--tf_log`);
  the input map is saved as an image.
- `models/networks/sync_batchnorm/`: the Synchronized-BatchNorm-PyTorch module that SPADE imports (upstream asks
  for it to be cloned there), unchanged.

The released weights load only into the patched code.

## Data

```bash
python baselines/data/gan_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR BASELINES_ROOT
```

This builds `$WORK_DIR/gan_data/<dataset>/` from the split lists in `splits/`; see `baselines/pix2pix/README.md`.
SPADE reads `AB/trainA` (`--label_dir`) and `AB/trainB` (`--image_dir`) and is tested on `AB/testA`, `AB/testB`
(`sar2opt`: `test512/`, the centre 512 × 512 crops). `spacenet6` needs the chips of
`scripts/datasets/build_spacenet6.py`.

## Train

```bash
bash baselines/spade/train.sh <dataset>   # run in $WORK_DIR/spade/train/<dataset>, checkpoint -> $CKPT_ROOT/spade/<dataset>/net_G.pth
```

| Dataset | `--preprocess_mode`, load → crop | Batch | Epochs (`niter` + `niter_decay`) |
|---|---|---:|---|
| `qxs-saropt` | `resize_and_crop`, 256 → 256 | 16 | 60 + 60 |
| `sar2opt` | `crop`, 600 → random 512 | 8 | 100 + 100 |
| `sar2eo` | `resize_and_crop`, 256 → 256 | 16 | 14 + 14 |
| `spacenet6` | `resize_and_crop`, 256 → 256 | 16 | 375 + 375 |

Upstream defaults otherwise: SPADE generator (64 filters, spectral norm, synchronized batch norm, no encoder),
two multi-scale discriminators, hinge loss with feature matching (λ = 10) and VGG loss (λ = 10), Adam with
β = (0, 0.9) and TTUR learning rates (generator 1e-4, discriminator 4e-4). `--no_pairing_check` skips the
upstream name check, but the dataset still asserts that each label/image pair has the same file stem.
`net_G.pth` is the run's `latest_net_G.pth`.

## Test

```bash
bash baselines/spade/test.sh <dataset>    # -> $WORK_DIR/results/spade/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/spade/<dataset>
```

`test.sh` runs the upstream `test.py` (evaluation mode) at the evaluation size (256; `sar2opt`:
`--preprocess_mode crop` at 512 on the centre crops) and keeps `synthesized_image/<stem>.png`.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/spade/net_G.pth` (CC BY-NC-SA 4.0). Copy it to
`$CKPT_ROOT/spade/<dataset>/net_G.pth` and run `test.sh`.
