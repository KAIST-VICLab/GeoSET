# Comparison methods

The sixteen prior methods of Tables 2 and 3 of the paper, retrained on the four public benchmarks: 61
checkpoints (Seg-CycleGAN is evaluated on SpaceNet6 only). Each folder holds the scripts and configurations
used and, where the upstream code was changed, a patch against the upstream repository at a pinned commit. The
upstream code is fetched from its original repository and is not redistributed here. Metrics and links to
every checkpoint are in [MODEL_ZOO.md](../MODEL_ZOO.md); the weights are under `baselines/<dataset>/<method>/`
in [`JeonghyeokDo/GeoSET`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines).

| Method | Venue | Folder | Upstream | Commit | Code licence | Weights licence | Datasets |
|---|---|---|---|---|---|---|---|
| pix2pix | CVPR'17 | [`pix2pix`](pix2pix) | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | `2a7afba` | BSD | BSD | all four |
| CycleGAN | ICCV'17 | [`cyclegan`](cyclegan) | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | `2a7afba` | BSD | BSD | all four |
| pix2pixHD | CVPR'18 | [`pix2pixhd`](pix2pixhd) | [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD) | `14b3b3c` | BSD | BSD | all four |
| SPADE | CVPR'19 | [`spade`](spade) | [NVlabs/SPADE](https://github.com/NVlabs/SPADE) | `fecacc9` | CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 | all four |
| DDPM (SR3) | TPAMI'22 | [`ddpm-sr3`](ddpm-sr3) | [DeepSARRS/E3Diff](https://github.com/DeepSARRS/E3Diff) (stage 1) | `3860109` | none | no upstream licence | all four |
| SD2.1 FT | CVPR'22 | [`sd21-ft`](sd21-ft) | [KAIST-VICLab/C-DiffSET](https://github.com/KAIST-VICLab/C-DiffSET) (stage 1) | `6fc3802` | MIT | CreativeML Open RAIL++-M | all four |
| BBDM | CVPR'23 | [`bbdm`](bbdm) | [xuekt98/BBDM](https://github.com/xuekt98/BBDM) | `02c3b13` | MIT | MIT | all four |
| ControlNet | ICCV'23 | [`controlnet`](controlnet) | [huggingface/diffusers](https://github.com/huggingface/diffusers) training example | `v0.37.0` | Apache-2.0 | CreativeML Open RAIL++-M | all four |
| HI-Diff | NeurIPS'23 | [`hi-diff`](hi-diff) | [zhengchen1999/HI-Diff](https://github.com/zhengchen1999/HI-Diff) | `b3bfd16` | Apache-2.0 | Apache-2.0 | all four |
| ResShift | NeurIPS'23 | [`resshift`](resshift) | [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift) | `bb03b7d` | S-Lab License 1.0 | S-Lab License 1.0 | all four |
| StegoGAN | CVPR'24 | [`stegogan`](stegogan) | [sian-wusidi/StegoGAN](https://github.com/sian-wusidi/StegoGAN) | `cad6199` | none | no upstream licence | all four |
| CondDiff | GRSL'23 | [`conddiff`](conddiff) | [Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation](https://github.com/Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation) | `8326f54` | none | no upstream licence | all four |
| E3Diff | GRSL'24 | [`e3diff`](e3diff) | [DeepSARRS/E3Diff](https://github.com/DeepSARRS/E3Diff) | `3860109` | none | no upstream licence | all four |
| cBBDM | GRSL'25 | [`cbbdm`](cbbdm) | [egshkim/ConditionalBBDM-for-VHR-SAR-to-Optical](https://github.com/egshkim/ConditionalBBDM-for-VHR-SAR-to-Optical) | `8ce1593` | MIT | MIT | all four |
| Seg-CycleGAN | GRSL'25 | [`seg-cyclegan`](seg-cyclegan) | [NWPU-LHH/Seg-CycleGAN](https://github.com/NWPU-LHH/Seg-CycleGAN) | `f12ff88` | none | no upstream licence | SpaceNet6 |
| C-DiffSET | TCSVT'26 | [`c-diffset`](c-diffset) | [KAIST-VICLab/C-DiffSET](https://github.com/KAIST-VICLab/C-DiffSET) | `6fc3802` | MIT | CreativeML Open RAIL++-M | all four |

The full commit hashes are in each `fetch.sh`. SD2.1 FT, ControlNet and C-DiffSET are derivatives of Stable
Diffusion 2.1-base and carry the use-based restrictions of the CreativeML Open RAIL++-M licence. "none" means
that the upstream repository publishes no licence; for those methods this repository distributes only its own
scripts and patches. The licence texts are in
[`baselines/licenses/`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/licenses) on the Hub.

## Setup

```bash
export BASELINES_ROOT=/path/to/upstream_code DATA_ROOT=/path/to/data WORK_DIR=/path/to/work CKPT_ROOT=/path/to/checkpoints
bash baselines/setup.sh                 # runs every <method>/fetch.sh
bash baselines/setup.sh bbdm e3diff     # or only the methods named
```

Each `fetch.sh` clones the upstream repository into `$BASELINES_ROOT/<method>`, checks out the pinned commit
and applies `<method>.patch`; running it again leaves an applied patch in place. ControlNet needs nothing to be
fetched. Every method runs in the Python environment of its upstream code; the method README lists what is
needed beyond it.

## Environment contract

Every script reads its locations from environment variables; none contains an absolute path.

| Variable | Meaning |
|---|---|
| `BASELINES_ROOT` | upstream clones written by `fetch.sh` (`$BASELINES_ROOT/<method>`) |
| `DATA_ROOT` | the datasets in the layout of [docs/DATASETS.md](../docs/DATASETS.md) |
| `WORK_DIR` | derived data views, run directories and outputs |
| `CKPT_ROOT` | checkpoints, `$CKPT_ROOT/<method>/<dataset>/<file>` with the file names of the Hub folders |
| `GPU` | optional, GPU index (default `0`) |
| `PY` | optional, Python interpreter (default `python`) |

Configuration files spell these variables as `${VAR}` and are expanded with `envsubst` (GNU gettext) at run
time. Dataset arguments are `qxs-saropt`, `sar2opt`, `sar2eo` and `spacenet6`.

## Data

Prepare `$DATA_ROOT` as in [docs/DATASETS.md](../docs/DATASETS.md); SpaceNet6 needs the chips of
`scripts/datasets/build_spacenet6.py`. Each method family reads views built from the split lists in
[`splits/`](../splits/README.md):

| Builder | Methods | Output |
|---|---|---|
| `python baselines/data/gan_prepare.py <dataset>` | pix2pix, CycleGAN, pix2pixHD, SPADE, StegoGAN, Seg-CycleGAN | `$WORK_DIR/gan_data/<dataset>/` |
| `python baselines/data/gan_spacenet6_masks.py` | Seg-CycleGAN (building masks, needs `rasterio`) | `$WORK_DIR/gan_data/spacenet6/AB/SegMask/` |
| `python baselines/data/ldm_prepare.py <dataset>` | BBDM, cBBDM, ControlNet, SD2.1 FT, C-DiffSET | `$WORK_DIR/ldm_data/<dataset>/` |
| `python baselines/data/diff_data.py e3diff <dataset>` | DDPM (SR3), E3Diff | `$WORK_DIR/diff_data/<dataset>/e3diff/` |
| `python baselines/data/diff_data.py conddiff <dataset>` | CondDiff | `$WORK_DIR/diff_data/<dataset>/conddiff/` |
| `python baselines/data/diff_data.py paired <dataset>` | ResShift, HI-Diff | `$WORK_DIR/diff_data/<dataset>/paired/` |

`gan_prepare.py` uses `combine_A_and_B.py` of the pix2pix clone (`bash baselines/pix2pix/fetch.sh`). SAR2Opt
test views are the center 512 × 512 crops, and SAR2EO test views hold the first 4,000 test pairs.

## Train, test and evaluate

```bash
bash baselines/<method>/train.sh <dataset>     # writes $CKPT_ROOT/<method>/<dataset>/
bash baselines/<method>/test.sh <dataset>      # writes $WORK_DIR/results/<method>/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/<method>/<dataset> --data-root $DATA_ROOT
```

pix2pix and CycleGAN keep the upstream `<stem>_fake_B.png` names; `evaluate.py` reads both forms.

To test a released checkpoint, download its Hub folder into the same place:

```bash
hf download JeonghyeokDo/GeoSET --include "baselines/<dataset>/<method>/*" --local-dir hf
mkdir -p $CKPT_ROOT/<method>/<dataset> && cp hf/baselines/<dataset>/<method>/* $CKPT_ROOT/<method>/<dataset>/
```

Some methods need weights that are not part of this release: Stable Diffusion 2.1-base (SD2.1 FT, ControlNet,
C-DiffSET), the CompVis latent-diffusion VQ-f4 autoencoder (BBDM, cBBDM), the ResShift VQ-f4
autoencoder (downloaded by `resshift/fetch.sh`), the CLIP discriminator of `vision-aided-loss` (E3Diff) and the
torchvision VGG16-BN ImageNet weights (Seg-CycleGAN guide network). The method READMEs give the exact sources,
the changes made by each patch, the training budget and the per-dataset settings.
