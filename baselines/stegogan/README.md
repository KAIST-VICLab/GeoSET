# StegoGAN

Wu, Chen, Mermet, Hurni, Schindler, Gonthier and Landrieu, "StegoGAN: Leveraging steganography for non-bijective
image-to-image translation", CVPR 2024. Unpaired training with A = SAR, B = EO; the output is G_A applied to the
SAR image alone.

| | |
|---|---|
| Upstream | https://github.com/sian-wusidi/StegoGAN |
| Pinned commit | `cad61997c0f82793444f60f81298142b80cdf3c1` |
| Licence | The upstream repository contains no licence file. The upstream code is used unchanged and is not distributed here; `fetch.sh` downloads it. |

## Setup

```bash
export BASELINES_ROOT=/path/to/upstream_code DATA_ROOT=/path/to/data WORK_DIR=/path/to/work CKPT_ROOT=/path/to/checkpoints
bash baselines/stegogan/fetch.sh    # clones into $BASELINES_ROOT/stegogan at the pinned commit (no patch)
bash baselines/pix2pix/fetch.sh     # gan_prepare.py uses its combine_A_and_B.py
```

Optional: `GPU` (default 0), `PY` (default `python`). Python packages: `torch`, `torchvision`, `numpy`, `pillow`,
`dominate` (and `opencv-python` for `gan_prepare.py`). The upstream model calls `.cuda()`, so a GPU is required.

## Data

```bash
python baselines/data/gan_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR BASELINES_ROOT
```

This builds `$WORK_DIR/gan_data/<dataset>/` from the split lists in `splits/` (A = SAR, B = EO); see
`baselines/pix2pix/README.md`. StegoGAN trains on `AB/{trainA,trainB}` and is tested on `AB/{testA,testB}`
(`sar2opt`: `test512/`, the centre 512 × 512 crops). `spacenet6` needs the chips of
`scripts/datasets/build_spacenet6.py`.

## Train

```bash
bash baselines/stegogan/train.sh <dataset>   # run in $WORK_DIR/stegogan/train/<dataset>, checkpoints -> $CKPT_ROOT/stegogan/<dataset>/net_G_{A,B}.pth
```

| Dataset | Load → crop | Batch | Epochs (constant + linear decay) |
|---|---|---:|---|
| `qxs-saropt` | 256 → 256 | 4 | 15 + 15 |
| `sar2opt` | 600 → random 512 | 2 | 111 + 111 |
| `sar2eo` | 256 → 256 | 4 | 4 + 4 |
| `spacenet6` | 256 → 256 | 4 | 125 + 125 |

All datasets: `--model stego_gan --lambda_reg 0.3 --lambda_consistency 1 --resnet_layer 8 --fusionblock
--lr_policy linear`, with the upstream defaults otherwise (G_A `resnet_9blocks_maskv1`, G_B
`resnet_9blocks_maskv3`, mask group 256, LSGAN loss, Adam with lr 2e-4). `--resnet_layer` and `--fusionblock`
define the architecture and are passed at test time too. `net_G_A.pth` / `net_G_B.pth` are the run's
`latest_net_G_A.pth` / `latest_net_G_B.pth`.

## Test

```bash
bash baselines/stegogan/test.sh <dataset>    # -> $WORK_DIR/results/stegogan/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/stegogan/<dataset>
```

`test.sh` runs the upstream `test.py` at the evaluation size (256; `sar2opt`: 512) and keeps
`images/fake_B_clean/<stem>.png`, i.e. G_A(SAR). The upstream test script runs the full training forward pass,
so it loads both generators and reads the EO test images as well; among the other images it writes, `fake_B`
feeds G_A with features that G_B extracts from the EO image and is not a translation of the SAR image alone.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/stegogan/net_G_A.pth` and `net_G_B.pth`. Copy them to
`$CKPT_ROOT/stegogan/<dataset>/` and run `test.sh`.
