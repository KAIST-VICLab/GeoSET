# Seg-CycleGAN

Zhang, Li, Lin, Zhang, Fan, Liu and Liu, "Seg-CycleGAN: SAR-to-optical image translation guided by a downstream
task", IEEE GRSL 2025. CycleGAN with an additional segmentation loss: a frozen guide SegNet segments the generated
EO image G_A(SAR), supervised by the target-class mask of the SAR tile. Here the target class is buildings, from the
SpaceNet6 footprints, so this baseline is trained and evaluated on `spacenet6` only.

| | |
|---|---|
| Upstream | https://github.com/NWPU-LHH/Seg-CycleGAN |
| Pinned commit | `f12ff88ac089b7099443ffa2c34fc47e035def6a` |
| Licence | The upstream repository contains no licence file (its CycleGAN components derive from junyanz/pytorch-CycleGAN-and-pix2pix, BSD). Only our changes are distributed here; `fetch.sh` downloads the upstream code. |

## Setup

```bash
export BASELINES_ROOT=/path/to/upstream_code DATA_ROOT=/path/to/data WORK_DIR=/path/to/work CKPT_ROOT=/path/to/checkpoints
bash baselines/seg-cyclegan/fetch.sh   # clones into $BASELINES_ROOT/seg-cyclegan and applies seg-cyclegan.patch
bash baselines/pix2pix/fetch.sh        # gan_prepare.py uses its combine_A_and_B.py
```

Optional: `GPU` (default 0), `PY` (default `python`). Python packages: `torch`, `torchvision`, `numpy`, `pillow`,
`dominate`, `opencv-python`, and `rasterio` for the building masks.

`seg-cyclegan.patch` changes:
- `train.py`: the guide SegNet weights are read from `$SEGNET_WEIGHTS` (set by `train.sh`) instead of a fixed
  path on the authors' machine.
- `sar2opt_eval.py`: the output directory is `$SEGCYC_OUT` (created if missing).
- `data/evaluation_dataset.py`: the input directory is `$SEGCYC_EVAL_DIR`; its files are processed in sorted order.
- `data/unaligned_dataset.py`: removes the unused `libtiff` import.

## Data

```bash
python scripts/datasets/build_spacenet6.py --data-root $DATA_ROOT   # SpaceNet6 chips
python baselines/data/gan_prepare.py spacenet6                       # env: DATA_ROOT WORK_DIR BASELINES_ROOT
python baselines/data/gan_spacenet6_masks.py                         # env: DATA_ROOT WORK_DIR
```

`gan_prepare.py` builds `$WORK_DIR/gan_data/spacenet6/AB/{trainA,trainB,testA,testB}` (A = SAR, B = EO; see
`baselines/pix2pix/README.md`). `gan_spacenet6_masks.py` writes `AB/SegMask/<chip>.png` for every training tile: the
footprints of `train/AOI_11_Rotterdam/geojson_buildings` burned in as 255 (background 0) on the 900 × 900 PS-RGB grid
of the tile, then resized to 256 × 256 (nearest), aligned with the chips. The loader reads the mask with the name
of the SAR image A.

## Guide SegNet

```bash
python baselines/seg-cyclegan/train_segnet.py --data-dir $WORK_DIR/gan_data/spacenet6/AB --out segnet_guide.pth
```

The upstream repository ships neither the guide weights nor a SegNet trainer. `train_segnet.py` trains the
repository's `SegNet(3, 2)` (VGG16-BN encoder initialised from the torchvision ImageNet weights) on the EO training
tiles (read as BGR, resized to 224, scaled to [0, 1]) against `SegMask/`, with class-weighted cross-entropy
(0.143 background, 0.857 building), Adam (lr 1e-3), batch 16 and 50 epochs; the last 128 training tiles in sorted
order are held out and the checkpoint with the best holdout mIoU is kept. The released `segnet_guide.pth` was
trained this way; `train.sh` trains it only if `$CKPT_ROOT/seg-cyclegan/spacenet6/segnet_guide.pth` is absent.

## Train

```bash
bash baselines/seg-cyclegan/train.sh spacenet6   # run in $WORK_DIR/seg-cyclegan/train/spacenet6, checkpoints -> $CKPT_ROOT/seg-cyclegan/spacenet6/net_G_{A,B}.pth
```

256 × 256, batch 8, 250 + 250 epochs (constant + linear decay), `--no_flip` (the masks are not augmented), with the
repository defaults otherwise: segmentation loss weight 0.3 with class weights (0.5, 0.5), and the CycleGAN settings
of `baselines/cyclegan` (`resnet_9blocks`, LSGAN, λ_A = λ_B = 10, identity loss 0.5, Adam with lr 2e-4).
`net_G_A.pth` / `net_G_B.pth` are the run's `latest_net_G_A.pth` / `latest_net_G_B.pth`.

## Test

```bash
bash baselines/seg-cyclegan/test.sh spacenet6    # -> $WORK_DIR/results/seg-cyclegan/spacenet6/<stem>.png
python scripts/evaluate.py --dataset spacenet6 --pred-dir $WORK_DIR/results/seg-cyclegan/spacenet6
```

`test.sh` runs the repository's `sar2opt_eval.py` (`--dataset_mode evaluation --preprocess none`), which writes
G_A(SAR) for every test SAR image. It loads both generators; the guide SegNet is not used at test time.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/spacenet6/seg-cyclegan/net_G_A.pth`, `net_G_B.pth` and `segnet_guide.pth`. Copy
them to `$CKPT_ROOT/seg-cyclegan/spacenet6/` and run `test.sh` (or `train.sh`, which then reuses the guide SegNet).
