#!/usr/bin/env bash
# Train Seg-CycleGAN (SAR -> EO, building-guided) on SpaceNet6 with the settings of the released checkpoints.
# usage: bash baselines/seg-cyclegan/train.sh spacenet6
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/gan_data/spacenet6/AB with SegMask/ (baselines/data/gan_prepare.py, gan_spacenet6_masks.py).
# The guide SegNet $CKPT_ROOT/seg-cyclegan/spacenet6/segnet_guide.pth (released file) is trained first if absent.
# Writes the run to $WORK_DIR/seg-cyclegan/train/spacenet6/ and copies latest_net_G_{A,B}.pth to
# $CKPT_ROOT/seg-cyclegan/spacenet6/net_G_{A,B}.pth.
set -euo pipefail
DS=${1:?usage: train.sh spacenet6}
[ "$DS" = spacenet6 ] || { echo "Seg-CycleGAN uses the SpaceNet6 building masks: spacenet6 only" >&2; exit 2; }
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
export BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT")
WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$WORK_DIR/gan_data/spacenet6/AB
CK=$CKPT_ROOT/seg-cyclegan/spacenet6
RUN=$WORK_DIR/seg-cyclegan/train
[ -d "$D/SegMask" ] || { echo "missing $D/SegMask (run baselines/data/gan_spacenet6_masks.py)" >&2; exit 1; }
mkdir -p "$CK"
export SEGNET_WEIGHTS=$CK/segnet_guide.pth
[ -f "$SEGNET_WEIGHTS" ] || $PY "$HERE/train_segnet.py" --data-dir "$D" --out "$SEGNET_WEIGHTS"
cd "$BASELINES_ROOT/seg-cyclegan"
$PY train.py --dataroot "$D" --name spacenet6 --model cycle_gan --dataset_mode unaligned --direction AtoB \
  --batch_size 8 --load_size 256 --crop_size 256 --no_flip --n_epochs 250 --n_epochs_decay 250 \
  --save_epoch_freq 25 --num_threads 8 --no_html --display_id 0 --checkpoints_dir "$RUN"
for g in G_A G_B; do cp "$RUN/spacenet6/latest_net_$g.pth" "$CK/net_$g.pth"; done
echo "checkpoints: $CK/net_G_{A,B}.pth, $CK/segnet_guide.pth"
