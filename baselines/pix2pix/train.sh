#!/usr/bin/env bash
# Train pix2pix (SAR -> EO) on one dataset with the settings of the released checkpoints.
# usage: bash baselines/pix2pix/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/gan_data/<dataset> (baselines/data/gan_prepare.py). Writes the run to
# $WORK_DIR/pix2pix/train/<dataset>/ and copies latest_net_G.pth to $CKPT_ROOT/pix2pix/<dataset>/net_G.pth.
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt) ARGS="--batch_size 16 --load_size 256 --crop_size 256 --n_epochs 60 --n_epochs_decay 60 --num_threads 16" ;;
  sar2opt)    ARGS="--batch_size 8 --load_size 600 --crop_size 512 --n_epochs 100 --n_epochs_decay 100 --num_threads 8" ;;
  sar2eo)     ARGS="--batch_size 16 --load_size 256 --crop_size 256 --n_epochs 14 --n_epochs_decay 14 --num_threads 8" ;;
  spacenet6)  ARGS="--batch_size 16 --load_size 256 --crop_size 256 --n_epochs 250 --n_epochs_decay 250 --save_epoch_freq 25 --num_threads 8" ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=$WORK_DIR/gan_data/$DS
RUN=$WORK_DIR/pix2pix/train
[ -d "$D/combined/train" ] || { echo "missing $D/combined (run baselines/data/gan_prepare.py $DS)" >&2; exit 1; }
cd "$BASELINES_ROOT/pix2pix"
$PY train.py --dataroot "$D/combined" --name "$DS" --model pix2pix --direction AtoB $ARGS --no_html \
  --checkpoints_dir "$RUN"
mkdir -p "$CKPT_ROOT/pix2pix/$DS"
cp "$RUN/$DS/latest_net_G.pth" "$CKPT_ROOT/pix2pix/$DS/net_G.pth"
echo "checkpoint: $CKPT_ROOT/pix2pix/$DS/net_G.pth"
