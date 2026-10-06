#!/usr/bin/env bash
# Train pix2pixHD (SAR -> EO, --label_nc 0 --no_instance) on one dataset with the settings of the
# released checkpoints.
# usage: bash baselines/pix2pixhd/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/gan_data/<dataset> (baselines/data/gan_prepare.py). Writes the run to
# $WORK_DIR/pix2pixhd/train/<dataset>/ and copies latest_net_G.pth to $CKPT_ROOT/pix2pixhd/<dataset>/net_G.pth.
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt) ARGS="--resize_or_crop resize_and_crop --loadSize 256 --fineSize 256 --batchSize 16 --niter 60 --niter_decay 60 --save_epoch_freq 20" ;;
  sar2opt)    ARGS="--resize_or_crop crop --loadSize 600 --fineSize 512 --batchSize 8 --niter 100 --niter_decay 100 --save_epoch_freq 50" ;;
  sar2eo)     ARGS="--resize_or_crop resize_and_crop --loadSize 256 --fineSize 256 --batchSize 16 --niter 14 --niter_decay 14 --save_epoch_freq 50" ;;
  spacenet6)  ARGS="--resize_or_crop resize_and_crop --loadSize 256 --fineSize 256 --batchSize 16 --niter 375 --niter_decay 375 --save_epoch_freq 150" ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=$WORK_DIR/gan_data/$DS
RUN=$WORK_DIR/pix2pixhd/train
[ -d "$D/p2phd/train_A" ] || { echo "missing $D/p2phd (run baselines/data/gan_prepare.py $DS)" >&2; exit 1; }
cd "$BASELINES_ROOT/pix2pixhd"
$PY train.py --name "$DS" --dataroot "$D/p2phd" --label_nc 0 --no_instance $ARGS \
  --nThreads 8 --save_latest_freq 16000 --checkpoints_dir "$RUN" --no_html
mkdir -p "$CKPT_ROOT/pix2pixhd/$DS"
cp "$RUN/$DS/latest_net_G.pth" "$CKPT_ROOT/pix2pixhd/$DS/net_G.pth"
echo "checkpoint: $CKPT_ROOT/pix2pixhd/$DS/net_G.pth"
