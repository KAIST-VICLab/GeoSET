#!/usr/bin/env bash
# Train SPADE (SAR -> EO, --label_nc 0 --no_instance) on one dataset with the settings of the
# released checkpoints.
# usage: bash baselines/spade/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/gan_data/<dataset> (baselines/data/gan_prepare.py). Writes the run to
# $WORK_DIR/spade/train/<dataset>/ and copies latest_net_G.pth to $CKPT_ROOT/spade/<dataset>/net_G.pth.
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt) ARGS="--preprocess_mode resize_and_crop --load_size 256 --crop_size 256 --batchSize 16 --niter 60 --niter_decay 60 --save_epoch_freq 20 --save_latest_freq 16000" ;;
  sar2opt)    ARGS="--preprocess_mode crop --load_size 600 --crop_size 512 --batchSize 8 --niter 100 --niter_decay 100 --save_epoch_freq 50 --save_latest_freq 160000" ;;
  sar2eo)     ARGS="--preprocess_mode resize_and_crop --load_size 256 --crop_size 256 --batchSize 16 --niter 14 --niter_decay 14 --save_epoch_freq 14 --save_latest_freq 160000" ;;
  spacenet6)  ARGS="--preprocess_mode resize_and_crop --load_size 256 --crop_size 256 --batchSize 16 --niter 375 --niter_decay 375 --save_epoch_freq 150 --save_latest_freq 160000" ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=$WORK_DIR/gan_data/$DS
RUN=$WORK_DIR/spade/train
[ -d "$D/AB/trainA" ] || { echo "missing $D/AB (run baselines/data/gan_prepare.py $DS)" >&2; exit 1; }
cd "$BASELINES_ROOT/spade"
$PY train.py --name "$DS" --dataset_mode custom --label_dir "$D/AB/trainA" --image_dir "$D/AB/trainB" \
  --label_nc 0 --no_instance --no_pairing_check $ARGS --aspect_ratio 1 --nThreads 8 \
  --checkpoints_dir "$RUN" --no_html
mkdir -p "$CKPT_ROOT/spade/$DS"
cp "$RUN/$DS/latest_net_G.pth" "$CKPT_ROOT/spade/$DS/net_G.pth"
echo "checkpoint: $CKPT_ROOT/spade/$DS/net_G.pth"
