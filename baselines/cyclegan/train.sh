#!/usr/bin/env bash
# Train CycleGAN (SAR -> EO) on one dataset with the settings of the released checkpoints.
# usage: bash baselines/cyclegan/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/gan_data/<dataset> (baselines/data/gan_prepare.py). Writes the run to
# $WORK_DIR/cyclegan/train/<dataset>/ and copies latest_net_G_{A,B}.pth to $CKPT_ROOT/cyclegan/<dataset>/net_G_{A,B}.pth.
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt) ARGS="--batch_size 8 --load_size 256 --crop_size 256 --n_epochs 100 --n_epochs_decay 100" ;;
  sar2opt)    ARGS="--batch_size 4 --load_size 600 --crop_size 512 --n_epochs 222 --n_epochs_decay 221 --save_epoch_freq 25" ;;
  sar2eo)     ARGS="--batch_size 8 --load_size 256 --crop_size 256 --n_epochs 6 --n_epochs_decay 6" ;;
  spacenet6)  ARGS="--batch_size 8 --load_size 256 --crop_size 256 --n_epochs 250 --n_epochs_decay 250 --save_epoch_freq 25" ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=$WORK_DIR/gan_data/$DS
RUN=$WORK_DIR/cyclegan/train
[ -d "$D/AB/trainA" ] || { echo "missing $D/AB (run baselines/data/gan_prepare.py $DS)" >&2; exit 1; }
cd "$BASELINES_ROOT/cyclegan"
$PY train.py --dataroot "$D/AB" --name "$DS" --model cycle_gan --direction AtoB $ARGS --num_threads 8 --no_html \
  --checkpoints_dir "$RUN"
mkdir -p "$CKPT_ROOT/cyclegan/$DS"
for g in G_A G_B; do cp "$RUN/$DS/latest_net_$g.pth" "$CKPT_ROOT/cyclegan/$DS/net_$g.pth"; done
echo "checkpoints: $CKPT_ROOT/cyclegan/$DS/net_G_{A,B}.pth"
