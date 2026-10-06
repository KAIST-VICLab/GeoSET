#!/usr/bin/env bash
# Train ResShift for 50,000 iterations on one dataset.
# usage: bash baselines/resshift/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/diff_data/<dataset>/paired (baselines/data/diff_data.py paired <dataset>);
# writes $CKPT_ROOT/resshift/<dataset>/ema_model.pth (the EMA weights at iteration 50,000).
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
case $DS in
  qxs-saropt|sar2opt|sar2eo|spacenet6) ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
D=$WORK_DIR/diff_data/$DS/paired
[ -d "$D/trainA" ] || { echo "missing $D (run baselines/data/diff_data.py paired $DS)" >&2; exit 1; }
export WORK_DIR BASELINES_ROOT CUDA_VISIBLE_DEVICES=${GPU:-0}
RUN=$WORK_DIR/resshift/$DS
mkdir -p "$RUN"
envsubst '${WORK_DIR} ${BASELINES_ROOT}' < "$HERE/configs/$DS.yaml" > "$RUN/config.yaml"
cd "$BASELINES_ROOT/resshift"
"$PY" main.py --cfg_path "$RUN/config.yaml" --save_dir "$RUN"
CKPT=$(ls -d "$RUN"/*/ema_ckpts/ema_model_50000.pth | sort | tail -1)
mkdir -p "$CKPT_ROOT/resshift/$DS"
cp "$CKPT" "$CKPT_ROOT/resshift/$DS/ema_model.pth"
echo "$CKPT -> $CKPT_ROOT/resshift/$DS/ema_model.pth"
