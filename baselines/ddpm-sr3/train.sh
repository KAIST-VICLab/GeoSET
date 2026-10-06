#!/usr/bin/env bash
# Train DDPM (SR3) -- E3Diff stage 1 -- for 250,000 iterations on one dataset.
# usage: bash baselines/ddpm-sr3/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/diff_data/<dataset>/e3diff (baselines/data/diff_data.py e3diff <dataset>);
# writes $CKPT_ROOT/ddpm-sr3/<dataset>/gen.pth.
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
case $DS in
  qxs-saropt|sar2opt|spacenet6) MAIN=$HERE/e3diff_rgb_main.py ;;
  sar2eo) MAIN=$BASELINES_ROOT/ddpm-sr3/main.py ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
D=$WORK_DIR/diff_data/$DS/e3diff
[ -d "$D/train/SAR-PPB" ] || { echo "missing $D (run baselines/data/diff_data.py e3diff $DS)" >&2; exit 1; }
export WORK_DIR CUDA_VISIBLE_DEVICES=${GPU:-0} E3DIFF_REPO=$BASELINES_ROOT/ddpm-sr3
export PYTHONPATH=$HERE/shims:$HERE/stub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=$WORK_DIR/ddpm-sr3/$DS
mkdir -p "$RUN"
envsubst '${WORK_DIR}' < "$HERE/configs/$DS.json" > "$RUN/train.json"
cd "$RUN"
"$PY" "$MAIN" -c train.json -p train -enable_wandb "" --seed 1
CKPT=$(ls -t "$RUN"/experiments/ddpm-sr3_"$DS"_*/checkpoint/I250000_E*_gen.pth | head -1)
mkdir -p "$CKPT_ROOT/ddpm-sr3/$DS"
cp "$CKPT" "$CKPT_ROOT/ddpm-sr3/$DS/gen.pth"
echo "$CKPT -> $CKPT_ROOT/ddpm-sr3/$DS/gen.pth"
