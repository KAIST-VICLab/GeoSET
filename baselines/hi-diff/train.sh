#!/usr/bin/env bash
# Train HI-Diff on one dataset: stage 1 (25,000 iterations) then stage 2 (25,000 iterations).
# usage: bash baselines/hi-diff/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/diff_data/<dataset>/paired (baselines/data/diff_data.py paired <dataset>); basicsr writes
# its run directories to $BASELINES_ROOT/hi-diff/experiments (an existing run of the same name is archived).
# Writes $CKPT_ROOT/hi-diff/<dataset>/{S1_net_g,S1_net_le,S2_net_g,S2_net_le_dm,S2_net_d}_latest.pth.
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
[ -d "$D/valA" ] || { echo "missing $D (run baselines/data/diff_data.py paired $DS)" >&2; exit 1; }
export WORK_DIR BASELINES_ROOT CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=$WORK_DIR/hi-diff/$DS
mkdir -p "$RUN" "$CKPT_ROOT/hi-diff/$DS"
cd "$BASELINES_ROOT/hi-diff"
for st in S1 S2; do
  envsubst '${WORK_DIR} ${BASELINES_ROOT}' < "$HERE/configs/${DS}_${st,,}.yml" > "$RUN/${st,,}.yml"
  "$PY" train.py -opt "$RUN/${st,,}.yml"
  for f in experiments/train_HI_Diff_"$DS"_$st/models/net_*_latest.pth; do
    cp "$f" "$CKPT_ROOT/hi-diff/$DS/${st}_$(basename "$f")"
  done
done
ls -l "$CKPT_ROOT/hi-diff/$DS"
