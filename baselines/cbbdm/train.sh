#!/usr/bin/env bash
# Train cBBDM (cBBDM-f4, SAR -> EO) on one dataset with the settings of the released checkpoints.
# usage: bash baselines/cbbdm/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/ldm_data/<dataset>/bbdm (baselines/data/ldm_prepare.py) and $CKPT_ROOT/vq-f4/model.ckpt.
# Writes the run to $WORK_DIR/cbbdm/<dataset>/cBBDM-f4/ and copies last_model.pth to $CKPT_ROOT/cbbdm/<dataset>/.
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in qxs-saropt|sar2opt|sar2eo|spacenet6) ;; *) echo "unknown dataset: $DS" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
export CUDA_VISIBLE_DEVICES=${GPU:-0}
[ -d "$WORK_DIR/ldm_data/$DS/bbdm" ] || { echo "missing $WORK_DIR/ldm_data/$DS (run baselines/data/ldm_prepare.py $DS)" >&2; exit 1; }
[ -f "$CKPT_ROOT/vq-f4/model.ckpt" ] || { echo "missing $CKPT_ROOT/vq-f4/model.ckpt (see baselines/cbbdm/README.md)" >&2; exit 1; }
CFG=$WORK_DIR/configs/cbbdm/$DS.yaml
mkdir -p "$(dirname "$CFG")" "$CKPT_ROOT/cbbdm/$DS"
envsubst '${WORK_DIR} ${CKPT_ROOT}' < "$HERE/configs/$DS.yaml" > "$CFG"
cd "$BASELINES_ROOT/cbbdm"
$PY main.py -c "$CFG" -t --gpu_ids 0 -r "$WORK_DIR/cbbdm"
cp "$WORK_DIR/cbbdm/$DS/cBBDM-f4/checkpoint/last_model.pth" "$CKPT_ROOT/cbbdm/$DS/last_model.pth"
echo "checkpoint: $CKPT_ROOT/cbbdm/$DS/last_model.pth"
