#!/usr/bin/env bash
# Train BBDM (LBBDM-f4, SAR -> EO) on one dataset with the settings of the released checkpoints.
# usage: bash baselines/bbdm/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/ldm_data/<dataset>/bbdm (baselines/data/ldm_prepare.py) and $CKPT_ROOT/vq-f4/model.ckpt.
# Writes the run to $WORK_DIR/bbdm/<dataset>/LBBDM-f4/ and copies last_model.pth to $CKPT_ROOT/bbdm/<dataset>/.
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in qxs-saropt|sar2opt|sar2eo|spacenet6) ;; *) echo "unknown dataset: $DS" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
export CUDA_VISIBLE_DEVICES=${GPU:-0}
[ -d "$WORK_DIR/ldm_data/$DS/bbdm" ] || { echo "missing $WORK_DIR/ldm_data/$DS (run baselines/data/ldm_prepare.py $DS)" >&2; exit 1; }
[ -f "$CKPT_ROOT/vq-f4/model.ckpt" ] || { echo "missing $CKPT_ROOT/vq-f4/model.ckpt (see baselines/bbdm/README.md)" >&2; exit 1; }
CFG=$WORK_DIR/configs/bbdm/$DS.yaml
mkdir -p "$(dirname "$CFG")" "$CKPT_ROOT/bbdm/$DS"
envsubst '${WORK_DIR} ${CKPT_ROOT}' < "$HERE/configs/$DS.yaml" > "$CFG"
cd "$BASELINES_ROOT/bbdm"
$PY main.py -c "$CFG" -t --gpu_ids 0 -r "$WORK_DIR/bbdm"
cp "$WORK_DIR/bbdm/$DS/LBBDM-f4/checkpoint/last_model.pth" "$CKPT_ROOT/bbdm/$DS/last_model.pth"
echo "checkpoint: $CKPT_ROOT/bbdm/$DS/last_model.pth"
