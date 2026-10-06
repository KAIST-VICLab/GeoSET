#!/usr/bin/env bash
# Train C-DiffSET (stage 2, confidence-guided) on one dataset with the settings of the released checkpoints.
# usage: bash baselines/c-diffset/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT DATA_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Initialised from the stage-1 run of baselines/sd21-ft/train.sh (accelerator_path in configs/<dataset>.yaml).
# Writes the run to $WORK_DIR/c-diffset/<dataset>/ and copies checkpoint-40000 to
# $CKPT_ROOT/c-diffset/<dataset>/diffusion_pytorch_model.safetensors.
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in qxs-saropt|sar2opt|sar2eo|spacenet6) ;; *) echo "unknown dataset: $DS" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
GPU=${GPU:-0}
export DATA_ROOT=$(realpath -m "$DATA_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
export CUDA_VISIBLE_DEVICES=$GPU
case $WORK_DIR in *train*) echo "WORK_DIR must not contain 'train' (the feeders take the split from the list path)" >&2; exit 1 ;; esac
[ -f "$WORK_DIR/ldm_data/$DS/cdiffset/train.txt" ] || { echo "missing $WORK_DIR/ldm_data/$DS (run baselines/data/ldm_prepare.py $DS)" >&2; exit 1; }
CFG=$WORK_DIR/configs/c-diffset/$DS.yaml
mkdir -p "$(dirname "$CFG")" "$CKPT_ROOT/c-diffset/$DS"
envsubst '${DATA_ROOT} ${WORK_DIR} ${CKPT_ROOT}' < "$HERE/configs/$DS.yaml" > "$CFG"
INIT=$(sed -n 's/^accelerator_path: //p' "$CFG")
[ -f "$INIT" ] || { echo "missing stage-1 checkpoint $INIT (run baselines/sd21-ft/train.sh $DS)" >&2; exit 1; }
cd "$BASELINES_ROOT/c-diffset"
$PY main.py --config "$CFG" --gpu "$GPU"
cp "$WORK_DIR/c-diffset/$DS/checkpoint-40000/model.safetensors" "$CKPT_ROOT/c-diffset/$DS/diffusion_pytorch_model.safetensors"
echo "checkpoint: $CKPT_ROOT/c-diffset/$DS/diffusion_pytorch_model.safetensors"
