#!/usr/bin/env bash
# Train SD2.1 FT (Stable Diffusion 2.1-base UNet fine-tuned with SAR-latent concatenation) on one dataset
# with the settings of the released checkpoints.
# usage: bash baselines/sd21-ft/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT DATA_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/ldm_data/<dataset> (baselines/data/ldm_prepare.py) and
# $CKPT_ROOT/sd21_unet8ch_init.safetensors (baselines/sd21-ft/make_init.py).
# Writes the run to $WORK_DIR/sd21-ft/<dataset>/ and copies checkpoint-40000 to
# $CKPT_ROOT/sd21-ft/<dataset>/diffusion_pytorch_model.safetensors.
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
INIT=$CKPT_ROOT/sd21_unet8ch_init.safetensors
[ -f "$INIT" ] || { echo "missing $INIT (run: python baselines/sd21-ft/make_init.py $INIT)" >&2; exit 1; }
CFG=$WORK_DIR/configs/sd21-ft/$DS.yaml
mkdir -p "$(dirname "$CFG")" "$CKPT_ROOT/sd21-ft/$DS"
envsubst '${DATA_ROOT} ${WORK_DIR} ${CKPT_ROOT}' < "$HERE/configs/$DS.yaml" > "$CFG"
cd "$BASELINES_ROOT/sd21-ft"
$PY main_stage1.py --config "$CFG" --gpu "$GPU"
cp "$WORK_DIR/sd21-ft/$DS/checkpoint-40000/model.safetensors" "$CKPT_ROOT/sd21-ft/$DS/diffusion_pytorch_model.safetensors"
echo "checkpoint: $CKPT_ROOT/sd21-ft/$DS/diffusion_pytorch_model.safetensors"
