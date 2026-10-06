#!/usr/bin/env bash
# Train a ControlNet on Stable Diffusion 2.1-base (SAR condition -> EO) with the settings of the released checkpoints.
# usage: bash baselines/controlnet/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads the imagefolder $WORK_DIR/ldm_data/<dataset>/controlnet (baselines/data/ldm_prepare.py).
# Writes the run to $WORK_DIR/controlnet/<dataset>/ and copies the final ControlNet
# (config.json + diffusion_pytorch_model.safetensors) to $CKPT_ROOT/controlnet/<dataset>/.
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt|sar2eo|spacenet6) RES=256; BS=32 ;;
  sar2opt) RES=512; BS=8 ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
WORK_DIR=$(realpath -m "$WORK_DIR"); CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
export CUDA_VISIBLE_DEVICES=${GPU:-0}
DATA=$WORK_DIR/ldm_data/$DS/controlnet
[ -f "$DATA/train/metadata.jsonl" ] || { echo "missing $DATA (run baselines/data/ldm_prepare.py $DS)" >&2; exit 1; }
RUN=$WORK_DIR/controlnet/$DS
mkdir -p "$RUN" "$CKPT_ROOT/controlnet/$DS"
$PY "$HERE/train_controlnet.py" \
  --pretrained_model_name_or_path Manojb/stable-diffusion-2-1-base \
  --train_data_dir "$DATA" \
  --image_column image \
  --conditioning_image_column conditioning_image \
  --caption_column text \
  --resolution $RES \
  --train_batch_size $BS \
  --dataloader_num_workers 8 \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --max_train_steps 50000 \
  --checkpointing_steps 10000 \
  --checkpoints_total_limit 2 \
  --resume_from_checkpoint latest \
  --seed 42 \
  --output_dir "$RUN"
cp "$RUN/config.json" "$RUN/diffusion_pytorch_model.safetensors" "$CKPT_ROOT/controlnet/$DS/"
echo "checkpoint: $CKPT_ROOT/controlnet/$DS"
