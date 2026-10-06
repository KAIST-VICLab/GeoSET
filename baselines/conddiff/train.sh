#!/usr/bin/env bash
# Train CondDiff for 50,000 updates on one dataset (resumes from the newest model*.pt in the run dir).
# usage: bash baselines/conddiff/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $WORK_DIR/diff_data/<dataset>/conddiff (baselines/data/diff_data.py conddiff <dataset>);
# writes $CKPT_ROOT/conddiff/<dataset>/ema_final.pt (the EMA 0.9999 weights at step 50,000).
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
PY=${PY:-python}
case $DS in
  qxs-saropt|sar2eo|spacenet6) RES=256; BS=24 ;;
  sar2opt) RES=512; BS=6 ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
REPO=$BASELINES_ROOT/conddiff
D=$WORK_DIR/diff_data/$DS/conddiff
[ -d "$D/train/sar" ] || { echo "missing $D (run baselines/data/diff_data.py conddiff $DS)" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=$REPO/_stubs:$REPO
LOG=$WORK_DIR/conddiff/$DS
mkdir -p "$LOG"
RESUME=$(ls "$LOG"/model*.pt 2>/dev/null | sort | tail -1 || true)
OPENAI_LOGDIR=$LOG "$PY" "$REPO/scripts/image_train.py" \
  --data_dir_sar "$D/train/sar" --data_dir_opt "$D/train/opt" \
  --image_size $RES --num_channels 128 --num_res_blocks 3 --learn_sigma False \
  --diffusion_steps 2000 --noise_schedule linear \
  --lr 1e-4 --batch_size $BS --lr_anneal_steps 50000 --save_interval 10000 \
  ${RESUME:+--resume_checkpoint "$RESUME"}
mkdir -p "$CKPT_ROOT/conddiff/$DS"
cp "$LOG/ema_0.9999_050000.pt" "$CKPT_ROOT/conddiff/$DS/ema_final.pt"
echo "$LOG/ema_0.9999_050000.pt -> $CKPT_ROOT/conddiff/$DS/ema_final.pt"
