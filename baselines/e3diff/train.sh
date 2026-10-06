#!/usr/bin/env bash
# Train E3Diff stage 2 (one-step generator) on one dataset: resumes the DDPM (SR3) stage-1 weights at
# iteration 250,000 and trains to iteration 310,000 (60,000 updates).
# usage: bash baselines/e3diff/train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Reads $CKPT_ROOT/ddpm-sr3/<dataset>/gen.pth and $WORK_DIR/diff_data/<dataset>/e3diff;
# writes $CKPT_ROOT/e3diff/<dataset>/gen.pth. Needs the vision-aided-loss package (README).
set -euo pipefail
DS=${1:?usage: train.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
DDPM=$(cd "$HERE/../ddpm-sr3" && pwd)
PY=${PY:-python}
# EPOCH = the stage-1 epoch counter at iteration 250,000 (E3Diff reads only iter/epoch from *_opt.pth)
case $DS in
  qxs-saropt) EPOCH=250;  MAIN=$DDPM/e3diff_rgb_main.py ;;
  sar2opt)    EPOCH=689;  MAIN=$DDPM/e3diff_rgb_main.py ;;
  sar2eo)     EPOCH=59;   MAIN=$BASELINES_ROOT/e3diff/main.py ;;
  spacenet6)  EPOCH=1563; MAIN=$DDPM/e3diff_rgb_main.py ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
S1=$CKPT_ROOT/ddpm-sr3/$DS/gen.pth
D=$WORK_DIR/diff_data/$DS/e3diff
[ -f "$S1" ] || { echo "missing $S1 (run baselines/ddpm-sr3/train.sh $DS or download it)" >&2; exit 1; }
[ -d "$D/train/SAR-PPB" ] || { echo "missing $D (run baselines/data/diff_data.py e3diff $DS)" >&2; exit 1; }
export WORK_DIR CUDA_VISIBLE_DEVICES=${GPU:-0} E3DIFF_REPO=$BASELINES_ROOT/e3diff
export PYTHONPATH=$DDPM/shims PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -c 'import vision_aided_loss.cvmodel' || { echo "vision-aided-loss is not installed (see README)" >&2; exit 1; }
RUN=$WORK_DIR/e3diff/$DS
mkdir -p "$RUN/stage1"
ln -sfn "$S1" "$RUN/stage1/I250000_gen.pth"
"$PY" -c 'import sys, torch; torch.save({"epoch": int(sys.argv[1]), "iter": 250000, "scheduler": None, "optimizer": None}, sys.argv[2])' \
  "$EPOCH" "$RUN/stage1/I250000_opt.pth"
envsubst '${WORK_DIR}' < "$HERE/configs/$DS.json" > "$RUN/train.json"
cd "$RUN"
"$PY" "$MAIN" -c train.json -p train -enable_wandb "" --seed 1
CKPT=$(ls -t "$RUN"/experiments/e3diff_"$DS"_*/checkpoint/I310000_E*_gen.pth | head -1)
mkdir -p "$CKPT_ROOT/e3diff/$DS"
cp "$CKPT" "$CKPT_ROOT/e3diff/$DS/gen.pth"
echo "$CKPT -> $CKPT_ROOT/e3diff/$DS/gen.pth"
