#!/usr/bin/env bash
# Translate the SpaceNet6 test SAR images with Seg-CycleGAN; writes $WORK_DIR/results/seg-cyclegan/spacenet6/<stem>.png.
# usage: bash baselines/seg-cyclegan/test.sh spacenet6
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/seg-cyclegan/spacenet6/net_G_{A,B}.pth (the released files or the output of train.sh);
# the output image is fake_B = G_A(SAR).
set -euo pipefail
DS=${1:?usage: test.sh spacenet6}
[ "$DS" = spacenet6 ] || { echo "Seg-CycleGAN uses the SpaceNet6 building masks: spacenet6 only" >&2; exit 2; }
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=$WORK_DIR/gan_data/spacenet6/AB
CK=$CKPT_ROOT/seg-cyclegan/spacenet6
W=$WORK_DIR/seg-cyclegan/test
OUT=$WORK_DIR/results/seg-cyclegan/spacenet6
for g in G_A G_B; do [ -f "$CK/net_$g.pth" ] || { echo "missing $CK/net_$g.pth" >&2; exit 1; }; done
[ -d "$D/testA" ] || { echo "missing $D (run baselines/data/gan_prepare.py spacenet6)" >&2; exit 1; }
rm -rf "$W/ckpt/spacenet6" "$OUT"
mkdir -p "$W/ckpt/spacenet6" "$OUT"
for g in G_A G_B; do ln -s "$CK/net_$g.pth" "$W/ckpt/spacenet6/latest_net_$g.pth"; done
cd "$BASELINES_ROOT/seg-cyclegan"
SEGCYC_EVAL_DIR=$D/testA SEGCYC_OUT=$OUT $PY sar2opt_eval.py --phase test --name spacenet6 --dataroot "$D" \
  --dataset_mode evaluation --preprocess none --no_flip --epoch latest --model cycle_gan --checkpoints_dir "$W/ckpt"
N=$(ls "$D/testA" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
