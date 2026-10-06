#!/usr/bin/env bash
# Translate the test SAR images with StegoGAN; writes $WORK_DIR/results/stegogan/<dataset>/<stem>.png.
# usage: bash baselines/stegogan/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/stegogan/<dataset>/net_G_{A,B}.pth (the released files or the output of train.sh).
# The output image is fake_B_clean = G_A(SAR), computed from the SAR image alone.
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt|sar2eo|spacenet6) SRC=AB; SIZE=256 ;;
  sar2opt) SRC=test512; SIZE=512 ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=$WORK_DIR/gan_data/$DS
CK=$CKPT_ROOT/stegogan/$DS
W=$WORK_DIR/stegogan/test
OUT=$WORK_DIR/results/stegogan/$DS
for g in G_A G_B; do [ -f "$CK/net_$g.pth" ] || { echo "missing $CK/net_$g.pth" >&2; exit 1; }; done
[ -d "$D/$SRC/testA" ] || { echo "missing $D/$SRC (run baselines/data/gan_prepare.py $DS)" >&2; exit 1; }
rm -rf "$W/ckpt/$DS" "$W/raw/$DS" "$OUT"
mkdir -p "$W/ckpt/$DS" "$OUT"
for g in G_A G_B; do ln -s "$CK/net_$g.pth" "$W/ckpt/$DS/latest_net_$g.pth"; done
cd "$BASELINES_ROOT/stegogan"
$PY test.py --dataroot "$D/$SRC" --name "$DS" --model stego_gan --gpu_ids 0 --phase test --no_dropout \
  --resnet_layer 8 --fusionblock --load_size $SIZE --crop_size $SIZE --num_test 99999 \
  --checkpoints_dir "$W/ckpt" --results_dir "$W/raw"
shopt -s nullglob
for f in "$W/raw/$DS/test_latest/images/fake_B_clean/"*.png; do
  mv "$f" "$OUT/"
done
N=$(ls "$D/$SRC/testA" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
