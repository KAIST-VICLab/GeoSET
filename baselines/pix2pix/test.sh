#!/usr/bin/env bash
# Translate the test SAR images with pix2pix; writes $WORK_DIR/results/pix2pix/<dataset>/<stem>_fake_B.png.
# usage: bash baselines/pix2pix/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/pix2pix/<dataset>/net_G.pth (the released file or the output of train.sh).
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt|sar2eo|spacenet6) SRC=combined; SIZE=256 ;;
  sar2opt) SRC=combined_test512; SIZE=512 ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=$WORK_DIR/gan_data/$DS
CK=$CKPT_ROOT/pix2pix/$DS
W=$WORK_DIR/pix2pix/test
OUT=$WORK_DIR/results/pix2pix/$DS
[ -f "$CK/net_G.pth" ] || { echo "missing $CK/net_G.pth" >&2; exit 1; }
[ -d "$D/$SRC/test" ] || { echo "missing $D/$SRC (run baselines/data/gan_prepare.py $DS)" >&2; exit 1; }
rm -rf "$W/ckpt/$DS" "$W/raw/$DS" "$OUT"
mkdir -p "$W/ckpt/$DS" "$OUT"
ln -s "$CK/net_G.pth" "$W/ckpt/$DS/latest_net_G.pth"
cd "$BASELINES_ROOT/pix2pix"
$PY test.py --dataroot "$D/$SRC" --name "$DS" --model pix2pix --direction AtoB \
  --load_size $SIZE --crop_size $SIZE --num_test 99999 --checkpoints_dir "$W/ckpt" --results_dir "$W/raw"
shopt -s nullglob
for f in "$W/raw/$DS/test_latest/images/"*_fake_B.png; do
  mv "$f" "$OUT/"
done
N=$(ls "$D/$SRC/test" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
