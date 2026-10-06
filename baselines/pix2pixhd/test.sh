#!/usr/bin/env bash
# Translate the test SAR images with pix2pixHD; writes $WORK_DIR/results/pix2pixhd/<dataset>/<stem>.png.
# usage: bash baselines/pix2pixhd/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/pix2pixhd/<dataset>/net_G.pth (the released file or the output of train.sh).
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt)       ARGS="--resize_or_crop resize_and_crop --loadSize 256 --fineSize 256 --how_many 4000" ;;
  sar2opt)          ARGS="--resize_or_crop crop --loadSize 512 --fineSize 512 --how_many 99999" ;;
  sar2eo|spacenet6) ARGS="--resize_or_crop resize_and_crop --loadSize 256 --fineSize 256 --how_many 99999" ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=$WORK_DIR/gan_data/$DS
CK=$CKPT_ROOT/pix2pixhd/$DS
W=$WORK_DIR/pix2pixhd/test
OUT=$WORK_DIR/results/pix2pixhd/$DS
[ -f "$CK/net_G.pth" ] || { echo "missing $CK/net_G.pth" >&2; exit 1; }
[ -d "$D/p2phd/test_A" ] || { echo "missing $D/p2phd (run baselines/data/gan_prepare.py $DS)" >&2; exit 1; }
rm -rf "$W/ckpt/$DS" "$W/raw/$DS" "$OUT"
mkdir -p "$W/ckpt/$DS" "$OUT"
ln -s "$CK/net_G.pth" "$W/ckpt/$DS/latest_net_G.pth"
cd "$BASELINES_ROOT/pix2pixhd"
$PY test.py --name "$DS" --dataroot "$D/p2phd" --label_nc 0 --no_instance $ARGS \
  --checkpoints_dir "$W/ckpt" --results_dir "$W/raw" --phase test --which_epoch latest
shopt -s nullglob
for f in "$W/raw/$DS/test_latest/images/"*_synthesized_image.png; do
  mv "$f" "$OUT/$(basename "$f" _synthesized_image.png).png"
done
N=$(ls "$D/p2phd/test_A/" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
