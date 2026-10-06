#!/usr/bin/env bash
# Translate the test SAR images with SPADE; writes $WORK_DIR/results/spade/<dataset>/<stem>.png.
# usage: bash baselines/spade/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/spade/<dataset>/net_G.pth (the released file or the output of train.sh).
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt|sar2eo|spacenet6) SRC=AB; ARGS="--preprocess_mode resize_and_crop --load_size 256 --crop_size 256 --batchSize 8" ;;
  sar2opt) SRC=test512; ARGS="--preprocess_mode crop --load_size 512 --crop_size 512 --batchSize 4" ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
BASELINES_ROOT=$(realpath -m "$BASELINES_ROOT") WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=$WORK_DIR/gan_data/$DS
CK=$CKPT_ROOT/spade/$DS
W=$WORK_DIR/spade/test
OUT=$WORK_DIR/results/spade/$DS
[ -f "$CK/net_G.pth" ] || { echo "missing $CK/net_G.pth" >&2; exit 1; }
[ -d "$D/$SRC/testA" ] || { echo "missing $D/$SRC (run baselines/data/gan_prepare.py $DS)" >&2; exit 1; }
rm -rf "$W/ckpt/$DS" "$W/raw/$DS" "$OUT"
mkdir -p "$W/ckpt/$DS" "$OUT"
ln -s "$CK/net_G.pth" "$W/ckpt/$DS/latest_net_G.pth"
cd "$BASELINES_ROOT/spade"
$PY test.py --name "$DS" --dataset_mode custom --label_dir "$D/$SRC/testA" --image_dir "$D/$SRC/testB" \
  --label_nc 0 --no_instance --no_pairing_check $ARGS --aspect_ratio 1 --nThreads 8 \
  --checkpoints_dir "$W/ckpt" --results_dir "$W/raw" --which_epoch latest
shopt -s nullglob
for f in "$W/raw/$DS/test_latest/images/synthesized_image/"*.png; do
  mv "$f" "$OUT/"
done
N=$(ls "$D/$SRC/testA" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
