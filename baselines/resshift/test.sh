#!/usr/bin/env bash
# Translate the test SAR images with ResShift (4 sampling steps, seed 12345).
# usage: bash baselines/resshift/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/resshift/<dataset>/ema_model.pth; writes $WORK_DIR/results/resshift/<dataset>/<stem>.png.
# SAR2Opt 512x512 inputs are processed as four non-overlapping 256x256 tiles.
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
D=$WORK_DIR/diff_data/$DS/paired
case $DS in
  qxs-saropt|spacenet6) IN=$D/testA; BS=16 ;;
  sar2opt) IN=$D/testA; BS=8 ;;
  sar2eo) IN=$D/test_first4000A; BS=16 ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
CK=$CKPT_ROOT/resshift/$DS/ema_model.pth
[ -f "$CK" ] || { echo "missing $CK" >&2; exit 1; }
[ -d "$IN" ] || { echo "missing $IN (run baselines/data/diff_data.py paired $DS)" >&2; exit 1; }
export WORK_DIR BASELINES_ROOT CUDA_VISIBLE_DEVICES=${GPU:-0}
RUN=$WORK_DIR/resshift/$DS
OUT=$WORK_DIR/results/resshift/$DS
mkdir -p "$RUN"
rm -rf "$OUT"
envsubst '${WORK_DIR} ${BASELINES_ROOT}' < "$HERE/configs/$DS.yaml" > "$RUN/config.yaml"
cd "$BASELINES_ROOT/resshift"
"$PY" inference_sar2eo.py --cfg_path "$RUN/config.yaml" --ckpt "$CK" --in_dir "$IN" --out_dir "$OUT" --bs $BS
N=$(ls "$IN" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
