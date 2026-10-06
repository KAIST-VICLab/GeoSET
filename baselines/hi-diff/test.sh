#!/usr/bin/env bash
# Translate the test SAR images with HI-Diff (stage-2 model; seed 100; images processed in the order of
# test_order/<dataset>.txt, the order that produced the released outputs).
# usage: bash baselines/hi-diff/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/hi-diff/<dataset>/S2_{net_g,net_le_dm,net_d}_latest.pth;
# writes $WORK_DIR/results/hi-diff/<dataset>/<stem>.png.
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
case $DS in
  qxs-saropt|sar2opt|sar2eo|spacenet6) ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
for f in S2_net_g S2_net_le_dm S2_net_d; do
  [ -f "$CKPT_ROOT/hi-diff/$DS/${f}_latest.pth" ] || { echo "missing $CKPT_ROOT/hi-diff/$DS/${f}_latest.pth" >&2; exit 1; }
done
D=$WORK_DIR/diff_data/$DS/paired
[ -d "$D/testA" ] || { echo "missing $D (run baselines/data/diff_data.py paired $DS)" >&2; exit 1; }
export WORK_DIR CKPT_ROOT TEST_ORDER=$HERE/test_order/$DS.txt
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=$WORK_DIR/hi-diff/$DS
OUT=$WORK_DIR/results/hi-diff/$DS
mkdir -p "$RUN" "$(dirname "$OUT")"
rm -rf "$OUT" "$BASELINES_ROOT/hi-diff/results/test_HI_Diff_$DS"
envsubst '${WORK_DIR} ${CKPT_ROOT} ${TEST_ORDER}' < "$HERE/configs/${DS}_test.yml" > "$RUN/test.yml"
cd "$BASELINES_ROOT/hi-diff"
"$PY" test.py -opt "$RUN/test.yml"
mv "results/test_HI_Diff_$DS/visualization/$DS" "$OUT"
N=$(wc -l < "$TEST_ORDER"); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
