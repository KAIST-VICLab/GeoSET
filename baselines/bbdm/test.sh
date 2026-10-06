#!/usr/bin/env bash
# Translate the test SAR images with BBDM; writes $WORK_DIR/results/bbdm/<dataset>/<stem>.png.
# usage: bash baselines/bbdm/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/bbdm/<dataset>/last_model.pth (the released file or the output of train.sh) and
# $CKPT_ROOT/vq-f4/model.ckpt. Sampling: EMA weights, 200 Brownian-bridge steps, seed 1234.
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in qxs-saropt|sar2opt|sar2eo|spacenet6) ;; *) echo "unknown dataset: $DS" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export WORK_DIR=$(realpath -m "$WORK_DIR") CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
export CUDA_VISIBLE_DEVICES=${GPU:-0}
CKPT=$CKPT_ROOT/bbdm/$DS/last_model.pth
[ -f "$CKPT" ] || { echo "missing $CKPT" >&2; exit 1; }
[ -d "$WORK_DIR/ldm_data/$DS/bbdm" ] || { echo "missing $WORK_DIR/ldm_data/$DS (run baselines/data/ldm_prepare.py $DS)" >&2; exit 1; }
CFG=$WORK_DIR/configs/bbdm/$DS.yaml
SAMPLES=$WORK_DIR/bbdm/$DS/LBBDM-f4/sample_to_eval
OUT=$WORK_DIR/results/bbdm/$DS
mkdir -p "$(dirname "$CFG")" "$(dirname "$OUT")"
envsubst '${WORK_DIR} ${CKPT_ROOT}' < "$HERE/configs/$DS.yaml" > "$CFG"
rm -rf "$SAMPLES" "$OUT"
cd "$BASELINES_ROOT/bbdm"
$PY main.py -c "$CFG" --gpu_ids 0 -r "$WORK_DIR/bbdm" --sample_to_eval --resume_model "$CKPT"
mv "$SAMPLES/200" "$OUT"
N=$(ls "$WORK_DIR/ldm_data/$DS/bbdm/test/A/" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
