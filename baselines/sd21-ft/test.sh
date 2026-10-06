#!/usr/bin/env bash
# Translate the test SAR images with SD2.1 FT; writes $WORK_DIR/results/sd21-ft/<dataset>/<stem>.png.
# usage: bash baselines/sd21-ft/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/sd21-ft/<dataset>/diffusion_pytorch_model.safetensors (the released file or the output
# of train.sh). Sampling: DDIM, 50 steps, fp32, one image at a time.
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in qxs-saropt|sar2opt|sar2eo|spacenet6) ;; *) echo "unknown dataset: $DS" >&2; exit 2 ;; esac
PY=${PY:-python}
WORK_DIR=$(realpath -m "$WORK_DIR"); CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
export CUDA_VISIBLE_DEVICES=${GPU:-0}
CKPT=$CKPT_ROOT/sd21-ft/$DS/diffusion_pytorch_model.safetensors
SAR=$WORK_DIR/ldm_data/$DS/test_sar
OUT=$WORK_DIR/results/sd21-ft/$DS
[ -f "$CKPT" ] || { echo "missing $CKPT" >&2; exit 1; }
[ -d "$SAR" ] || { echo "missing $SAR (run baselines/data/ldm_prepare.py $DS)" >&2; exit 1; }
rm -rf "$OUT"
cd "$BASELINES_ROOT/sd21-ft"
$PY test_stage1.py --sar-dir "$SAR" --output-dir "$OUT" --checkpoint "$CKPT"
N=$(ls "$SAR/" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
