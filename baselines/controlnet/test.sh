#!/usr/bin/env bash
# Translate the test SAR images with ControlNet; writes $WORK_DIR/results/controlnet/<dataset>/<stem>.png.
# usage: bash baselines/controlnet/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/controlnet/<dataset>/ (config.json + diffusion_pytorch_model.safetensors).
# Sampling: UniPC, 50 steps, guidance 7.5, bf16, one generator seeded 42 consumed batch by batch
# (batch 32 at 256 px, 8 at 512 px).
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
case $DS in
  qxs-saropt|sar2eo|spacenet6) RES=256; BS=32 ;;
  sar2opt) RES=512; BS=8 ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
WORK_DIR=$(realpath -m "$WORK_DIR"); CKPT_ROOT=$(realpath -m "$CKPT_ROOT")
export CUDA_VISIBLE_DEVICES=${GPU:-0}
CKPT=$CKPT_ROOT/controlnet/$DS
SAR=$WORK_DIR/ldm_data/$DS/test_sar
OUT=$WORK_DIR/results/controlnet/$DS
[ -f "$CKPT/diffusion_pytorch_model.safetensors" ] || { echo "missing $CKPT/diffusion_pytorch_model.safetensors" >&2; exit 1; }
[ -d "$SAR" ] || { echo "missing $SAR (run baselines/data/ldm_prepare.py $DS)" >&2; exit 1; }
rm -rf "$OUT"
$PY "$HERE/infer_controlnet.py" --controlnet_dir "$CKPT" --sar_dir "$SAR" --out_dir "$OUT" \
  --steps 50 --batch_size $BS --resolution $RES --seed 42
N=$(ls "$SAR/" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
