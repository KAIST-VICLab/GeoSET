#!/usr/bin/env bash
# Translate the test SAR images with CondDiff (DDPM sampling respaced to 250 steps).
# usage: bash baselines/conddiff/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/conddiff/<dataset>/ema_final.pt; writes $WORK_DIR/results/conddiff/<dataset>/<stem>.png.
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
PY=${PY:-python}
case $DS in
  qxs-saropt|sar2eo|spacenet6) RES=256; BS=24 ;;
  sar2opt) RES=512; BS=6 ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
REPO=$BASELINES_ROOT/conddiff
CK=$CKPT_ROOT/conddiff/$DS/ema_final.pt
D=$WORK_DIR/diff_data/$DS/conddiff
[ -f "$CK" ] || { echo "missing $CK" >&2; exit 1; }
[ -f "$D/manifest_test.json" ] || { echo "missing $D (run baselines/data/diff_data.py conddiff $DS)" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=$REPO/_stubs:$REPO
RUN=$WORK_DIR/conddiff/$DS/test
OUT=$WORK_DIR/results/conddiff/$DS
rm -rf "$RUN" "$OUT"
mkdir -p "$RUN" "$OUT"
N=$(ls "$D/test/sar" | wc -l)
OPENAI_LOGDIR=$RUN/log "$PY" "$REPO/scripts/image_sample_realtime.py" \
  --model_path "$CK" --test_dir "$D/test" --out_dir "$RUN" \
  --image_size $RES --num_channels 128 --num_res_blocks 3 --learn_sigma False \
  --diffusion_steps 2000 --noise_schedule linear \
  --timestep_respacing 250 --batch_size $BS --num_samples "$N"
"$PY" - "$D/manifest_test.json" "$RUN/gen_opt" "$OUT" <<'PY'
import json, os, sys
manifest, gen, out = sys.argv[1:]
for pid, stem in json.load(open(manifest)).items():
    os.replace(os.path.join(gen, f'{int(pid)}.png'), os.path.join(out, stem + '.png'))
PY
M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
