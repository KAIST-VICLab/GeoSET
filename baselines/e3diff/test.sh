#!/usr/bin/env bash
# Translate the test SAR images with E3Diff (one-step DDIM, seed 1).
# usage: bash baselines/e3diff/test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>
# env:   BASELINES_ROOT WORK_DIR CKPT_ROOT [GPU=0] [PY=python]
# Loads $CKPT_ROOT/e3diff/<dataset>/gen.pth; writes $WORK_DIR/results/e3diff/<dataset>/<stem>.png.
set -euo pipefail
DS=${1:?usage: test.sh <qxs-saropt|sar2opt|sar2eo|spacenet6>}
: "${BASELINES_ROOT:?}" "${WORK_DIR:?}" "${CKPT_ROOT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
DDPM=$(cd "$HERE/../ddpm-sr3" && pwd)
PY=${PY:-python}
case $DS in
  qxs-saropt|sar2opt|spacenet6) MAIN=$DDPM/e3diff_rgb_main.py ;;
  sar2eo) MAIN=$BASELINES_ROOT/e3diff/main.py ;;
  *) echo "unknown dataset: $DS" >&2; exit 2 ;;
esac
CK=$CKPT_ROOT/e3diff/$DS/gen.pth
D=$WORK_DIR/diff_data/$DS/e3diff
[ -f "$CK" ] || { echo "missing $CK" >&2; exit 1; }
[ -d "$D/val/SAR-PPB" ] || { echo "missing $D (run baselines/data/diff_data.py e3diff $DS)" >&2; exit 1; }
export WORK_DIR CUDA_VISIBLE_DEVICES=${GPU:-0} E3DIFF_REPO=$BASELINES_ROOT/e3diff
export PYTHONPATH=$DDPM/shims PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -c 'import vision_aided_loss.cvmodel' || { echo "vision-aided-loss is not installed (see README)" >&2; exit 1; }
RUN=$WORK_DIR/e3diff/$DS/test
OUT=$WORK_DIR/results/e3diff/$DS
rm -rf "$RUN" "$OUT"
mkdir -p "$RUN" "$(dirname "$OUT")"
ln -s "$CK" "$RUN/model_gen.pth"
envsubst '${WORK_DIR}' < "$HERE/configs/$DS.json" | "$PY" -c '
import json, sys
c = json.load(sys.stdin)
c["phase"] = "val"
c["path"]["resume_state"] = sys.argv[1]
c["datasets"]["val"]["data_len"] = -1
json.dump(c, open(sys.argv[2], "w"), indent=2)' "$RUN/model" "$RUN/test.json"
cd "$RUN"
"$PY" "$MAIN" -c test.json -p val -enable_wandb "" --seed 1
mv "$RUN"/model_S*/sample "$OUT"
N=$(ls "$D/val/EO" | wc -l); M=$(ls "$OUT" | wc -l)
[ "$M" -eq "$N" ] || { echo "expected $N outputs, found $M" >&2; exit 1; }
echo "$M images in $OUT"
