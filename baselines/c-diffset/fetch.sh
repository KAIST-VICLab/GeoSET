#!/usr/bin/env bash
# Clone KAIST-VICLab/C-DiffSET at the pinned commit into $BASELINES_ROOT/c-diffset (no source changes).
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/KAIST-VICLab/C-DiffSET
COMMIT=6fc380215ae97c9d134e8f1434061b652339b45c
DEST=$BASELINES_ROOT/c-diffset
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
echo "c-diffset: $DEST at $COMMIT"
