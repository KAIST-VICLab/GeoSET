#!/usr/bin/env bash
# Clone NVIDIA/pix2pixHD at the pinned commit into $BASELINES_ROOT/pix2pixhd and apply pix2pixhd.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/NVIDIA/pix2pixHD
COMMIT=14b3b3c7fff413086e3b58df52096f16b6891172
DEST=$BASELINES_ROOT/pix2pixhd
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/pix2pixhd.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "pix2pixhd.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "pix2pixhd: $DEST at $COMMIT"
