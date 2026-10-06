#!/usr/bin/env bash
# Clone NWPU-LHH/Seg-CycleGAN at the pinned commit into $BASELINES_ROOT/seg-cyclegan and apply seg-cyclegan.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/NWPU-LHH/Seg-CycleGAN
COMMIT=f12ff88ac089b7099443ffa2c34fc47e035def6a
DEST=$BASELINES_ROOT/seg-cyclegan
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/seg-cyclegan.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "seg-cyclegan.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "seg-cyclegan: $DEST at $COMMIT"
