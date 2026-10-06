#!/usr/bin/env bash
# Clone junyanz/pytorch-CycleGAN-and-pix2pix at the pinned commit into $BASELINES_ROOT/cyclegan and apply cyclegan.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
COMMIT=2a7afba2895d52556dd5dfe07e8555ef657ced6f
DEST=$BASELINES_ROOT/cyclegan
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/cyclegan.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "cyclegan.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "cyclegan: $DEST at $COMMIT"
