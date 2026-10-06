#!/usr/bin/env bash
# Clone junyanz/pytorch-CycleGAN-and-pix2pix at the pinned commit into $BASELINES_ROOT/pix2pix and apply pix2pix.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
COMMIT=2a7afba2895d52556dd5dfe07e8555ef657ced6f
DEST=$BASELINES_ROOT/pix2pix
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/pix2pix.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "pix2pix.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "pix2pix: $DEST at $COMMIT"
