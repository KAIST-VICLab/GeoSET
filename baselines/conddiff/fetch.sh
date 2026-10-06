#!/usr/bin/env bash
# Clone Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation at the pinned commit into $BASELINES_ROOT/conddiff and apply conddiff.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation
COMMIT=8326f54adf3a8d717792cb7abc7d3d2b7dfac565
DEST=$BASELINES_ROOT/conddiff
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/conddiff.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "conddiff.patch already applied"
else
  sed -i 's/\r$//' "$DEST/scripts/image_sample_realtime.py"   # upstream file has CRLF line endings
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "conddiff: $DEST at $COMMIT"
