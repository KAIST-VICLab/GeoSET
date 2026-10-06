#!/usr/bin/env bash
# Clone zsyOAOA/ResShift at the pinned commit into $BASELINES_ROOT/resshift, apply resshift.patch and
# download the frozen VQ-f4 autoencoder of the ResShift v2.0 release into weights/.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/zsyOAOA/ResShift
COMMIT=bb03b7d21614cace01787e097c8a6ab6b945227d
DEST=$BASELINES_ROOT/resshift
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/resshift.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "resshift.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
VQ=$DEST/weights/autoencoder_vq_f4.pth
if [ ! -f "$VQ" ]; then
  mkdir -p "$DEST/weights"
  curl -L --fail -o "$VQ.part" https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth
  mv "$VQ.part" "$VQ"
fi
echo "e967eaa2c380152811c53e312990d6967a8a957c4f7590f19490ec72110b2889  $VQ" | sha256sum -c -
echo "resshift: $DEST at $COMMIT"
