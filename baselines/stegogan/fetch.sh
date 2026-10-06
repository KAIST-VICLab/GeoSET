#!/usr/bin/env bash
# Clone sian-wusidi/StegoGAN at the pinned commit into $BASELINES_ROOT/stegogan (no source changes).
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/sian-wusidi/StegoGAN
COMMIT=cad61997c0f82793444f60f81298142b80cdf3c1
DEST=$BASELINES_ROOT/stegogan
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
echo "stegogan: $DEST at $COMMIT"
