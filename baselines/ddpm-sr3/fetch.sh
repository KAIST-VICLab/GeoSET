#!/usr/bin/env bash
# Clone DeepSARRS/E3Diff at the pinned commit into $BASELINES_ROOT/ddpm-sr3 and apply ddpm-sr3.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/DeepSARRS/E3Diff
COMMIT=38601093ab8f8e4b478144621f20890b100a3b74
DEST=$BASELINES_ROOT/ddpm-sr3
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/ddpm-sr3.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "ddpm-sr3.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "ddpm-sr3: $DEST at $COMMIT"
