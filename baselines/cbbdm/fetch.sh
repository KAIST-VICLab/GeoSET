#!/usr/bin/env bash
# Clone egshkim/ConditionalBBDM-for-VHR-SAR-to-Optical at the pinned commit into $BASELINES_ROOT/cbbdm and apply cbbdm.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/egshkim/ConditionalBBDM-for-VHR-SAR-to-Optical
COMMIT=8ce15934f4d4e3f01efe70d11e2d9b9e0859210c
DEST=$BASELINES_ROOT/cbbdm
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/cbbdm.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "cbbdm.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "cbbdm: $DEST at $COMMIT"
