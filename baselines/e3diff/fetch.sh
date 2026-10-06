#!/usr/bin/env bash
# Clone DeepSARRS/E3Diff at the pinned commit into $BASELINES_ROOT/e3diff and apply e3diff.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/DeepSARRS/E3Diff
COMMIT=38601093ab8f8e4b478144621f20890b100a3b74
DEST=$BASELINES_ROOT/e3diff
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/e3diff.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "e3diff.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "e3diff: $DEST at $COMMIT"
