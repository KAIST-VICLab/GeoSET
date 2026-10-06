#!/usr/bin/env bash
# Clone NVlabs/SPADE at the pinned commit into $BASELINES_ROOT/spade and apply spade.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/NVlabs/SPADE
COMMIT=fecacc920c1367a038995c45a39c15f6521ca64f
DEST=$BASELINES_ROOT/spade
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/spade.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "spade.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "spade: $DEST at $COMMIT"
