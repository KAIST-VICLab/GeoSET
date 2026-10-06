#!/usr/bin/env bash
# Clone xuekt98/BBDM at the pinned commit into $BASELINES_ROOT/bbdm and apply bbdm.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/xuekt98/BBDM
COMMIT=02c3b13c9f9dfab0853e32123100680a0640c4ed
DEST=$BASELINES_ROOT/bbdm
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/bbdm.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "bbdm.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "bbdm: $DEST at $COMMIT"
