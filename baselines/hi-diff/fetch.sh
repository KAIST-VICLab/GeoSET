#!/usr/bin/env bash
# Clone zhengchen1999/HI-Diff at the pinned commit into $BASELINES_ROOT/hi-diff and apply hi-diff.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/zhengchen1999/HI-Diff
COMMIT=b3bfd167997e27f8edd57681cf70e5031a0e35f2
DEST=$BASELINES_ROOT/hi-diff
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/hi-diff.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "hi-diff.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "hi-diff: $DEST at $COMMIT"
