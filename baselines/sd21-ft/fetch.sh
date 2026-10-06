#!/usr/bin/env bash
# Clone KAIST-VICLab/C-DiffSET at the pinned commit into $BASELINES_ROOT/sd21-ft and apply sd21-ft.patch.
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
URL=https://github.com/KAIST-VICLab/C-DiffSET
COMMIT=6fc380215ae97c9d134e8f1434061b652339b45c
DEST=$BASELINES_ROOT/sd21-ft
[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH=$(cd "$(dirname "$0")" && pwd)/sd21-ft.patch
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "sd21-ft.patch already applied"
else
  git -C "$DEST" apply --whitespace=nowarn "$PATCH"
fi
echo "sd21-ft: $DEST at $COMMIT"
