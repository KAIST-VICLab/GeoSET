#!/usr/bin/env bash
# Fetch the upstream code of every comparison method (or of the methods named) into $BASELINES_ROOT:
# each <method>/fetch.sh clones the upstream repository at its pinned commit and applies the method's patch.
#   bash baselines/setup.sh                 # all methods
#   bash baselines/setup.sh bbdm e3diff     # selected methods
set -euo pipefail
: "${BASELINES_ROOT:?set BASELINES_ROOT}"
HERE=$(cd "$(dirname "$0")" && pwd)
if [ $# -eq 0 ]; then
  set -- $(for f in "$HERE"/*/fetch.sh; do basename "$(dirname "$f")"; done)
fi
mkdir -p "$BASELINES_ROOT"
for m in "$@"; do
  if [ -f "$HERE/$m/fetch.sh" ]; then
    bash "$HERE/$m/fetch.sh"
  elif [ -d "$HERE/$m" ] && [ "$m" != data ]; then
    echo "$m: nothing to fetch (scripts run from baselines/$m)"
  else
    echo "unknown method: $m" >&2
    exit 2
  fi
done
