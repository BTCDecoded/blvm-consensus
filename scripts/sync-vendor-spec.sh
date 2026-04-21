#!/bin/bash
# Refresh blvm-consensus/blvm-spec from sibling ../blvm-spec (monorepo) before release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/../blvm-spec"
DST="${ROOT}/blvm-spec"
if [[ ! -f "${SRC}/PROTOCOL.md" ]]; then
  echo "Expected ${SRC}/PROTOCOL.md — clone or place blvm-spec next to this repo."
  exit 1
fi
mkdir -p "${DST}"
cp "${SRC}/PROTOCOL.md" "${SRC}/ARCHITECTURE.md" "${DST}/"
echo "Updated ${DST} from ${SRC}"
