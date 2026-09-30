#!/usr/bin/env bash
# Fetch golden-vector era blocks (gitignored .bin files).
#
#   481824 — SegWit activation
#   709632 — Taproot activation
#
# Usage:
#   ./scripts/fetch_golden_block_481824.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$PROJECT_ROOT/tests/test_data/golden_vectors"

mkdir -p "$OUT_DIR"

fetch_block() {
  local height="$1"
  local hash="$2"
  local output="$OUT_DIR/block_${height}.bin"
  local url="https://blockstream.info/api/block/${hash}/raw"

  if [[ -f "$output" ]]; then
    echo "✅ block $height fixture already present: $output"
    return 0
  fi

  echo "Downloading block $height from Blockstream..."
  curl -fsSL "$url" -o "$output.tmp"
  mv "$output.tmp" "$output"
  echo "✅ Saved $output ($(wc -c < "$output") bytes)"
}

fetch_block 481824 "0000000000000000001c8018d9cb3b742ef25114f27563e3fc4a1902167f9893"
fetch_block 709632 "0000000000000000000687bca986194dc2c1f949318629b44bb54ec0a94d8244"
