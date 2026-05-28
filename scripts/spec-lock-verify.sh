#!/usr/bin/env bash
# Run blvm-spec-lock verify locally — same logic as the CI Verify step but without
# the attestation / artifact uploads.
#
# Usage (from blvm-consensus root):
#   ./scripts/spec-lock-verify.sh            # verify only
#   ./scripts/spec-lock-verify.sh --drift    # check-drift then verify
#   ./scripts/spec-lock-verify.sh --install  # (re-)install blvm-spec-lock, then verify
#   ./scripts/spec-lock-verify.sh --build    # build spec-lock from local ../blvm-spec-lock source, then verify
#
# The script automatically locates the Orange Paper spec (../blvm-spec or the BLVM_SPEC_PATH
# env var) and uses the toolchain version from rust-toolchain.toml.
#
# Exit codes:
#   0 — verify passed (PARTIAL results are NOT failures; only FAILED counts)
#   1 — verify failed or required tool not found
#   2 — spec path not found

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Options ─────────────────────────────────────────────────────────────────
DO_DRIFT=0
DO_INSTALL=0
DO_BUILD=0
TIMEOUT="${SPEC_LOCK_Z3_TIMEOUT_SECS:-60}"   # seconds; CI uses 120, local default 60

for arg in "$@"; do
  case "$arg" in
    --drift)   DO_DRIFT=1 ;;
    --install) DO_INSTALL=1 ;;
    --build)   DO_BUILD=1 ;;
    --timeout=*) TIMEOUT="${arg#*=}" ;;
    -h|--help)
      sed -n '2,25p' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# ── Locate spec ──────────────────────────────────────────────────────────────
if [ -n "${BLVM_SPEC_PATH:-}" ]; then
  SPEC_DIR="$BLVM_SPEC_PATH"
elif [ -f "$CRATE_DIR/../blvm-spec/PROTOCOL.md" ]; then
  SPEC_DIR="$CRATE_DIR/../blvm-spec"
elif [ -f "$CRATE_DIR/../blvm-spec/THE_ORANGE_PAPER.md" ]; then
  SPEC_DIR="$CRATE_DIR/../blvm-spec"
else
  echo "❌ Could not find blvm-spec alongside blvm-consensus."
  echo "   Clone it or set BLVM_SPEC_PATH=/path/to/blvm-spec"
  exit 2
fi

if [ -f "$SPEC_DIR/PROTOCOL.md" ] && [ -f "$SPEC_DIR/ARCHITECTURE.md" ]; then
  SPEC_ARGS="--crate-path $CRATE_DIR --spec-path $SPEC_DIR/PROTOCOL.md $SPEC_DIR/ARCHITECTURE.md"
elif [ -f "$SPEC_DIR/THE_ORANGE_PAPER.md" ]; then
  SPEC_ARGS="--crate-path $CRATE_DIR --spec-path $SPEC_DIR/THE_ORANGE_PAPER.md"
else
  echo "❌ blvm-spec directory found but no recognised spec file."
  exit 2
fi

# ── Determine toolchain channel ───────────────────────────────────────────────
CHANNEL=$(grep -E '^channel\s*=' "$CRATE_DIR/rust-toolchain.toml" 2>/dev/null \
          | head -1 | sed -E 's/.*=\s*"([^"]+)".*/\1/')
if [ -z "$CHANNEL" ]; then
  echo "⚠️  Could not read channel from rust-toolchain.toml; using active toolchain"
  CARGO_CMD="cargo"
else
  CARGO_CMD="rustup run $CHANNEL cargo"
fi

# ── Install or build blvm-spec-lock ──────────────────────────────────────────
if [ "$DO_BUILD" -eq 1 ]; then
  SPEC_LOCK_SRC="${BLVM_SPEC_LOCK_PATH:-$CRATE_DIR/../blvm-spec-lock}"
  if [ ! -f "$SPEC_LOCK_SRC/Cargo.toml" ]; then
    echo "❌ blvm-spec-lock source not found at $SPEC_LOCK_SRC"
    echo "   Clone it alongside blvm-consensus or set BLVM_SPEC_LOCK_PATH"
    exit 1
  fi
  echo "🔨 Building blvm-spec-lock from source ($SPEC_LOCK_SRC)..."
  (cd "$SPEC_LOCK_SRC" && $CARGO_CMD install --path . --features z3 --locked --force)
elif [ "$DO_INSTALL" -eq 1 ] || ! command -v cargo-spec-lock >/dev/null 2>&1; then
  echo "📦 Installing blvm-spec-lock from crates.io..."
  $CARGO_CMD install blvm-spec-lock --version '>=0.1, <1' --locked --features z3
fi

if ! command -v cargo-spec-lock >/dev/null 2>&1; then
  echo "❌ cargo-spec-lock not found after install attempt"
  exit 1
fi

INSTALLED_VERSION=$(cargo-spec-lock --version 2>/dev/null || echo "unknown")
echo "🔧 Using: $INSTALLED_VERSION"
echo "📁 Crate: $CRATE_DIR"
echo "📚 Spec:  $SPEC_DIR"
echo ""

# ── check-drift (optional) ────────────────────────────────────────────────────
if [ "$DO_DRIFT" -eq 1 ]; then
  echo "📎 Running blvm-spec-lock check-drift..."
  DRIFT_ARGS="$SPEC_ARGS --scoped-unparseables"
  if cargo-spec-lock check-drift --help 2>&1 | grep -Fq -- '--scoped-formulas'; then
    DRIFT_ARGS="$DRIFT_ARGS --scoped-formulas"
  fi
  # shellcheck disable=SC2086
  cargo-spec-lock check-drift $DRIFT_ARGS
  echo "✅ check-drift passed"
  echo ""
fi

# ── verify ────────────────────────────────────────────────────────────────────
echo "🔒 Running blvm-spec-lock verify (timeout: ${TIMEOUT}s)..."
export SPEC_LOCK_STRICT=1
export SPEC_LOCK_Z3_TIMEOUT_SECS="$TIMEOUT"

# shellcheck disable=SC2086
cargo-spec-lock verify $SPEC_ARGS --timeout "$TIMEOUT" --format human
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✅ blvm-spec-lock verify passed"
else
  echo ""
  echo "❌ blvm-spec-lock verify FAILED (exit $EXIT_CODE)"
  echo "   Run with --build to test against a local blvm-spec-lock branch."
fi

exit $EXIT_CODE
