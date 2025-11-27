#!/bin/bash
# Run fast and medium speed Kani proofs locally
# Excludes slow proofs (unwind >= 10)

set -e

cd "$(dirname "$0")/.."

echo "🔍 Running Kani proofs - Fast + Medium tier (unwind <= 9)..."
echo "Note: Slow proofs (unwind >= 10) are excluded"
echo ""

# Get list of fast + medium proofs
FAST_MEDIUM_PROOFS=$(python3 scripts/get_proofs_by_tier.py fast_medium)
PROOF_COUNT=$(echo "$FAST_MEDIUM_PROOFS" | wc -w)

if [ "$PROOF_COUNT" -eq 0 ]; then
    echo "❌ No fast or medium proofs found!"
    exit 1
fi

echo "Found $PROOF_COUNT fast + medium proofs to run"
echo ""

# Build harness filter arguments
HARNESS_ARGS=""
for proof in $FAST_MEDIUM_PROOFS; do
    HARNESS_ARGS="$HARNESS_ARGS --harness $proof"
done

# Run Kani with fast + medium proofs
# Using --jobs 2 to avoid OOM (same as CI)
# Using --output-format terse for cleaner output
# Filtering out "aborting path on assume(false)" noise
echo "Starting Kani verification..."
echo ""

cargo kani \
    --features verify \
    --output-format terse \
    --solver cadical \
    --jobs 2 \
    $HARNESS_ARGS \
    2>&1 | grep -v "aborting path on assume(false)" || {
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -eq 143 ]; then
        echo ""
        echo "⚠️  Kani proofs were terminated (exit code 143 = SIGTERM)"
        echo "This usually means the process was killed due to resource limits"
        exit $EXIT_CODE
    else
        echo ""
        echo "❌ Fast + Medium Kani proofs failed (exit code: $EXIT_CODE)"
        exit $EXIT_CODE
    fi
}

echo ""
echo "✅ Fast + Medium proofs verified successfully!"
echo "Total proofs verified: $PROOF_COUNT"


