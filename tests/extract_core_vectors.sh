#!/bin/bash
# Extract Bitcoin Core test vectors for integration
#
# This script extracts test vectors from Bitcoin Core repository
# and prepares them for use in consensus-proof tests.

set -e

BITCOIN_CORE_PATH="${BITCOIN_CORE_PATH:-/home/user/src/bitcoin}"
TEST_DATA_DIR="tests/test_data/core_vectors"

echo "Extracting Bitcoin Core test vectors from: $BITCOIN_CORE_PATH"

# Create test data directories
mkdir -p "$TEST_DATA_DIR/transactions"
mkdir -p "$TEST_DATA_DIR/scripts"
mkdir -p "$TEST_DATA_DIR/blocks"

# Extract transaction test vectors
if [ -f "$BITCOIN_CORE_PATH/src/test/data/tx_valid.json" ]; then
    echo "Copying tx_valid.json..."
    cp "$BITCOIN_CORE_PATH/src/test/data/tx_valid.json" "$TEST_DATA_DIR/transactions/"
    echo "✓ Copied tx_valid.json"
else
    echo "⚠ tx_valid.json not found"
fi

if [ -f "$BITCOIN_CORE_PATH/src/test/data/tx_invalid.json" ]; then
    echo "Copying tx_invalid.json..."
    cp "$BITCOIN_CORE_PATH/src/test/data/tx_invalid.json" "$TEST_DATA_DIR/transactions/"
    echo "✓ Copied tx_invalid.json"
else
    echo "⚠ tx_invalid.json not found"
fi

# Extract script test vectors
if [ -f "$BITCOIN_CORE_PATH/src/test/data/script_tests.json" ]; then
    echo "Copying script_tests.json..."
    cp "$BITCOIN_CORE_PATH/src/test/data/script_tests.json" "$TEST_DATA_DIR/scripts/"
    echo "✓ Copied script_tests.json"
else
    echo "⚠ script_tests.json not found"
fi

# Note: Core doesn't have separate script_valid.json/script_invalid.json
# script_tests.json contains both valid and invalid test cases

# Extract block test vectors (if available)
# Core doesn't have standard block_valid.json/block_invalid.json files
# but we can extract from test fixtures
if [ -d "$BITCOIN_CORE_PATH/src/test/data" ]; then
    echo "Checking for block test data..."
    # Core uses C++ test fixtures, not JSON for blocks
    # We'll need to convert or use different approach
    echo "⚠ Block test vectors need to be extracted from C++ test fixtures"
fi

echo ""
echo "Test vector extraction complete!"
echo ""
echo "To run tests with Core vectors:"
echo "  cargo test --test core_test_vectors::integration_test"
echo ""
echo "Note: Some test vectors may require additional parsing logic"
echo "for Core's specific JSON format."



