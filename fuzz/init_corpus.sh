#!/bin/bash
# Initialize corpus directories with seed inputs
# This script creates initial corpus seeds from test vectors and real blockchain data

set -e

FUZZ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$FUZZ_DIR/.."

CORPUS_DIR="${1:-fuzz/corpus}"
mkdir -p "$CORPUS_DIR"

echo "Initializing corpus directories for all fuzz targets..."

# Create corpus directories for all targets
TARGETS=(
    "transaction_validation"
    "block_validation"
    "script_execution"
    "segwit_validation"
    "mempool_operations"
    "utxo_commitments"
    "compact_block_reconstruction"
    "pow_validation"
    "economic_validation"
    "serialization"
    "script_opcodes"
    "differential_fuzzing",
    "sigop_validation"
)

for target in "${TARGETS[@]}"; do
    mkdir -p "$CORPUS_DIR/$target"
    echo "Created corpus directory: $CORPUS_DIR/$target"
done

# Function to add a seed file
add_seed() {
    local target=$1
    local filename=$2
    local content=$3
    
    echo -n "$content" > "$CORPUS_DIR/$target/$filename"
    echo "Added seed: $target/$filename"
}

echo ""
echo "Adding basic seed inputs..."

# Transaction validation seeds
# Minimal valid transaction
add_seed "transaction_validation" "minimal_valid.hex" "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0401010101ffffffff0100f90295000000001976a914000000000000000000000000000000000000000088ac00000000"

# Block validation seeds
# Minimal block header (80 bytes)
add_seed "block_validation" "minimal_header.hex" "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c"

# Script execution seeds
# OP_1 (simple script)
add_seed "script_execution" "op_1.hex" "51"
# OP_DUP OP_HASH160
add_seed "script_execution" "p2pkh_pattern.hex" "76a914"
# OP_HASH256
add_seed "script_execution" "hash256.hex" "aa"

# Serialization seeds
# VarInt encoding examples
add_seed "serialization" "varint_0.hex" "00"
add_seed "serialization" "varint_1.hex" "01"
add_seed "serialization" "varint_127.hex" "7f"
add_seed "serialization" "varint_128.hex" "8001"

# Pow validation seeds
# Genesis block header
add_seed "pow_validation" "genesis_header.hex" "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c"

# Economic validation seeds
# Height 0 (genesis)
add_seed "economic_validation" "height_0.hex" "0000000000000000"
# Height at first halving
add_seed "economic_validation" "height_210000.hex" "a086010000000000"

# sigop_validation seeds
# ------------------------------------------------------------------
# CORE OPCODE BEHAVIOR
# -----------------------------
add_seed "sigop_validation" "checksig.hex" "ac"
add_seed "sigop_validation" "multi_checksig.hex" "acacac"
add_seed "sigop_validation" "checksigverify.hex" "ad"

# Mixed basic opcodes
add_seed "sigop_validation" "core_mixed.hex" "acadacacad"

# -----------------------------
# MULTISIG BEHAVIOR
# -----------------------------
add_seed "sigop_validation" "multisig_accurate.hex" "52ae"
add_seed "sigop_validation" "multisig_max.hex" "60ae"
add_seed "sigop_validation" "multisig_invalid_prefix.hex" "faae"
add_seed "sigop_validation" "multisig_no_prefix.hex" "ae"

# Repeated multisig stress (survives slicing well)
add_seed "sigop_validation" "multisig_repeat.hex" "$(printf '52ae%.0s' {1..20})"

# -----------------------------
# PUSHDATA MASKING (CRITICAL CONSENSUS AREA)
# -----------------------------
add_seed "sigop_validation" "pushdata1_mask.hex" "4c03acacac"
add_seed "sigop_validation" "pushdata2_mask.hex" "4d0400acadaeaf"
add_seed "sigop_validation" "pushdata4_mask.hex" "4e02000000acac"

# Direct push masking (very important for your bug class)
add_seed "sigop_validation" "direct_push.hex" "05acacacacac"

# Push then real opcode boundary test
add_seed "sigop_validation" "push_then_opcode.hex" "03acacacacacacacacac"

# Large push block (DoS-style edge)
add_seed "sigop_validation" "large_push.hex" "4c64$(printf 'ac%.0s' {1..50})"

# -----------------------------
# TRUNCATION / MALFORMED PUSHES
# -----------------------------
add_seed "sigop_validation" "trunc_push1.hex" "4c"
add_seed "sigop_validation" "trunc_push2.hex" "4d01"
add_seed "sigop_validation" "trunc_push4.hex" "4e010000"

# Extended malformed truncation stress
add_seed "sigop_validation" "trunc_repeat.hex" "$(printf '4c%.0s' {1..10})"

# -----------------------------
# TAPSCRIPT / MODERN OPCODES
# -----------------------------
add_seed "sigop_validation" "tapscript_sigop.hex" "ba"

# Mixed tapscript + legacy mix
add_seed "sigop_validation" "tapscript_mixed.hex" "acbaadacba"

# Tapscript stress pattern
add_seed "sigop_validation" "tapscript_repeat.hex" "$(printf 'ba%.0s' {1..20})"

# -----------------------------
# CROSS-BOUNDARY / FUZZ-CRITICAL SEEDS
# (IMPORTANT FOR YOUR SPLITTING MODEL)
# -----------------------------
add_seed "sigop_validation" "boundary_mix1.hex" "ac52ae4c03acba4e010000ad"
add_seed "sigop_validation" "boundary_mix2.hex" "faaeacac4c03acacbaad"
add_seed "sigop_validation" "boundary_mix3.hex" "52aeac52aeac52aeac52ae"

# alternating opcode/push chaos
add_seed "sigop_validation" "alternating.hex" "01ac01ac01ac01ac01ac"

# random-like dense opcode stream
add_seed "sigop_validation" "dense_stream.hex" "acac52ae4cadbaac4e01000052ae"

# -----------------------------
# STRESS / DOS-STYLE INPUTS
# -----------------------------
# many CHECKSIG (classic sigop explosion case)
add_seed "sigop_validation" "many_checksig.hex" "$(printf 'ac%.0s' {1..100})"

# mixed explosion pattern
add_seed "sigop_validation" "mixed_stress.hex" "$(printf 'ac52aeadba%.0s' {1..30})"

# Add Bitcoin Core test vectors if available
BITCOIN_CORE_TEST_DIR="${BITCOIN_CORE_TEST_DIR:-/home/user/src/bitcoin/test/functional/data/util}"
if [ -d "$BITCOIN_CORE_TEST_DIR" ]; then
    echo ""
    echo "Adding Bitcoin Core test vectors from $BITCOIN_CORE_TEST_DIR..."
    
    # Transaction validation - various transaction types
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreate1.hex" ]; then
        add_seed "transaction_validation" "core_txcreate1.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreate1.hex" | tr -d '[:space:]')"
    fi
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreate2.hex" ]; then
        add_seed "transaction_validation" "core_txcreate2.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreate2.hex" | tr -d '[:space:]')"
    fi
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatesignv1.hex" ]; then
        add_seed "transaction_validation" "core_signed_v1.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatesignv1.hex" | tr -d '[:space:]')"
    fi
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatesignv2.hex" ]; then
        add_seed "transaction_validation" "core_signed_v2.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatesignv2.hex" | tr -d '[:space:]')"
    fi
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatesignsegwit1.hex" ]; then
        add_seed "transaction_validation" "core_segwit.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatesignsegwit1.hex" | tr -d '[:space:]')"
    fi
    
    # Script execution - various script patterns
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatescript1.hex" ]; then
        add_seed "script_execution" "core_script1.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatescript1.hex" | tr -d '[:space:]')"
    fi
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatescript2.hex" ]; then
        add_seed "script_execution" "core_script2.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatescript2.hex" | tr -d '[:space:]')"
    fi
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatescript3.hex" ]; then
        add_seed "script_execution" "core_script3.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatescript3.hex" | tr -d '[:space:]')"
    fi
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatescript4.hex" ]; then
        add_seed "script_execution" "core_script4.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatescript4.hex" | tr -d '[:space:]')"
    fi
    
    # Serialization - transaction serialization examples
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatedata1.hex" ]; then
        add_seed "serialization" "core_txdata1.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatedata1.hex" | tr -d '[:space:]')"
    fi
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatedata2.hex" ]; then
        add_seed "serialization" "core_txdata2.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatedata2.hex" | tr -d '[:space:]')"
    fi
    
    # Mempool operations - RBF examples
    if [ -f "$BITCOIN_CORE_TEST_DIR/txreplace1.hex" ]; then
        add_seed "mempool_operations" "core_rbf_replace1.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txreplace1.hex" | tr -d '[:space:]')"
    fi
    
    # Script opcodes - multisig examples
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatemultisig1.hex" ]; then
        add_seed "script_opcodes" "core_multisig1.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatemultisig1.hex" | tr -d '[:space:]')"
    fi
    if [ -f "$BITCOIN_CORE_TEST_DIR/txcreatemultisig2.hex" ]; then
        add_seed "script_opcodes" "core_multisig2.hex" "$(cat "$BITCOIN_CORE_TEST_DIR/txcreatemultisig2.hex" | tr -d '[:space:]')"
    fi
    
    echo "Bitcoin Core test vectors added."
else
    echo ""
    echo "Bitcoin Core test directory not found at $BITCOIN_CORE_TEST_DIR"
    echo "Set BITCOIN_CORE_TEST_DIR environment variable to use test vectors"
fi

echo ""
echo "Corpus initialization complete!"
echo ""
echo "To add more seeds:"
echo "  - Download from bitcoin-core/qa-assets (if available)"
echo "  - Extract from Bitcoin Core test vectors"
echo "  - Use real blockchain data (mainnet/testnet)"
echo ""
echo "Corpus directories: $CORPUS_DIR"


