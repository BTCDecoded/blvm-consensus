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
    "differential_fuzzing"
    "bip66_validation"
    "sighash_computation"
    "witness_validation"
    "sigop_validation"
    "sequence_locks_validation"
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

add_binary_seed() {
    local target=$1
    local filename=$2
    local content=$3
    printf '%b' "$content" > "$CORPUS_DIR/$target/$filename"
    echo "Added binary seed: $target/$filename"
    shift 2
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

# BIP66 validation seeds
# Input format: byte[0]=selector (0=pre-activation, 1=activation boundary, 2=regtest),
# bytes[1..]=signature candidate
# Valid minimal DER: selector + 0x30 [total-len] 0x02 [R-len] [R] 0x02 [S-len] [S] [sighash]
add_seed "bip66_validation" "valid_minimal_regtest.hex" "02300602010102010101"
add_seed "bip66_validation" "valid_minimal_mainnet.hex" "01300602010102010101"
# Pre-activation: anything passes
add_seed "bip66_validation" "pre_activation_garbage.hex" "00ffffff"
# Too short for DER (should reject when active)
add_seed "bip66_validation" "too_short.hex" "023001"
# Wrong SEQUENCE tag (should reject when active)
add_seed "bip66_validation" "wrong_tag.hex" "02310602010102010101"
# Empty signature (selector only)
add_seed "bip66_validation" "empty_sig.hex" "02"

# Sighash computation seeds (binary format — uses printf, not add_seed)
# Header: [scenario, sighash_byte, input_idx, n_inputs, n_outputs,
#          version(4), locktime(4), prevout_value(8), seq_seed, val_seed, script...]
# Scenario 0: Legacy SIGHASH_ALL determinism + perturbation (two distinct script halves)
add_binary_seed "sighash_computation" "legacy_all_determ.bin" \
    '\x00\x01\x00\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\xe1\xf5\x05\x00\x00\x00\x00\xff\x32\x76\xa9\x14\x00\x00\x51\x52\x53'

# Scenario 1: SIGHASH_SINGLE quirk — 2 inputs, 0 outputs (triggers quirk)
add_binary_seed "sighash_computation" "single_quirk_empty.bin" \
    '\x01\x03\x00\x02\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\xe1\xf5\x05\x00\x00\x00\x00\xff\x32\x76\xa9'

# Scenario 1: SIGHASH_SINGLE boundary — 2 inputs, 1 output, input_index=1 (idx >= outputs.len())
add_binary_seed "sighash_computation" "single_quirk_boundary.bin" \
    '\x01\x03\x01\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\xe1\xf5\x05\x00\x00\x00\x00\xff\x32\x76\xa9'

# Scenario 2: Legacy bounds error — input_index=0xFF with 1 input
add_binary_seed "sighash_computation" "legacy_bounds.bin" \
    '\x02\x01\xff\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\xe1\xf5\x05\x00\x00\x00\x00\xff\x32\x76\xa9'

# Scenario 3: BIP143 SIGHASH_ALL determinism + perturbation
add_binary_seed "sighash_computation" "bip143_all_determ.bin" \
    '\x03\x01\x00\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\xe1\xf5\x05\x00\x00\x00\x00\xff\x32\x76\xa9\x14\x00\x00\x51\x52\x53'

# Scenario 4: BIP143 NONE|ANYONECANPAY precomputed equivalence
add_binary_seed "sighash_computation" "bip143_precomputed.bin" \
    '\x04\x82\x00\x02\x02\x01\x00\x00\x00\x00\x00\x00\x00\x00\x88\x13\x00\x00\x00\x00\x00\x00\xff\x32\x76\xa9\x14'

# Scenario 5: BIP143 bounds error — input_index=0xFF with 1 input
add_binary_seed "sighash_computation" "bip143_bounds.bin" \
    '\x05\x01\xff\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\xe1\xf5\x05\x00\x00\x00\x00\xff\x32\x76\xa9'

# Extra: SIGHASH_NONE with 3 inputs, varied sequences
add_binary_seed "sighash_computation" "legacy_none_multi.bin" \
    '\x00\x02\x01\x03\x02\x01\x00\x00\x00\x00\x00\x00\x00\x00\xe1\xf5\x05\x00\x00\x00\x00\xab\x32\x76\xa9\x14\x00\x00\x51\x52\x53'
    
# Witness validation seeds
# Boundary 520 bytes
add_seed "witness_validation" "boundary_520.hex" "$(printf '01%.0s' {1..520})"
# Boundary 521 bytes
add_seed "witness_validation" "boundary_521.hex" "$(printf '01%.0s' {1..521})"
# Taproot Control Block (33 bytes)
add_seed "witness_validation" "taproot_cb_33.hex" "$(printf '00%.0s' {1..33})"
# Taproot Control Block (65 bytes)
add_seed "witness_validation" "taproot_cb_65.hex" "$(printf '00%.0s' {1..65})"
# Taproot Invalid CB (34 bytes)
add_seed "witness_validation" "taproot_cb_34.hex" "$(printf '00%.0s' {1..34})"
# Degenerate: 32 empty elements
add_seed "witness_validation" "degenerate_empty.hex" "$(printf '00%.0s' {1..32})"

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

# Sequence locks validation seeds
# Layout follows fuzz_targets/sequence_locks_validation.rs custom Arbitrary parser:
# input_len | header_len | inputs... | headers... | tx_version | flags | block_height |
# block_time | lock_time_probe | stack_locktime_probe
add_binary_seed "sequence_locks_validation" "disabled_sequence.bin" \
"\x01\x00\
\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\
\x02\x00\x00\x00\x00\x00\x00\x00\
\x01\x00\x00\x00\
\x65\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\
\x00\x00\x00\x00"

add_binary_seed "sequence_locks_validation" "height_boundary_equal.bin" \
"\x01\x00\
\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\
\x02\x00\x00\x00\x00\x00\x00\x00\
\x01\x00\x00\x00\
\x01\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\
\x00\x00\x00\x00"

add_binary_seed "sequence_locks_validation" "time_based_threshold.bin" \
"\x01\x01\
\x00\x00\x40\x00\x0a\x00\x00\x00\x00\x00\x00\x00\
\xf4\x01\x00\x00\x00\x00\x00\x00\
\x02\x00\x00\x00\x00\x00\x00\x00\
\x01\x00\x00\x00\
\x14\x00\x00\x00\x00\x00\x00\x00\
\xf3\x01\x00\x00\x00\x00\x00\x00\
\x00\x65\xcd\x1d\
\x00\x65\xcd\x1d"

add_binary_seed "sequence_locks_validation" "time_based_unsorted_eleven.bin" \
"\x01\x0b\
\xff\xff\x40\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x08\x00\x00\x00\x00\x00\x00\
\x00\x04\x00\x00\x00\x00\x00\x00\
\x00\x0c\x00\x00\x00\x00\x00\x00\
\x00\x02\x00\x00\x00\x00\x00\x00\
\x00\x0a\x00\x00\x00\x00\x00\x00\
\x00\x06\x00\x00\x00\x00\x00\x00\
\x00\x0e\x00\x00\x00\x00\x00\x00\
\x00\x01\x00\x00\x00\x00\x00\x00\
\x00\x10\x00\x00\x00\x00\x00\x00\
\x00\x03\x00\x00\x00\x00\x00\x00\
\x00\x12\x00\x00\x00\x00\x00\x00\
\x00\x05\x00\x00\x00\x00\x00\x00\
\x02\x00\x00\x00\x00\x00\x00\x00\
\x01\x00\x00\x00\
\xff\x11\x00\x00\x00\x00\x00\x00\
\xff\x11\x00\x00\x00\x00\x00\x00\
\xff\x64\xcd\x1d\
\x00\x65\xcd\x1d"

add_binary_seed "sequence_locks_validation" "locktime_threshold_types.bin" \
"\x00\x00\
\x01\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\
\xff\x64\xcd\x1d\
\x00\x65\xcd\x1d"

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