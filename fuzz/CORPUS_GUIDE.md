# Fuzzing Corpus Guide

## Overview

This guide explains how to use corpus seeds to improve fuzzing coverage. Corpus seeds are real-world inputs that guide the fuzzer toward interesting code paths.

## Corpus Directory Structure

```
fuzz/corpus/
├── transaction_validation/
├── block_validation/
├── script_execution/
├── compact_block_reconstruction/
├── segwit_validation/
├── mempool_operations/
├── utxo_commitments/
├── pow_validation/
├── economic_validation/
├── serialization/
└── script_opcodes/
```

## Adding Corpus Seeds

### Transaction Validation

Add real Bitcoin transaction files (hex-encoded):

```bash
# Download real transactions from blockchain
# Convert to hex and save to corpus
echo "0100000001..." > fuzz/corpus/transaction_validation/real_tx_1.hex
```

### Block Validation

Add real Bitcoin block files:

```bash
# Download block hex from block explorers
echo "01000000..." > fuzz/corpus/block_validation/real_block_1.hex
```

### Script Execution

Add interesting script patterns:

- Common P2PKH scripts
- P2SH scripts
- SegWit witness scripts
- Multi-sig scripts
- Time-locked scripts

```bash
# Example: P2PKH script
echo "76a914..." > fuzz/corpus/script_execution/p2pkh.hex
```

## Initializing Corpus

```bash
cd consensus-proof/fuzz

# Initialize corpus directories with seed inputs
./init_corpus.sh

# Or specify custom corpus directory
./init_corpus.sh /path/to/corpus
```

## Running with Corpus

```bash
cd consensus-proof

# Run with corpus (merges new findings)
cargo +nightly fuzz run transaction_validation -- -merge=1 fuzz/corpus/transaction_validation

# Run with existing corpus
cargo +nightly fuzz run transaction_validation fuzz/corpus/transaction_validation

# Run all targets with test runner
python3 fuzz/test_runner.py fuzz/corpus/

# Run with sanitizers
RUSTFLAGS="-Zsanitizer=address" cargo +nightly fuzz run transaction_validation fuzz/corpus/transaction_validation
```

## Corpus Best Practices

1. **Diversity**: Include various transaction types, block sizes, script patterns
2. **Real-world data**: Prefer actual Bitcoin transactions/blocks over synthetic
3. **Size limits**: Keep corpus files reasonable (fuzzer has max_len defaults)
4. **Regular updates**: Add new interesting cases as you discover them

## Sources for Corpus Data

- **Bitcoin Core test vectors**: `/home/user/src/bitcoin/test/functional/data/util/`
  - The `init_corpus.sh` script automatically extracts test vectors if available
  - Set `BITCOIN_CORE_TEST_DIR` environment variable to specify custom path
  - Includes: transaction samples, script patterns, multisig examples, RBF examples
- **Blockchain explorers**: Download hex transactions/blocks
- **Testnet data**: Use testnet for more diverse patterns
- **Historical blocks**: Include blocks from different eras (pre/post SegWit, Taproot, etc.)

## Bitcoin Core Test Vectors

The corpus initialization script automatically includes Bitcoin Core test vectors if available:
- Transaction samples (P2PKH, P2SH, SegWit)
- Script execution patterns
- Serialization examples
- Mempool operations (RBF)
- Multisig examples

These are used as seed data only - consensus-proof validates independently.

