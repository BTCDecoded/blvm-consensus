# Reference JSON vector Integration

This directory contains integration code for running reference JSON vectors to verify consensus correctness.

## Overview

The reference `bitcoin/bitcoin` tree ships JSON for **transactions** (`tx_valid.json`, `tx_invalid.json`) under `src/test/data/`. This crate wires those into `transaction_tests.rs`.

**Block** JSON vectors are **not** published by Core the same way; `block_tests.rs` only runs cases if you supply your own `block_valid.json` / `block_invalid.json` locally (this repo does not commit them).

Script consensus is covered by in-tree Rust tests (for example `tests/unit/script_tests.rs`), not by committed Core JSON in this crate.

These test vectors represent decades of consensus bug fixes and edge cases discovered through real-world usage.

## Setup

### 1. Download Core Test Vectors

Test vectors are located in the upstream repository:
```
https://github.com/bitcoin/bitcoin/tree/master/src/test/data
```

You can download them directly:
```bash
# Create test data directory
mkdir -p tests/test_data/core_vectors/transactions
# Optional: mkdir -p tests/test_data/core_vectors/blocks   # only if you add custom block_*.json

# Download test vectors
curl -o tests/test_data/core_vectors/transactions/tx_valid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/tx_valid.json

curl -o tests/test_data/core_vectors/transactions/tx_invalid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/tx_invalid.json
```

### 2. Run Tests

```bash
# Run transaction test vectors
cargo test --test core_test_vectors::transaction_tests

# Run block test vectors
cargo test --test core_test_vectors::block_tests
```

## Test Vector Formats

### Transaction Test Vectors

Format: `[[tx_hex, witness_hex?, flags, description], ...]`

Example:
```json
[
  ["0100000001...", "0x0001", "Standard transaction"],
  ["0100000002...", "0x0001", "P2SH transaction"]
]
```

### Block Test Vectors

Format: `[[block_hex, height, description], ...]`

Example:
```json
[
  ["01000000...", 0, "Genesis block"],
  ["01000000...", 481824, "First SegWit block"]
]
```

## Integration with CI

To run Core test vectors in CI, add them to your test data directory or use a git submodule:

```bash
git submodule add https://github.com/bitcoin/bitcoin.git test_data/bitcoin-core
```

Then update the paths in the test files to point to the submodule.

## Coverage

These test vectors provide coverage for:
- All consensus-critical validation rules
- Historical consensus bugs (CVE tests)
- Edge cases discovered through real-world usage
- Soft fork activation scenarios
- Script opcode combinations
- Serialization edge cases

## Notes

- Test vectors are optional - if the directory doesn't exist, tests will pass (empty vectors)
- Some test vectors may require additional context (UTXO sets, previous blocks) not provided in the JSON
- Test vectors are updated as Core discovers new edge cases






