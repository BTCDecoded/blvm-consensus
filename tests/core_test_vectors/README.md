# Bitcoin Core Test Vector Integration

This directory contains integration code for running Bitcoin Core's test vectors to verify consensus correctness.

## Overview

Bitcoin Core maintains comprehensive test vectors in JSON format that cover:
- Transaction validation (`tx_valid.json`, `tx_invalid.json`)
- Script execution (`script_valid.json`, `script_invalid.json`)
- Block validation (`block_valid.json`, `block_invalid.json`)

These test vectors represent decades of consensus bug fixes and edge cases discovered through real-world usage.

## Setup

### 1. Download Core Test Vectors

Test vectors are located in Bitcoin Core's repository:
```
https://github.com/bitcoin/bitcoin/tree/master/src/test/data
```

You can download them directly:
```bash
# Create test data directory
mkdir -p tests/test_data/core_vectors/{transactions,scripts,blocks}

# Download test vectors
curl -o tests/test_data/core_vectors/transactions/tx_valid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/tx_valid.json

curl -o tests/test_data/core_vectors/transactions/tx_invalid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/tx_invalid.json

curl -o tests/test_data/core_vectors/scripts/script_valid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/script_valid.json

curl -o tests/test_data/core_vectors/scripts/script_invalid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/script_invalid.json
```

### 2. Run Tests

```bash
# Run transaction test vectors
cargo test --test core_test_vectors::transaction_tests

# Run script test vectors
cargo test --test core_test_vectors::script_tests

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

### Script Test Vectors

Format: `[[scriptSig_hex, scriptPubKey_hex, flags, expected, description], ...]`

Example:
```json
[
  ["51", "51", "0x0001", true, "OP_1 OP_1 OP_EQUAL"],
  ["00", "ac", "0x0001", false, "Invalid signature"]
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



