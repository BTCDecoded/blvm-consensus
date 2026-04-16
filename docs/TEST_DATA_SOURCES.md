# Test Data Sources

This document describes all sources of test data used by blvm-consensus tests.

## Overview

Test data is organized into three main categories:
1. **Reference JSON vectors** - Official test cases from the reference `bitcoin/bitcoin` tree
2. **Mainnet Blocks** - Real Bitcoin blocks at key consensus-era heights
3. **UTXO Set Checkpoints** - Verified UTXO set hashes at specific heights

## Upstream reference vectors

### Source
- **Repository**: https://github.com/bitcoin/bitcoin
- **Path**: `src/test/data/`
- **License**: MIT (same license as upstream)
- **Update Frequency**: Updated when upstream refreshes vectors

### Files

#### Transaction Test Vectors
- **File**: `tx_valid.json`
- **Format**: JSON array of transaction test cases
- **Size**: ~86KB
- **Content**: Valid transaction test cases with expected results
- **Usage**: Used by `tests/core_test_vectors/transaction_tests.rs`

- **File**: `tx_invalid.json`
- **Format**: JSON array of transaction test cases
- **Size**: ~53KB
- **Content**: Invalid transaction test cases with rejection reasons
- **Usage**: Used by `tests/core_test_vectors/transaction_tests.rs`

#### Script Test Vectors
- **Files**: `script_valid.json`, `script_invalid.json`
- **Status**: Not available as standalone JSON files
- **Note**: The reference client uses functional tests for scripts, not JSON vectors
- **Alternative**: Use our comprehensive script tests in `tests/unit/script_tests.rs`

#### Block Test Vectors
- **Files**: `block_valid.json`, `block_invalid.json`
- **Status**: Not available as standalone JSON files
- **Note**: The reference client uses functional tests for blocks, not JSON vectors
- **Alternative**: Use our comprehensive block tests and mainnet blocks

### Download

**Automatic**:
```bash
./scripts/download_test_data.sh --core-vectors
```

**Manual**:
```bash
mkdir -p tests/test_data/core_vectors/transactions
cd tests/test_data/core_vectors/transactions

curl -L -o tx_valid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/tx_valid.json

curl -L -o tx_invalid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/tx_invalid.json
```

### Verification

Test vectors should be verified against the official repository:
- Check file hashes match upstream
- Verify JSON format is valid
- Ensure test cases are complete

## Mainnet Blocks

### Source
- **Primary**: Blockstream API (`https://blockstream.info/api/block/{hash}/raw`)
- **Alternative**: reference full-node RPC (`getblock` command with verbosity=0)
- **Format**: Raw Bitcoin wire format (hex-encoded)
- **Update Frequency**: As needed for new consensus-era testing

### Key Heights

Blocks are downloaded at key consensus-era heights:

| Height | Event | Block Hash (Genesis) |
|--------|-------|---------------------|
| 0 | Genesis block | 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f |
| 100000 | Pre-SegWit era | (varies) |
| 200000 | Pre-SegWit era | (varies) |
| 300000 | Pre-SegWit era | (varies) |
| 400000 | Pre-SegWit era | (varies) |
| 481824 | SegWit activation | (varies) |
| 500000 | Post-SegWit | (varies) |
| 600000 | Post-SegWit | (varies) |
| 709632 | Taproot activation | (varies) |
| 800000 | Post-Taproot | (varies) |
| 900000 | Post-Taproot | (varies) |

### Download

**Automatic**:
```bash
./scripts/download_test_data.sh --mainnet-blocks
# or
./scripts/download_mainnet_blocks.sh
```

**Manual** (using Blockstream API):
```bash
# Get block hash for height
HEIGHT=481824
HASH=$(curl -s "https://blockstream.info/api/block-height/$HEIGHT")

# Download block
curl -s "https://blockstream.info/api/block/$HASH/raw" | \
  xxd -p -c 0 > tests/test_data/mainnet_blocks/block_${HEIGHT}.hex
```

**Manual** (using reference full-node RPC):
```bash
# Get block hash
HASH=$(bitcoin-cli getblockhash 481824)

# Get raw block
bitcoin-cli getblock $HASH 0 > tests/test_data/mainnet_blocks/block_481824.hex
```

### Verification

Blocks should be verified:
- Check block hash matches expected value
- Verify block format is valid Bitcoin wire format
- Ensure block deserializes correctly
- Validate block height matches filename

## UTXO Set Checkpoints

### Source
- **Primary**: Generated from historical block replay
- **Alternative**: reference full-node RPC (`gettxoutsetinfo` at specific heights)
- **Format**: JSON files with height, UTXO set hash, block hash
- **Update Frequency**: As needed for new checkpoint heights

### Format

```json
{
  "height": 100000,
  "utxo_set_hash": "hex_string_32_bytes",
  "block_hash": "hex_string_32_bytes",
  "timestamp": 1234567890,
  "utxo_count": 1234567,
  "total_amount": 12345678901234
}
```

### Generation

**From Historical Replay**:
```rust
// In tests/integration/historical_replay.rs
let utxo_hash = calculate_utxo_set_hash(&utxo_set);
let checkpoint = Checkpoint {
    height: 100000,
    utxo_set_hash: utxo_hash,
    block_hash: block.header.merkle_root,
    timestamp: block.header.timestamp,
};
```

**From reference full-node RPC**:
```bash
# Get UTXO set info at specific height
bitcoin-cli gettxoutsetinfo | jq '{height: .height, hash: .hash}'
```

### Usage

Checkpoints are used in `tests/integration/historical_replay.rs` to verify UTXO set correctness during block replay.

## Data Integrity

### Checksums

Test data should be verified using checksums:

```bash
# Generate checksums
find tests/test_data -type f -exec sha256sum {} \; > test_data_checksums.txt

# Verify checksums
sha256sum -c test_data_checksums.txt
```

### Version Tracking

Test data versions should be tracked:
- Document which upstream snapshot test vectors are from
- Track when mainnet blocks were downloaded
- Note checkpoint generation timestamp

## Storage Considerations

### Repository Policy

- **Test data is NOT committed** to the repository (too large, ~20MB)
- **Scripts ARE committed** for downloading test data
- **CI downloads test data** during test runs
- **Local development** uses cached test data

### Size Estimates

- **Core test vectors**: ~150KB
- **Mainnet blocks**: ~10-20MB (depending on number of blocks)
- **Checkpoints**: Minimal (<10KB)
- **Total**: ~20MB for complete dataset

### Caching

Test data can be cached:
- Local: `tests/test_data/` (gitignored)
- CI: Cache test data between runs
- Shared: Use shared test data directory for multiple projects

## Troubleshooting

### Download Failures

If downloads fail:
1. Check network connectivity
2. Verify URLs are accessible
3. Check for rate limiting (Blockstream API)
4. Try alternative sources

### Invalid Data

If test data appears invalid:
1. Re-download from source
2. Verify file format matches expected structure
3. Check for corruption (checksums)
4. Ensure file permissions are correct

### Missing Data

Some tests gracefully handle missing data:
- Tests skip if data not available
- Check test output for skip messages
- Re-download missing data using scripts

## References

- [Upstream test data](https://github.com/bitcoin/bitcoin/tree/master/src/test/data)
- [Blockstream API](https://blockstream.info/api-doc)
- [reference full-node RPC](https://bitcoincore.org/en/doc/)
- [Test Data Management](../scripts/manage_test_data.sh)

