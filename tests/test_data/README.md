# Test Data Directory

This directory contains all test data used by blvm-consensus tests.

## Directory Structure

```
test_data/
├── core_vectors/          # Reference JSON vectors
│   ├── transactions/      # Transaction test vectors (tx_valid.json, tx_invalid.json)
│   ├── scripts/           # Script test vectors (if available)
│   └── blocks/            # Block test vectors (if available)
├── ibd_failure_height_*/  # Optional IBD repro dumps (see `tests/block_ibd_repro.rs`)
│   ├── info.txt           # Committed: human-readable height / error summary
│   └── *.bin              # Gitignored: block.bin, utxo_set.bin, witnesses.bin (large)
├── mainnet_blocks/        # Real mainnet blocks at key heights
│   ├── block_0.hex        # Genesis block
│   ├── block_481824.hex   # SegWit activation
│   ├── block_709632.hex   # Taproot activation
│   └── ...
└── checkpoints/           # UTXO set checkpoint hashes
    └── README.md          # Checkpoint format documentation
```

## Test Data Sources

### Reference JSON vectors

**Source**: upstream `bitcoin/bitcoin`  
**Location**: `https://github.com/bitcoin/bitcoin/tree/master/src/test/data`  
**Files**:
- `tx_valid.json` - Valid transaction test cases
- `tx_invalid.json` - Invalid transaction test cases
- `script_valid.json` - Valid script test cases (if available)
- `script_invalid.json` - Invalid script test cases (if available)

**Usage**: Used by `tests/core_test_vectors/` for consensus validation testing.

**Download**: Run `./scripts/download_test_data.sh --core-vectors`

### Mainnet Blocks

**Source**: Blockstream API or reference RPC  
**Location**: Downloaded from `https://blockstream.info/api/block/{hash}/raw`  
**Format**: Hex-encoded Bitcoin wire format  
**Heights**: Key consensus-era blocks (genesis, SegWit activation, Taproot activation, etc.)

**Usage**: Used by `tests/mainnet_blocks.rs` for real-world block validation.

**Download**: Run `./scripts/download_test_data.sh --mainnet-blocks`

### UTXO Set Checkpoints

**Source**: Generated from historical block replay or reference RPC  
**Format**: JSON files with height, UTXO set hash, and block hash  
**Usage**: Used by `tests/integration/historical_replay.rs` for UTXO set verification.

**Generation**: Created during historical block replay or downloaded from trusted sources.

## Downloading Test Data

### Automatic Download

Use the unified download script:

```bash
# Download all test data
./scripts/download_test_data.sh --all

# Download specific categories
./scripts/download_test_data.sh --core-vectors
./scripts/download_test_data.sh --mainnet-blocks
./scripts/download_test_data.sh --checkpoints
```

### Manual Download

#### Core Test Vectors

```bash
mkdir -p tests/test_data/core_vectors/transactions
cd tests/test_data/core_vectors/transactions

curl -L -o tx_valid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/tx_valid.json

curl -L -o tx_invalid.json \
  https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/test/data/tx_invalid.json
```

#### Mainnet Blocks

```bash
# Use the dedicated mainnet blocks script
./scripts/download_mainnet_blocks.sh
```

## Test Data Management

### Verification

After downloading, verify test data integrity:

```bash
# Check Core test vectors
ls -lh tests/test_data/core_vectors/transactions/*.json

# Check mainnet blocks
ls -lh tests/test_data/mainnet_blocks/*.hex

# Verify file formats
file tests/test_data/core_vectors/transactions/tx_valid.json
```

### Cleaning

To remove all test data (useful for fresh downloads):

```bash
rm -rf tests/test_data/core_vectors/transactions/*.json
rm -rf tests/test_data/mainnet_blocks/*.hex
rm -rf tests/test_data/mainnet_blocks/*.bin
```

### Size Considerations

- **Core test vectors**: ~150KB total (tx_valid.json + tx_invalid.json)
- **Mainnet blocks**: ~10-20MB (depending on number of blocks downloaded)
- **Checkpoints**: Minimal size (JSON files)

**Total**: ~20MB for complete test dataset

## Test Data Updates

### When to Update

- **Core test vectors**: When upstream adds new test cases
- **Mainnet blocks**: When new consensus-era heights are needed
- **Checkpoints**: When UTXO set verification is added at new heights

### Update Process

1. Download new test data using scripts
2. Verify new data is valid
3. Run tests to ensure compatibility
4. Update documentation if formats change

## Test Data in CI/CD

Test data is **not** committed to the repository (too large). Instead:

1. **CI downloads test data** during test runs
2. **Local development** uses cached test data
3. **Documentation** explains how to download

### CI Configuration

Add to CI workflow:

```yaml
- name: Download test data
  run: ./scripts/download_test_data.sh --core-vectors
```

## Troubleshooting

### Test Data Not Found

If tests fail with "test data not found":

1. Run `./scripts/download_test_data.sh --all`
2. Verify files exist in `tests/test_data/`
3. Check file permissions
4. Verify network connectivity for downloads

### Invalid Test Data

If test data appears corrupted:

1. Delete corrupted files
2. Re-download using scripts
3. Verify file integrity (checksums if available)
4. Check file format matches expected structure

### Missing Test Data

Some tests gracefully handle missing test data:

- Core test vectors: Tests skip if files not found
- Mainnet blocks: Tests skip if blocks not available
- Checkpoints: Optional for UTXO verification

Check test output for skip messages.

## References

- [Core Test Vectors Documentation](../core_test_vectors/README.md)
- [Mainnet Blocks Documentation](mainnet_blocks/README.md)
- [Historical Replay Tests](../integration/historical_replay.rs)
- [Download Scripts](../../scripts/download_test_data.sh)

