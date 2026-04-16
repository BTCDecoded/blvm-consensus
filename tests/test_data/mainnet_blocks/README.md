# Mainnet Block Test Data

This directory contains Bitcoin mainnet blocks at key consensus-era heights for testing.

## Block Files

Blocks are stored in one of two formats:
- **Binary format**: `block_{height}.bin` (Bitcoin wire format, most efficient)
- **Hex format**: `block_{height}.hex` (hex-encoded, easier to inspect)

## Key Heights

The following blocks are available for testing:

- **Height 0** (Genesis): The first Bitcoin block
- **Height 100000**: Pre-SegWit era
- **Height 200000**: Pre-SegWit era
- **Height 300000**: Pre-SegWit era
- **Height 400000**: Pre-SegWit era
- **Height 481824**: SegWit activation (first SegWit block)
- **Height 500000**: Post-SegWit era
- **Height 600000**: Post-SegWit era
- **Height 709632**: Taproot activation (first Taproot block)
- **Height 800000**: Post-Taproot era
- **Height 900000**: Post-Taproot era

## Downloading Blocks

Use the download script to fetch blocks:

```bash
./scripts/download_mainnet_blocks.sh
```

This will download blocks from Blockstream API and store them in this directory.

## Usage in Tests

Blocks can be loaded using the `load_mainnet_block_from_disk()` helper:

```rust
use consensus_proof::tests::mainnet_blocks::load_mainnet_block_from_disk;

let block_dir = std::path::PathBuf::from("tests/test_data/mainnet_blocks");
let (block, witnesses) = load_mainnet_block_from_disk(&block_dir, 481824)?;
```

## Block Sources

- **Blockstream API**: `https://blockstream.info/api/block/{hash}/raw`
- Alternative: reference RPC `getblock` command
- Alternative: Public block archives

## File Size

- Genesis block: ~285 bytes
- Typical block: 1-2 MB
- Large blocks (with many transactions): Up to 4 MB

## Notes

- Blocks are downloaded from public APIs (Blockstream)
- Blocks are verified to match reference validation
- Hex format is used for easier inspection and debugging
- Binary format is more efficient for large-scale testing

