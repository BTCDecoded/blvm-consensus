# Consensus Scripts

This directory contains scripts for managing test data and verification for blvm-consensus.

## Scripts

- `download_test_data.sh` - Download test data for consensus tests
- `download_mainnet_blocks.sh` - Download mainnet blocks for testing
- `manage_test_data.sh` - Manage test data files
- `verify_core_test_vectors.sh` - Verify upstream reference vectors

## Usage

These scripts are used to:
- Download and manage test data for consensus validation
- Verify against upstream vectors
- Prepare test data for fuzzing and property testing

See [blvm-consensus/README.md](../README.md) for more information about the consensus layer.

## Related

- `../fuzz/` - Fuzzing scripts (init_corpus.sh, run_campaigns.sh, etc.)
- `../tests/` - Test files that use this test data

