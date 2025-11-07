# Bitcoin Commons Consensus Proof

**Direct mathematical implementation of Bitcoin consensus rules from the Orange Paper.**

> **For verified system status**: See [SYSTEM_STATUS.md](https://github.com/BTCDecoded/.github/blob/main/SYSTEM_STATUS.md) in the BTCDecoded organization repository.

[![Verification Status](https://github.com/BTCDecoded/bllvm-consensus/workflows/Verify%20Consensus%20(Formal%20Verification)/badge.svg)](https://github.com/BTCDecoded/bllvm-consensus/actions/workflows/verify.yml)
[![Kani Verification](https://img.shields.io/badge/Kani-Verified-green)](https://model-checking.github.io/kani/)
[![Property Tests](https://img.shields.io/badge/Proptest-Covered-blue)](https://docs.rs/proptest/)

This crate provides pure, side-effect-free functions that implement the mathematical specifications defined in the Orange Paper. It serves as the mathematical foundation for Bitcoin consensus validation with **formal verification** ensuring mathematical correctness.

## Architecture Position

This is **Tier 2** of the 5-tier Bitcoin Commons architecture (BLLVM technology stack):

```
1. bllvm-spec (Orange Paper - mathematical foundation)
2. bllvm-consensus (pure math implementation) ← THIS CRATE
3. bllvm-protocol (Bitcoin abstraction)
4. bllvm-node (full node implementation)
5. bllvm-sdk (developer toolkit)
```

## Core Functions

This crate implements all major Bitcoin consensus functions from the Orange Paper:

### Transaction Validation
- Transaction structure and limit validation
- Input validation against UTXO set
- Script execution and verification

### Block Validation
- Block connection and validation
- Transaction application to UTXO set
- Proof of work verification

### Economic Model
- Block reward calculation
- Total supply computation
- Difficulty adjustment

### Mempool Protocol
- Transaction mempool validation
- Standard transaction checks
- Transaction replacement (RBF) logic

### Mining Protocol
- Block creation from mempool
- Block mining and nonce finding
- Block template generation

### Chain Management
- Chain reorganization handling
- P2P network message processing

### Advanced Features
- **SegWit**: Witness data validation and weight calculation
- **Taproot**: P2TR output validation and key aggregation


## Design Principles

1. **Pure Functions**: All functions are deterministic and side-effect-free
2. **Mathematical Accuracy**: Direct implementation of Orange Paper specifications
3. **Exact Version Pinning**: All consensus-critical dependencies pinned to exact versions
4. **Comprehensive Testing**: Extensive test coverage with integration tests
5. **No Consensus Rule Interpretation**: Only mathematical implementation
6. **Formal Verification**: Kani model checking and property-based testing ensure correctness

## Formal Verification

This crate implements **mathematical verification** of Bitcoin consensus rules using:

- **Kani Model Checker**: Symbolic verification with bounded model checking
- **Property-Based Testing**: Randomized testing with `proptest` to discover edge cases
- **Mathematical Specifications**: Formal documentation of consensus invariants
- **CI Enforcement**: Automated verification blocks merge if proofs fail

### Verification Commands

```bash
# Run all tests and verification
cargo test --all-features

# Run Kani model checking
cargo kani --features verify

# Run property tests only
cargo test --test property_tests

# Run specific verification
cargo kani --features verify --harness kani_verify_function
```

### Verification Status

✅ **Chain Selection**: `should_reorganize`, `calculate_chain_work` verified  
✅ **Block Subsidy**: `get_block_subsidy` halving schedule verified  
✅ **Proof of Work**: `check_proof_of_work`, target expansion verified  
✅ **Transaction Validation**: `check_transaction` structure rules verified  
✅ **Block Connection**: `connect_block` UTXO consistency verified  

See [docs/VERIFICATION.md](docs/VERIFICATION.md) for detailed verification documentation.

## Dependencies

All consensus-critical dependencies are pinned to exact versions:

```toml
# Consensus-critical cryptography - EXACT VERSIONS
secp256k1 = "=0.28.2"
sha2 = "=0.10.9"
ripemd = "=0.1.3"
bitcoin_hashes = "=0.11.0"

# Non-consensus-critical utilities - COMPATIBLE VERSIONS
serde = { version = "~1.0", features = ["derive"] }
serde_json = "~1.0"
anyhow = "~1.0"
thiserror = "~1.0"
```

## Testing

```bash
# Run all tests and verification
cargo test --all-features

# Run with coverage
cargo tarpaulin --out Html --output-dir coverage

# Run integration tests
cargo test --test integration_tests
cargo test --test integration_opportunities

# Run formal verification
cargo kani --features verify
```

## Orange Paper Compliance

This implementation covers all major Orange Paper sections:

- **Section 5**: Transaction and Block Validation
- **Section 6**: Script System
- **Section 7**: Economic Model
- **Section 8**: Proof of Work
- **Section 9**: Mempool and Network Protocol
- **Section 10**: Mining Protocol
- **Section 11**: Advanced Features (SegWit, Taproot)

## Security

This crate implements **mathematically verified** Bitcoin consensus rules with:

- **Formal Verification**: Kani model checking prevents consensus violations
- **Property Testing**: Randomized testing discovers edge cases
- **Audit Trail**: OpenTimestamps provides immutable proof of verification
- **CI Enforcement**: No human override of verification results

See [SECURITY.md](SECURITY.md) for security policies and [BTCDecoded Security Policy](https://github.com/BTCDecoded/.github/blob/main/SECURITY.md) for organization-wide guidelines.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [BTCDecoded Contribution Guide](https://github.com/BTCDecoded/.github/blob/main/CONTRIBUTING.md).

**Note**: All consensus changes must pass formal verification before merge. See [docs/VERIFICATION.md](docs/VERIFICATION.md) for verification requirements.

## License

MIT License - see LICENSE file for details.
