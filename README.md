# Bitcoin Commons Consensus Proof

Pure mathematical implementation of Bitcoin consensus rules from the Orange Paper with formal verification.

> **📚 Comprehensive Documentation**: See [blvm-docs](../blvm-docs/) for complete system documentation.  
> **For verified system status**: See [SYSTEM_STATUS.md](https://github.com/BTCDecoded/.github/blob/main/SYSTEM_STATUS.md) in the BTCDecoded organization repository.

[![Verification Status](https://github.com/BTCDecoded/blvm-consensus/workflows/Verify%20Consensus%20(Formal%20Verification)/badge.svg)](https://github.com/BTCDecoded/blvm-consensus/actions/workflows/verify.yml)
[![blvm-spec-lock Verification](https://img.shields.io/badge/blvm--spec--lock-Verified-green)](https://github.com/BTCDecoded/blvm-spec-lock)
[![Property Tests](https://img.shields.io/badge/Proptest-Covered-blue)](https://docs.rs/proptest/)

Provides pure, side-effect-free functions implementing Orange Paper mathematical specifications. Serves as the mathematical foundation for Bitcoin consensus validation with formal verification ensuring mathematical correctness.

## Architecture Position

Tier 2 of the 6-tier Bitcoin Commons architecture (BLVM technology stack):

```
1. blvm-spec (Orange Paper - mathematical foundation)
2. blvm-consensus (pure math implementation)
3. blvm-protocol (Bitcoin abstraction)
4. blvm-node (full node implementation)
5. blvm-sdk (developer toolkit)
6. blvm-commons (governance enforcement)
```

## Core Functions

Implements all major Bitcoin consensus functions from the Orange Paper:

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
6. **Formal Verification**: blvm-spec-lock formal verification and property-based testing ensure correctness

See [docs/VERIFICATION.md](docs/VERIFICATION.md) for detailed verification documentation.

## Formal Verification

Implements mathematical verification of Bitcoin consensus rules using:

- **blvm-spec-lock**: Formal verification linking code to Orange Paper specifications
- **Property-Based Testing**: Randomized testing with `proptest` to discover edge cases
- **Mathematical Specifications**: Formal documentation of consensus invariants
- **CI Enforcement**: Automated verification blocks merge if proofs fail

### Verification Commands

```bash
# Run all tests and verification
cargo test --all-features

# Run property tests only
cargo test --test property_tests
```

### Verification Status

**Chain Selection**: `should_reorganize`, `calculate_chain_work` verified  
**Block Subsidy**: `get_block_subsidy` halving schedule verified  
**Proof of Work**: `check_proof_of_work`, target expansion verified  
**Transaction Validation**: `check_transaction` structure rules verified  
**Block Connection**: `connect_block` UTXO consistency verified  

## BIP Implementation Status


All critical Bitcoin Improvement Proposals (BIPs) are implemented and integrated:

- **BIP30** - Duplicate coinbase prevention (integrated in `connect_block()`)
- **BIP34** - Block height in coinbase (integrated in `connect_block()`)  
- **BIP66** - Strict DER signatures (enforced via script verification with flag 0x04)
- **BIP90** - Block version enforcement (integrated in `connect_block()`)
- **BIP147** - NULLDUMMY enforcement (enforced via script verification with flag 0x10)

**Integration**:
- All BIPs integrated into block validation
- BIP66 and BIP147 enforced during script verification (called for all transactions in `connect_block()`)
- Integration tests verify enforcement
- Kani proofs exist for critical BIPs

BIP66 and BIP147 are enforced during script verification, which is called for all transactions in `connect_block()`. This is the correct approach as these BIPs apply to individual signatures and script execution.

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

## Performance Optimizations

### Profile-Guided Optimization (PGO)

For maximum performance, use Profile-Guided Optimization:

```bash
# Automated PGO build (recommended)
./scripts/pgo-build.sh

# Clean and rebuild with PGO
./scripts/pgo-build.sh clean
```

**Expected gain**: 1.10-1.15x performance improvement

PGO works by:
1. Building with instrumentation to collect execution profiles
2. Running benchmarks to generate profile data
3. Rebuilding with profile data to optimize hot paths

**Requirements**: Rust 1.70+ (LLVM 15+)

### Production Build

For production builds with all optimizations:

```bash
cargo build --release --features production
```

This enables:
- LTO (Link-Time Optimization)
- Runtime optimizations (caching, context reuse)
- SIMD vectorization
- Memory layout optimizations

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

## Mathematical Lock

Implementation is mathematically locked to the Orange Paper specification:

- Every function implements a mathematical specification from the Orange Paper
- Every critical function has a Kani proof verifying correctness
- All proofs reference Orange Paper sections and theorems
- No consensus rule can be changed without updating both spec and proof

Formal verification linkage level is unique among Bitcoin implementations.

**Chain of Trust**:
```
Orange Paper (Math Spec) → Kani Proof → Implementation → Bitcoin Consensus
```

**Why This Matters for Bitcoin**:
- Consensus rules are immutable - once deployed, they cannot change
- Network divergence is catastrophic - all nodes must agree
- Security is critical - billions of dollars depend on correctness
- Mathematical proof exceeds human review

**Verification Statistics**:
- 201 Kani proofs verify all critical consensus functions (201 in `src/`, 9 in `tests/`)
- 35 property tests verify mathematical invariants
- 913 runtime assertions catch edge cases (814 `assert!` + 99 `debug_assert!`)
- 13 fuzz targets discover vulnerabilities

## Orange Paper Compliance

Covers all major Orange Paper sections:

- **Section 5**: Transaction and Block Validation
- **Section 6**: Script System
- **Section 7**: Economic Model
- **Section 8**: Proof of Work
- **Section 9**: Mempool and Network Protocol
- **Section 10**: Mining Protocol
- **Section 11**: Advanced Features (SegWit, Taproot)

## Security

Implements mathematically verified Bitcoin consensus rules with:

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
