# BLVM consensus

Orange Paper–aligned Bitcoin consensus in Rust: deterministic validation, **`#[spec_locked]`** checks with **blvm-spec-lock**, and broad coverage (unit, integration, property, fuzz).

**Crate:** [`blvm-consensus`](https://crates.io/crates/blvm-consensus) · [docs.rs](https://docs.rs/blvm-consensus)

> **Documentation:** [docs.btcdecoded.org/blvm](https://docs.btcdecoded.org/blvm/) · Sources: [blvm-docs](https://github.com/BTCDecoded/blvm-docs).  
> **Ecosystem status:** [SYSTEM_STATUS.md](https://github.com/BTCDecoded/.github/blob/main/SYSTEM_STATUS.md) (BTCDecoded org).

[![crates.io](https://img.shields.io/crates/v/blvm-consensus.svg)](https://crates.io/crates/blvm-consensus)
[![docs.rs](https://docs.rs/blvm-consensus/badge.svg)](https://docs.rs/blvm-consensus)
[![CI](https://github.com/BTCDecoded/blvm-consensus/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BTCDecoded/blvm-consensus/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

[![Property Tests](https://img.shields.io/badge/Proptest-Covered-blue)](https://docs.rs/proptest/)

## Architecture Position

Tier 2 of the BLVM stack. Bitcoin Commons is the specification and the governance model. BLVM is one implementation of that spec.

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
6. **Spec-lock**: `#[spec_locked]` functions are Z3-checked. The current figures are in [docs/VERIFICATION.md](docs/VERIFICATION.md)

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

# Example: property-heavy integration targets (see tests/*.rs)
cargo test --test consensus_property_tests
cargo test --test orange_paper_property_tests
```

### Verification status

A function is proved when a deliberate break in its code makes the Z3 check fail. A green `cargo spec-lock verify` run is not that proof.

Two lists count different things. Do not add them.

- [docs/VERIFICATION.md](docs/VERIFICATION.md) is the written list: **93** proved functions out of **224** tracked. `get_block_subsidy`, `check_proof_of_work`, `expand_target`, and `check_transaction` are on it. `should_reorganize`, `calculate_chain_work`, and `connect_block` are not.
- `blvm-spec-lock` `docs/MUTATION_COVERAGE.md` is the list the tool writes: **34** functions, all proved. `connect_block` is on this list. `connect_block_inner` has no `#[spec_locked]` attribute.

The Orange Paper inventory (`blvm-spec-lock` `docs/sentence-inventory.tsv`) has 274 sentences. The subsidy theorem (6.1.2) is proved in Rust for the heights the tool checks and proved in Lean for every natural number. Sixteen sentences are Lean proofs of the specification only. Lean does not check the Rust code. The other 257 sentences are not proved yet.  

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
- Spec-lock verification for critical BIPs

BIP66 and BIP147 are enforced during script verification, which is called for all transactions in `connect_block()`. This is the correct approach as these BIPs apply to individual signatures and script execution.

## Dependencies

**Monorepo vs crates.io:** In the BTCDecoded layout, this crate’s `Cargo.toml` uses **path** dependencies for sibling `blvm-*` crates. When you depend on a **published** release from [crates.io](https://crates.io/crates/blvm-consensus), use:

```toml
[dependencies]
blvm-consensus = "0.1"
```

(`0.1` matches any **0.1.**z on crates.io; use `=` + a full version if you need a locked patch for reproducibility.)

**Dependency constraints** follow the canonical [`Cargo.toml`](https://github.com/BTCDecoded/blvm-consensus/blob/main/Cargo.toml): many **blvm-*** crates use **pre-1.0 ranges** on crates.io; **third-party** crates use **`=`** pins only where the manifest does. Do not treat this README as a second copy of the lockfile—open the manifest (and lockfile if your checkout has one) for the exact graph.

**Requirements**: satisfy **`rust-version`** in **`Cargo.toml`** (currently **1.83+**).

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

Property-based tests (`proptest`), differential comparisons where configured, and fuzz targets under **`fuzz/`** (see [`fuzz/README.md`](fuzz/README.md)). CI runs **`cargo test --all-features`** and a nightly **Miri** job on selected targets — see [`docs/MIRI.md`](docs/MIRI.md) for scope and limitations.

```bash
# Run all tests and verification
cargo test --all-features

# Run with coverage
cargo tarpaulin --out Html --output-dir coverage

# Run integration tests
cargo test --test integration_tests
cargo test --test integration_opportunities

# Run formal verification (blvm-spec-lock)
cargo spec-lock verify --crate-path .
```

## Spec linkage

Annotated functions carry `#[spec_locked]` and name an Orange Paper section. CI runs Z3 on those annotations. The written list is 93 proved functions out of 224, in [docs/VERIFICATION.md](docs/VERIFICATION.md). The tool's own list is 34 proved functions. The subsidy theorem (6.1.2) is the one Orange Paper sentence proved in both Rust and Lean. Witnesses in `spec_witnesses.rs` are not the production functions. `connect_block_inner` has no `#[spec_locked]` attribute.

**Chain of trust**:
```
Orange Paper (spec) → #[spec_locked] on the functions that carry it → implementation
```

**Verification approaches** (details and scope: [docs/VERIFICATION.md](docs/VERIFICATION.md)):
- **Spec-lock** on `#[spec_locked]` functions against the Orange Paper
- **Property tests**, integration tests, and optional Bolero-backed tests
- **Runtime assertions** on selected paths (see feature flags and [PROOF_LIMITATIONS.md](docs/PROOF_LIMITATIONS.md))
- **Coverage-guided fuzzing** ([Fuzzing (BLVM docs)](https://docs.btcdecoded.org/blvm/development/fuzzing.html); harnesses under `fuzz/`)

## Orange Paper sections

The top-level sections in `PROTOCOL.md` are:

- **5**: State Transition Functions
- **6**: Economic Model
- **7**: Proof of Work
- **8**: Security Properties
- **9**: Mempool Protocol
- **10**: Network Protocol
- **11**: Advanced Features
- **12**: Mining Protocol

An annotated function names one of those sections. That name is not a proof that the section is fully implemented. The mutation gate is in [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Security

Checks on this crate:

- **Spec-lock**: Z3 on `#[spec_locked]` functions. A passing verify job is not the mutation gate in [docs/VERIFICATION.md](docs/VERIFICATION.md).
- **Property testing**: randomized tests under `tests/`, including `consensus_property_tests`.
- **CI**: the verify job runs unless the commit message contains `[skip ci]` or `[skip_verify]`, or a workflow dispatch sets `skip_tests`. OpenTimestamps is designed to stamp the verify artifacts later. It is not a step in `.github/workflows/ci.yml` today.

See [SECURITY.md](SECURITY.md) for security policies and [BTCDecoded Security Policy](https://github.com/BTCDecoded/.github/blob/main/SECURITY.md) for organization-wide guidelines.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [BTCDecoded Contribution Guide](https://github.com/BTCDecoded/.github/blob/main/CONTRIBUTING.md).

**Note**: Consensus changes are expected to pass the tests and the verify job. A commit message can skip that job. See [docs/VERIFICATION.md](docs/VERIFICATION.md).

## License

MIT License - see LICENSE file for details.
