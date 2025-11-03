# Fuzzing Infrastructure for Consensus-Proof

This directory contains comprehensive fuzzing infrastructure for the consensus-proof crate, modeled after Bitcoin Core's fuzzing approach.

## Overview

- **12 Fuzz Targets**: Covering all critical consensus validation functions
- **libFuzzer**: Primary fuzzing engine (LLVM-based)
- **Sanitizers**: ASAN, UBSAN, MSAN support
- **Corpus Management**: Automated corpus initialization and management
- **Test Runner**: Python script for parallel execution and corpus management
- **Differential Fuzzing**: Internal consistency testing (independent of Bitcoin Core)
- **CI Integration**: Continuous fuzzing in GitHub Actions

## Quick Start

### 1. Initialize Corpus

```bash
cd consensus-proof/fuzz
./init_corpus.sh
```

This creates corpus directories and adds basic seed inputs.

### 2. Run a Fuzzing Campaign

```bash
# Run single target (5 minutes)
cargo +nightly fuzz run transaction_validation

# Run with corpus
cargo +nightly fuzz run transaction_validation fuzz/corpus/transaction_validation

# Run all targets (24 hours each, background)
./run_campaigns.sh --background

# Run with test runner (parallel execution)
python3 test_runner.py fuzz/corpus/ --parallel
```

### 3. Build with Sanitizers

```bash
# AddressSanitizer (ASAN)
./build_with_sanitizers.sh asan

# UndefinedBehaviorSanitizer (UBSAN)
./build_with_sanitizers.sh ubsan

# All sanitizers
./build_with_sanitizers.sh all
```

## Fuzz Targets

### Core Consensus (Critical)
1. **transaction_validation** - Transaction parsing and validation
2. **block_validation** - Block validation and connection
3. **script_execution** - Script VM execution
4. **script_opcodes** - Individual opcode execution

### Advanced Features
5. **segwit_validation** - SegWit weight calculations and witness validation
6. **mempool_operations** - Mempool acceptance, RBF, standardness checks
7. **utxo_commitments** - UTXO commitment verification

### Infrastructure
8. **serialization** - Serialization/deserialization round-trips
9. **pow_validation** - Proof of Work and difficulty adjustment
10. **economic_validation** - Supply calculations and fee validation
11. **compact_block_reconstruction** - Compact block parsing
12. **differential_fuzzing** - Internal consistency testing (validation, serialization, calculations)

## Running Campaigns

### Short Verification (5 minutes per target)

```bash
./run_campaigns.sh 300
```

### Full Campaigns (24 hours per target)

```bash
# Sequential (one at a time)
./run_campaigns.sh 86400

# Background (all in parallel)
./run_campaigns.sh --background
```

### Using Test Runner

```bash
# Run all targets sequentially
python3 test_runner.py fuzz/corpus/

# Run specific targets
python3 test_runner.py fuzz/corpus/ transaction_validation block_validation

# Run with sanitizers
python3 test_runner.py fuzz/corpus/ --sanitizer asan --max-time 86400

# Run in parallel mode
python3 test_runner.py fuzz/corpus/ --parallel --jobs 4
```

## Sanitizers

### AddressSanitizer (ASAN)
Detects memory errors:
- Use-after-free
- Buffer overflows
- Memory leaks

```bash
./build_with_sanitizers.sh asan
RUSTFLAGS="-Zsanitizer=address" cargo +nightly fuzz run transaction_validation
```

### UndefinedBehaviorSanitizer (UBSAN)
Detects undefined behavior:
- Integer overflow
- Null pointer dereference
- Invalid shifts

```bash
./build_with_sanitizers.sh ubsan
RUSTFLAGS="-Zsanitizer=undefined" cargo +nightly fuzz run transaction_validation
```

### MemorySanitizer (MSAN)
Detects uninitialized memory reads:
- Requires instrumented libstd
- More complex setup

```bash
./build_with_sanitizers.sh msan
RUSTFLAGS="-Zsanitizer=memory" cargo +nightly fuzz run transaction_validation
```

## Corpus Management

### Adding Seeds

```bash
# Add transaction seed
echo "01000000..." > fuzz/corpus/transaction_validation/real_tx.hex

# Add block seed
echo "01000000..." > fuzz/corpus/block_validation/real_block.hex

# Add script seed
echo "76a914..." > fuzz/corpus/script_execution/p2pkh.hex
```

### Corpus Sources

1. **Bitcoin Core Test Vectors**: `bitcoin-core/src/test/data/`
2. **QA Assets**: `bitcoin-core/qa-assets` (if available)
3. **Real Blockchain Data**: Mainnet/testnet transactions and blocks
4. **Historical Blocks**: Pre/post SegWit, Taproot activation blocks

### Corpus Best Practices

- **Diversity**: Include various transaction types, block sizes, script patterns
- **Real-world data**: Prefer actual Bitcoin data over synthetic
- **Size limits**: Keep corpus files reasonable (fuzzer has max_len defaults)
- **Regular updates**: Add new interesting cases as discovered

## Comparison with Bitcoin Core

### Similarities
- ✅ libFuzzer as primary fuzzer
- ✅ Sanitizer support (ASAN, UBSAN, MSAN)
- ✅ Corpus-based approach
- ✅ Test runner for automation
- ✅ Comprehensive target coverage

### Advantages
- ✅ Rust's memory safety (fewer memory errors)
- ✅ Better integration with Kani proofs
- ✅ Property-based testing integration (proptest)

## Metrics

Track these metrics over time:
- **Coverage**: Code coverage percentage
- **Unique crashes**: Crashes found per target
- **Corpus size**: Inputs in corpus
- **Execution rate**: Executions per second
- **Coverage growth**: New coverage over time

## Troubleshooting

### Build Issues

```bash
# Clean build
cargo clean
cargo +nightly fuzz build

# Check Rust version (needs nightly)
rustc +nightly --version
```

### Runtime Issues

```bash
# Check corpus directory exists
ls fuzz/corpus/

# Check target exists
cargo +nightly fuzz list

# Run with verbose output
cargo +nightly fuzz run transaction_validation -- -print_final_stats=1
```

### Sanitizer Issues

```bash
# ASAN requires specific options
export ASAN_OPTIONS="detect_leaks=1:detect_stack_use_after_return=1"
RUSTFLAGS="-Zsanitizer=address" cargo +nightly fuzz run transaction_validation
```

## Differential Fuzzing

The `differential_fuzzing` target tests internal consistency within consensus-proof:

- **Serialization round-trips**: Ensures serialize→deserialize preserves all properties
- **Validation consistency**: Same transaction validates the same way after round-trip
- **Calculation idempotency**: Weight calculations, economic calculations are deterministic
- **Cross-validation**: Different code paths agree on validation results

This does NOT call Bitcoin Core - it tests consensus-proof independently.

```bash
cargo +nightly fuzz run differential_fuzzing
```

## CI Integration

Continuous fuzzing is configured in `.github/workflows/fuzz.yml`:

- **On PRs**: Runs critical targets (transaction, block, script validation) for 5 minutes each
- **On Schedule**: Runs all 12 targets daily at 2 AM UTC
- **Crash Reporting**: Automatically uploads crash artifacts on failures
- **Corpus Management**: Stores corpus between runs on scheduled builds

The CI workflow:
1. Installs Rust nightly and cargo-fuzz
2. Initializes corpus from test vectors
3. Runs fuzzing campaigns with sanitizers
4. Collects and reports crashes
5. Uploads artifacts for analysis

View CI fuzzing results in the GitHub Actions tab.

## References

- [Bitcoin Core Fuzzing Documentation](https://github.com/bitcoin/bitcoin/blob/master/doc/fuzzing.md)
- [libFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
- [Rust Fuzzing Book](https://rust-fuzz.github.io/book/)


