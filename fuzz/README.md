# blvm-consensus fuzz (cargo-fuzz)

Coverage-guided fuzzing with **libFuzzer**. **Harness names** are defined in **`Cargo.toml`** (`[[bin]]`); do not trust READMEs for a full list.

Narrative overview (timeless): [blvm-docs: Fuzzing](https://github.com/BTCDecoded/blvm-docs/blob/main/src/development/fuzzing.md).

## Quick Start

### 1. Initialize Corpus

```bash
cd blvm-consensus/fuzz
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

1. **Public JSON test vectors** (same families used across the ecosystem): see `blvm-consensus/docs/TEST_DATA_SOURCES.md`
2. **QA Assets**: `bitcoin-core/qa-assets` (if available)
3. **Real Blockchain Data**: Mainnet/testnet transactions and blocks
4. **Historical Blocks**: Pre/post SegWit, Taproot activation blocks

### Corpus Best Practices

- **Diversity**: Include various transaction types, block sizes, script patterns
- **Real-world data**: Prefer actual Bitcoin data over synthetic
- **Size limits**: Keep corpus files reasonable (fuzzer has max_len defaults)
- **Regular updates**: Add new interesting cases as discovered

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

## Internal differential harness

The `differential_fuzzing` binary checks consistency inside this crate (e.g. serialize/deserialize round-trips vs validation). It does not call an external node.

```bash
cargo +nightly fuzz run differential_fuzzing
```

## CI

Where fuzzing runs in CI, schedules and target matrices are defined in the workflow YAML for that repo—read the file; do not rely on this README for exact steps.

## References

- [libFuzzer](https://llvm.org/docs/LibFuzzer.html)
- [Rust Fuzz Book](https://rust-fuzz.github.io/book/)


