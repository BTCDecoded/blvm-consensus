# Benchmarking Improvements

This document describes the improvements made to make the consensus-proof codebase more conducive to benchmarking.

## Problem Statement

The original code had several issues that made benchmarking difficult:

1. **Global static caches persist across benchmark runs** - Script and hash caches would accumulate state between runs, making results inconsistent
2. **Thread-local state causes warmup effects** - First use in a thread has initialization overhead
3. **Environment variable dependency** - `get_assume_valid_height()` reads from environment, making results non-reproducible
4. **No way to reset state** - No public API to clear caches or reset thread-local pools

## Solutions Implemented

### 1. Cache Disabling (`disable_caching`)

Added ability to completely disable caching for benchmarking:

```rust
use consensus_proof::disable_caching;

// Disable caches for consistent benchmarks
disable_caching(true);
// Run benchmarks...
disable_caching(false); // Re-enable
```

**Location**: `consensus-proof/src/script.rs`

### 2. Cache Clearing Functions

Added functions to clear caches between benchmark runs:

```rust
use consensus_proof::{clear_script_cache, clear_hash_cache, clear_all_caches, clear_sighash_templates};

// Clear individual caches
clear_script_cache();
clear_hash_cache();
clear_sighash_templates(); // Sighash templates cache

// Or clear all at once
clear_all_caches();
```

**Location**: 
- `consensus-proof/src/script.rs` (script and hash caches)
- `consensus-proof/src/transaction_hash.rs` (sighash templates cache)

**Note**: `clear_sighash_templates()` is currently a no-op because the cache isn't populated yet, but provided for API consistency.

### 3. Thread-Local Pool Clearing

Added function to clear thread-local stack pools:

```rust
use consensus_proof::clear_stack_pool;

// Clear pool to reset allocation state
clear_stack_pool();
```

**Location**: `consensus-proof/src/script.rs`

### 4. Assume-Valid Height Override

Added ability to override `assume_valid_height` without environment variables:

```rust
use consensus_proof::{set_assume_valid_height, reset_assume_valid_height};

// Set to validate all blocks (no skipping)
set_assume_valid_height(0);
// Run benchmarks...
reset_assume_valid_height(); // Reset to use environment
```

**Location**: `consensus-proof/src/block.rs`

### 5. Reset All State

Convenience function to reset everything:

```rust
use consensus_proof::reset_benchmarking_state;

// Reset all caches, pools, and settings
reset_benchmarking_state();
```

**Location**: `consensus-proof/src/script.rs`

## Feature Flag

All benchmarking utilities are gated behind the `benchmarking` feature flag:

```toml
# In Cargo.toml
[features]
benchmarking = []
```

**Usage**: `cargo bench --features production,benchmarking`

## Example Usage

```rust
use consensus_proof::{
    clear_all_caches, disable_caching, reset_benchmarking_state,
    set_assume_valid_height,
};

// Before benchmark run
reset_benchmarking_state(); // Clear all state
set_assume_valid_height(0); // Validate all blocks
disable_caching(true); // Disable caches for consistent results

// Run benchmarks...

// After benchmark run
disable_caching(false); // Re-enable caches
reset_assume_valid_height(); // Reset to environment
```

## Benefits

1. **Reproducible benchmarks** - No cache state pollution between runs
2. **Consistent results** - Can disable caches to measure raw performance
3. **Configurable validation** - Can test different assume-valid configurations
4. **Easy state management** - Single function to reset everything

## Implementation Details

- All benchmarking functions are `#[cfg(all(feature = "production", feature = "benchmarking"))]`
- Cache disabling uses `AtomicBool` for thread-safe flag
- Assume-valid override uses `AtomicU64` with `u64::MAX` as "not set" sentinel
- All functions are zero-cost when benchmarking feature is disabled

## Notes

- These functions are **only available with both `production` and `benchmarking` features enabled**
- They have **no effect on production code** when benchmarking feature is disabled
- Cache disabling is **thread-safe** and affects all threads
- Assume-valid override is **thread-safe** and affects all threads

