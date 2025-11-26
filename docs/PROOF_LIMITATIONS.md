# Kani Proof Limitations and Bounds

This document describes the current limitations of Kani formal verification proofs in `bllvm-consensus`, including proof bounds and edge cases that are covered by tests rather than formal proofs.

## Overview

Kani proofs use bounded verification to ensure tractability. This means proofs verify properties for inputs up to a certain size, rather than for all possible inputs. This is a standard approach in formal verification to balance completeness with proof performance.

## Current Proof Bounds

### Transaction Bounds

**Location**: `src/kani_helpers.rs` - `proof_limits` module

- **MAX_TX_INPUTS_FOR_PROOF**: 2 inputs
- **MAX_TX_OUTPUTS_FOR_PROOF**: 2 outputs
- **Actual Bitcoin Limits**: 1000 inputs, 1000 outputs

**Rationale**: Transaction proofs focus on correctness of validation logic rather than exhaustive input/output combinations. The logic is proven correct for small transactions, and property-based tests verify larger transactions.

**Coverage**: All transaction validation logic is proven correct for transactions with ≤2 inputs and ≤2 outputs. Edge cases with more inputs/outputs are covered by:
- Property-based tests (proptest)
- Integration tests with real Bitcoin transactions
- Mainnet block tests

### Block Bounds

**Location**: Various proof files

- **MAX_TRANSACTIONS_PER_BLOCK_FOR_PROOF**: 3 transactions
- **Actual Bitcoin Limit**: ~10,000 transactions per block (practical limit based on block size)

**Rationale**: Block proofs verify the correctness of block validation logic. The logic is proven correct for small blocks, and larger blocks are verified through tests.

**Coverage**: Block validation logic is proven correct for blocks with ≤3 transactions. Edge cases with more transactions are covered by:
- Mainnet block tests (real Bitcoin blocks)
- Property-based tests
- Integration tests

### Mempool Bounds

**Location**: `src/kani_helpers.rs`

- **MAX_MEMPOOL_TXS_FOR_PROOF**: 3 transactions
- **Actual Limit**: Effectively unbounded (limited by memory)

**Rationale**: Mempool proofs verify consistency properties. The logic is proven correct for small mempools, and larger mempools are verified through tests.

**Coverage**: Mempool consistency is proven for mempools with ≤3 transactions. Edge cases are covered by:
- Property-based tests
- Stress tests with large mempools
- Integration tests

### Chain Reorganization Bounds

**Location**: `src/reorganization.rs`

- **MAX_CHAIN_LENGTH_FOR_PROOF**: 3-5 blocks
- **Actual Limit**: Effectively unbounded (limited by blockchain length)

**Rationale**: Reorganization proofs verify the correctness of chain reorganization logic. The logic is proven correct for small reorganizations, and larger reorganizations are verified through tests.

**Coverage**: Reorganization logic is proven correct for chains with ≤5 blocks. Edge cases are covered by:
- Historical reorganization tests
- Property-based tests
- Integration tests

### Network Message Bounds

**Location**: `src/network.rs`

- **MAX_HEADERS_FOR_PROOF**: 100 headers
- **MAX_INV_ITEMS_FOR_PROOF**: 100 items
- **Actual Bitcoin Limits**: 2000 headers, 50000 inventory items

**Rationale**: Network message proofs verify boundary checks and size limits. The logic is proven correct for reasonable message sizes, and edge cases are covered by tests.

**Coverage**: Network message validation is proven for messages with ≤100 items. Edge cases are covered by:
- Boundary tests (exactly at limits)
- Stress tests (very large messages)
- Integration tests

## Why Bounds Are Limited

### Proof Tractability

Kani proofs use bounded model checking, which means they explore all possible execution paths up to a certain bound. As bounds increase:

1. **State space grows exponentially**: Each additional element multiplies the state space
2. **Proof time increases**: Larger bounds require more computation
3. **Memory usage increases**: More states need to be tracked

### Current CI Constraints

- **Fast tier**: Must complete in <30 minutes
- **Fast+Medium tier**: Must complete in <2 hours
- **All tier**: Must complete in <6 hours

Current bounds are tuned to ensure proofs complete within these time limits while still providing meaningful verification coverage.

## Edge Case Coverage

Edge cases beyond proof bounds are covered by:

### 1. Property-Based Testing (Proptest)

Property-based tests generate random inputs of various sizes and verify properties hold. These tests complement Kani proofs by:
- Testing larger inputs than proofs can handle
- Discovering edge cases through random generation
- Verifying properties hold across a wide range of inputs

**Location**: `tests/` directory, files with `property` in name

### 2. Mainnet Block Tests

Real Bitcoin mainnet blocks are used to verify correctness with actual transaction patterns and sizes. These tests:
- Use real-world transaction sizes and patterns
- Verify correctness with actual Bitcoin data
- Catch edge cases that might not appear in synthetic tests

**Location**: `tests/mainnet_blocks.rs`

### 3. Integration Tests

Integration tests verify end-to-end correctness with realistic scenarios. These tests:
- Exercise multiple components together
- Use realistic data sizes
- Verify system-level properties

**Location**: `tests/integration/` directory

### 4. Fuzz Testing

Fuzz tests generate random inputs to find bugs. These tests:
- Explore edge cases through random generation
- Find bugs that might not be caught by other tests
- Complement formal verification

**Location**: `fuzz/` directory (if present)

## Increasing Proof Bounds

### When to Increase Bounds

Consider increasing proof bounds when:
1. **Proofs complete quickly**: If current proofs finish in <1 minute, bounds can likely be increased
2. **CI has capacity**: If CI times are well below limits, bounds can be increased
3. **Critical paths identified**: If certain code paths are particularly critical, increase bounds for those proofs

### How to Increase Bounds

1. **Test incrementally**: Start with small increases (e.g., 2→3, 3→5)
2. **Measure proof time**: Track how proof time changes with bound increases
3. **Stop at intractability**: If proof time exceeds 5 minutes, revert to previous bound
4. **Document limits**: Update this document with new bounds and rationale

### Example: Increasing Transaction Input Bound

```rust
// Current (in kani_helpers.rs)
pub const MAX_TX_INPUTS_FOR_PROOF: usize = 2;

// Test increase to 3
pub const MAX_TX_INPUTS_FOR_PROOF: usize = 3;

// Run proof and measure time
// If time < 5 minutes, keep increase
// If time > 5 minutes, revert to 2
```

## Proof Coverage Summary

| Component | Proof Bound | Actual Limit | Coverage Method |
|-----------|-------------|--------------|-----------------|
| Transaction inputs | 2 | 1000 | Proofs + Tests |
| Transaction outputs | 2 | 1000 | Proofs + Tests |
| Block transactions | 3 | ~10,000 | Proofs + Tests |
| Mempool size | 3 | Unbounded | Proofs + Tests |
| Chain length | 3-5 | Unbounded | Proofs + Tests |
| Headers message | 100 | 2000 | Proofs + Tests |
| Inv message | 100 | 50000 | Proofs + Tests |

## Verification Strategy

The verification strategy uses a layered approach:

1. **Kani Proofs**: Verify correctness for bounded inputs (this document)
2. **Property Tests**: Verify properties for larger inputs
3. **Mainnet Tests**: Verify correctness with real Bitcoin data
4. **Integration Tests**: Verify end-to-end correctness
5. **Fuzz Tests**: Find edge cases through random generation

This layered approach ensures comprehensive coverage while maintaining proof tractability.

## Future Improvements

### Potential Bound Increases

As proof performance improves or CI capacity increases, consider:

1. **Transaction bounds**: 2→5 inputs/outputs (if tractable)
2. **Block bounds**: 3→10 transactions (if tractable)
3. **Mempool bounds**: 3→10 transactions (if tractable)
4. **Chain bounds**: 5→10 blocks (if tractable)

### Proof Techniques

Future improvements might include:
- **Unbounded proofs**: For properties that can be proven without bounds
- **Inductive proofs**: For properties that hold by induction
- **Abstract interpretation**: For properties that can be proven abstractly

## References

- [Kani Documentation](https://model-checking.github.io/kani/)
- [Orange Paper](https://github.com/BTCDecoded/orange-paper) - Bitcoin consensus specification
- [Gap Resolution Plan](../GAP_RESOLUTION_PLAN.md) - Implementation plan for addressing proof gaps

