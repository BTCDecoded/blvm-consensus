# Verification Bounds and Coverage

This document describes formal verification coverage in `blvm-consensus` and how tests complement it for edge cases beyond verified bounds.

## Overview

blvm-consensus uses **blvm-spec-lock** (Z3-based) to verify functions against Orange Paper specifications. Spec-lock verifies specific functions and their contracts; property-based tests and integration tests cover larger inputs and edge cases.

## Verification Coverage

### Spec-lock (Orange Paper)

Functions with `#[spec_locked("section")]` are verified against the Orange Paper. See `blvm-spec-lock` and `SPEC_LOCK_COVERAGE.md` for which sections are covered.

### Transaction Bounds

- **Actual Bitcoin Limits**: 1000 inputs, 1000 outputs
- **Coverage**: Transaction validation logic verified via spec-lock where applicable. Edge cases covered by:
  - Property-based tests (proptest)
  - Integration tests with real Bitcoin transactions
  - Mainnet block tests

### Block Bounds

- **Actual Bitcoin Limit**: ~10,000 transactions per block (practical limit based on block size)
- **Coverage**: Block validation logic verified via spec-lock. Edge cases covered by:
  - Mainnet block tests (real Bitcoin blocks)
  - Property-based tests
  - Integration tests

### Mempool Bounds

- **Actual Limit**: Effectively unbounded (limited by memory)
- **Coverage**: Mempool consistency verified via spec-lock where applicable. Edge cases covered by:
  - Property-based tests
  - Stress tests with large mempools
  - Integration tests

## Edge Case Coverage

### 1. Property-Based Testing (Proptest)

Property-based tests generate random inputs of various sizes and verify properties hold. They complement formal verification by:
- Testing larger inputs than formal verification can handle
- Discovering edge cases through random generation
- Verifying properties hold across a wide range of inputs

**Location**: `tests/` directory, files with `property` in name

### 2. Mainnet Block Tests

Real Bitcoin mainnet blocks verify correctness with actual transaction patterns and sizes.

**Location**: `tests/mainnet_blocks.rs`

### 3. Integration Tests

Integration tests verify end-to-end correctness with realistic scenarios.

**Location**: `tests/integration/` directory

### 4. Fuzz Testing

Fuzz tests generate random inputs to find bugs.

**Location**: `fuzz/` directory (if present)

## Verification Strategy

1. **Spec-lock**: Verify functions against Orange Paper specifications
2. **Property Tests**: Verify properties for larger inputs
3. **Mainnet Tests**: Verify correctness with real Bitcoin data
4. **Integration Tests**: Verify end-to-end correctness
5. **Fuzz Tests**: Find edge cases through random generation

## References

- [blvm-spec-lock `SPEC_LOCK_COVERAGE.md`](https://github.com/BTCDecoded/blvm-spec-lock/blob/main/SPEC_LOCK_COVERAGE.md)
- [Orange Paper](https://github.com/BTCDecoded/blvm-spec) - Bitcoin consensus specification
