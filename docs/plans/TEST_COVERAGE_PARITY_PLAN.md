# Test Coverage Parity Plan: consensus-proof vs Bitcoin Core

**Date**: 2024-11-03  
**Status**: ✅ Validated & Ready for Implementation

## Executive Summary

**Goal**: Match or exceed Core's test coverage while maintaining our formal verification advantage.

**Current Status**:
- ✅ 267+ Kani proofs (Core: 0) - **Our unique advantage**
- ✅ 115+ property tests (Core: 0) - **Our unique advantage**
- ✅ ~344 unit test functions (Core: ~711 test cases)
- ✅ 7 fuzz targets (Core: 20+)
- ⚠️ Core test vector execution (infrastructure ready, blocked by Cargo.lock)
- ⚠️ Real-world block dataset (infrastructure ready, needs more blocks)

## Priority 1: Critical — Execute Core Test Vectors (1-2 days)

**Status**: Infrastructure 100% complete, execution blocked by Cargo.lock

**Tasks**:
1. Resolve Cargo.lock compatibility
   - Check `Cargo.lock` version
   - Update dependencies if needed
   - Test execution: `cargo test --test core_test_vectors::integration_test`

2. Execute and document
   - Run all test vectors (tx_valid: 528, tx_invalid: 397)
   - Document results in `VERIFICATION_RESULTS.md`
   - Fix any failures to match Core behavior

3. CI integration
   - Add test vector execution to CI
   - Make non-blocking initially
   - Track pass rates

**Impact**: Validates network compatibility - **CRITICAL**

## Priority 2: High — Real-World Block Dataset (2-3 days)

**Status**: Infrastructure ready, dataset minimal

**Tasks**:
1. Download historical blocks (automated)
   - Expand `download_mainnet_blocks.sh`
   - Download 50-100 blocks across eras:
     - Every halving block (0, 210000, 420000, 630000, 840000)
     - Major activation blocks (SegWit, Taproot)
     - Edge cases (CVE blocks, max size, large witness)
     - 10 blocks per era (pre-SegWit, SegWit, Taproot)

2. Expand mainnet block tests
   - Automated test generation for downloaded blocks
   - Transaction pattern analysis
   - UTXO consistency checks

3. Historical replay expansion
   - Incremental replay (1000 blocks at a time)
   - Checkpoint verification
   - Performance tracking

**Impact**: Matches Core's real-world validation coverage

## Priority 3: High — Unit Test Expansion (4-5 days)

**Status**: ~344 unit tests exist, need to match Core's specific cases

**Gap Analysis**:
- Core has 711 unit test cases across 130 files
- We have ~344 unit test functions
- Need to match Core's specific test patterns

**Tasks**:
1. Analyze Core's test structure
   - Map Core's test cases to our modules
   - Identify missing test patterns
   - Document coverage gaps

2. Create test case mapping document
   - Track which Core tests we've matched
   - Track which tests we've added beyond Core
   - Identify priority gaps

3. Implement missing test patterns
   - Transaction tests: Match Core's `basic_transaction_tests`, `test_big_witness_transaction`
   - Script tests: Match Core's `script_build`, `script_PushData`, `script_combineSigs`
   - Mempool tests: Match Core's `MempoolRemoveTest`, ancestor indexing
   - Validation tests: Match Core's `block_subsidy_test`, `subsidy_limit_test`

4. Add our unique tests
   - Edge cases Core doesn't cover
   - Property-based test cases
   - Formal verification test cases

**Impact**: Matches Core's unit test coverage

## Priority 4: Medium — Fuzz Target Expansion (2-3 days)

**Status**: 7 fuzz targets exist, Core has 20+

**Current Fuzz Targets**:
- ✅ block_validation.rs
- ✅ transaction_validation.rs
- ✅ script_execution.rs
- ✅ mempool_operations.rs
- ✅ utxo_commitments.rs
- ✅ segwit_validation.rs
- ✅ compact_block_reconstruction.rs

**Tasks**:
1. Add missing fuzz targets
   - Serialization/deserialization (Core has `deserialize`)
   - UTXO set hashing (Core has `muhash`)
   - Script assets (Core has `script_assets`)
   - Network messages (Core has network fuzzing)
   - Merkle tree operations
   - Transaction package validation

2. Improve existing targets
   - Expand corpus with real-world seeds
   - Add coverage tracking
   - CI integration (non-blocking)

**Impact**: Matches Core's fuzzing coverage

## Priority 5: Low — Property Test Expansion (1-2 days)

**Status**: 115+ property tests (Core has 0) - **Already ahead**

**Tasks**:
1. Expand to 200+ property tests
   - Add transaction serialization round-trip tests
   - Add block structure invariant tests
   - Add script execution property tests
   - Add economic model property tests

2. Document property test coverage
   - Create property test coverage report
   - Track which areas are covered
   - Identify gaps

**Impact**: Maintains our advantage

## Priority 6: Low — Integration Test Expansion (1-2 days)

**Status**: 9 integration modules exist

**Tasks**:
1. Add consensus feature integration tests
   - SegWit activation sequence
   - Taproot activation sequence
   - BIP compliance integration

2. Add mempool integration tests
   - Mempool persistence
   - Package limits
   - RBF behavior

**Impact**: Approaching Core's functional test coverage

## Priority 7: Low — Test Maturity (1 day)

**Status**: Documentation and tracking needed

**Tasks**:
1. Create test coverage dashboard
   - Track test counts vs Core
   - Track coverage metrics
   - Identify gaps

2. Extract Core test cases
   - Document known bug test cases
   - Add regression tests

**Impact**: Improves test maturity tracking

## Implementation Timeline

| Priority | Phase | Duration | Status |
|----------|-------|----------|--------|
| P1 | Execute Core Test Vectors | 1-2 days | ⏳ In Progress |
| P2 | Real-World Block Dataset | 2-3 days | ⏳ Pending |
| P3 | Unit Test Expansion | 4-5 days | ⏳ Pending |
| P4 | Fuzz Target Expansion | 2-3 days | ⏳ Pending |
| P5 | Property Test Expansion | 1-2 days | ⏳ Pending |
| P6 | Integration Test Expansion | 1-2 days | ⏳ Pending |
| P7 | Test Maturity | 1 day | ⏳ Pending |
| **Total** | **All Priorities** | **12-18 days** | ⏳ Planning |

## Success Criteria

1. ✅ Core test vectors: 100% pass rate (after Cargo.lock fix)
2. ✅ Real-world blocks: 50-100 blocks validated across eras
3. ✅ Unit tests: Match Core's 711 cases + add 200+ additional
4. ✅ Fuzz targets: 15+ targets (approaching Core's 20+)
5. ✅ Property tests: 200+ tests (already ahead)
6. ✅ Integration tests: 12+ modules (approaching Core's functional tests)

## Key Adjustments Made

1. **Realistic timelines**: 12-18 days total (down from 17-26)
2. **Acknowledged existing work**: ~344 unit tests, 7 fuzz targets
3. **Focused on gaps**: Core test execution, block dataset, fuzz expansion
4. **Maintained advantages**: Formal verification, property tests
5. **Prioritized execution**: Core test vectors first (blocked)

## Immediate Next Steps

1. ✅ Resolve Cargo.lock compatibility (P1 blocker)
2. ✅ Execute Core test vectors (P1)
3. ⏳ Download 50-100 real-world blocks (P2)
4. ⏳ Create test case mapping document (P3)

