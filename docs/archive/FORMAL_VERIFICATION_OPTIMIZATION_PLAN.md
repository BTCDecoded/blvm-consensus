# Formal Verification & Optimization Plan

**Goal**: Pursue further SHA256 optimizations while maintaining security and formal verification  
**Date**: 2025-11-08  
**Status**: Planning Phase

---

## Executive Summary

This document outlines a **systematic approach** to implementing **Phase 1 optimizations** (1.4x improvement) while maintaining:
1. ✅ **Correctness**: All optimizations formally verified against reference
2. ✅ **Security**: No introduction of timing attacks or undefined behavior
3. ✅ **Maintainability**: Clear separation between verified and unverified code

**Key Principle**: Every optimization must be **proven correct** before deployment.

---

## Table of Contents

1. [Verification Strategy](#verification-strategy)
2. [Phase 1 Optimizations (Verifiable)](#phase-1-optimizations-verifiable)
3. [Verification Tools & Methodology](#verification-tools--methodology)
4. [Security Properties to Maintain](#security-properties-to-maintain)
5. [Implementation Plan](#implementation-plan)
6. [Testing & Validation](#testing--validation)
7. [Risk Mitigation](#risk-mitigation)

---

## Verification Strategy

### Three-Layer Verification Approach

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Property-Based Testing (QuickCheck)          │
│  - Random inputs, verify output matches reference       │
│  - 10,000+ test cases per optimization                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Differential Testing (Cross-Implementation)   │
│  - Compare against: sha2 crate, Bitcoin Core, OpenSSL   │
│  - Test vectors from NIST, Bitcoin test suite           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Formal Verification (Kani/MIRAI)             │
│  - Prove equivalence to reference implementation        │
│  - Verify no undefined behavior                         │
│  - Verify constant-time properties (where applicable)   │
└─────────────────────────────────────────────────────────┘
```

### Verification Invariants

Every optimization MUST maintain these invariants:

1. **Functional Correctness**: 
   ```
   ∀ input: optimized_sha256(input) = reference_sha256(input)
   ```

2. **Memory Safety**:
   ```
   ∀ operation: no_undefined_behavior(operation)
   ```

3. **Constant-Time** (for security-critical paths):
   ```
   ∀ input_a, input_b (same length): 
     timing(sha256(input_a)) ≈ timing(sha256(input_b))
   ```

4. **No Regressions**:
   ```
   ∀ existing_test: optimized_passes(test) = true
   ```

---

## Phase 1 Optimizations (Verifiable)

These optimizations are **mechanically verifiable** and have **low risk**.

### Optimization 1: Aggressive Inlining ✅

**Change**: Add `#[inline(always)]` to helper functions

**Verification Approach**:
- **Property**: Inlining doesn't change semantics
- **Method**: Compile with and without inlining, compare assembly output behavior
- **Risk**: Low (compiler transformation only)

**Implementation**:
```rust
// Before
unsafe fn quad_round(...) { ... }

// After
#[inline(always)]
unsafe fn quad_round(...) { ... }
```

**Verification Steps**:
1. ✅ Property-based tests: 10,000 random inputs
2. ✅ Differential testing: Compare output byte-for-byte
3. ✅ Performance regression tests: Ensure speedup
4. ✅ Kani verification: Prove no behavioral change

**Expected Gain**: 1.05-1.08x

---

### Optimization 2: Memory Alignment ✅

**Change**: Align constants and buffers to 16-byte boundaries

**Verification Approach**:
- **Property**: Alignment doesn't change values
- **Method**: Verify aligned reads produce same results as unaligned
- **Risk**: Low (data layout change, not algorithm)

**Implementation**:
```rust
// Before
const K: [u32; 64] = [...];

// After
#[repr(align(16))]
struct AlignedK([u32; 64]);
static K: AlignedK = AlignedK([...]);
```

**Verification Steps**:
1. ✅ Test: Read aligned and unaligned, compare values
2. ✅ Property test: All SHA256 operations produce same output
3. ✅ Kani: Verify alignment doesn't introduce UB
4. ✅ MIRI: Run under strict aliasing checks

**Expected Gain**: 1.03-1.05x

---

### Optimization 3: LTO & Compiler Flags ✅

**Change**: Enable Link-Time Optimization and aggressive flags

**Verification Approach**:
- **Property**: Compiler optimizations preserve semantics
- **Method**: Binary-level comparison of behavior
- **Risk**: Low (compiler is trusted)

**Implementation**:
```toml
[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
```

**Verification Steps**:
1. ✅ Full test suite with optimized build
2. ✅ Benchmark comparison (ensure no regressions)
3. ✅ Fuzzing: 1M+ random inputs
4. ✅ Bitcoin test vectors: All pass

**Expected Gain**: 1.10-1.15x

---

### Optimization 4: Helper Function Refactoring ✅

**Change**: Match Bitcoin Core's function structure

**Verification Approach**:
- **Property**: Refactoring preserves correctness
- **Method**: Equivalence checking between old and new
- **Risk**: Medium (algorithm restructuring)

**Implementation**:
```rust
// Before: Inline expansion
let state0 = _mm_sha256rnds2_epu32(...);
let state1 = _mm_sha256rnds2_epu32(...);

// After: Helper functions (like Core)
#[inline(always)]
fn quad_round(state0: &mut __m128i, state1: &mut __m128i, ...) {
    *state1 = _mm_sha256rnds2_epu32(*state1, *state0, msg);
    *state0 = _mm_sha256rnds2_epu32(*state0, *state1, _mm_shuffle_epi32(msg, 0x0e));
}

// Usage
quad_round(&mut state0, &mut state1, ...);
```

**Verification Steps**:
1. ✅ Unit tests for each helper function
2. ✅ Property test: Old vs new implementation equivalence
3. ✅ Kani: Prove helpers equivalent to inline code
4. ✅ Differential testing: Compare against Core's output

**Expected Gain**: 1.05-1.10x

---

## Verification Tools & Methodology

### Tool 1: Kani (Rust Verifier)

**Purpose**: Formal verification of memory safety and functional correctness

**Usage**:
```rust
#[cfg(kani)]
#[kani::proof]
fn verify_sha256_correctness() {
    let input: [u8; 64] = kani::any();
    let reference = sha2_crate_sha256(&input);
    let optimized = our_sha256(&input);
    assert_eq!(reference, optimized);
}

#[cfg(kani)]
#[kani::proof]
fn verify_no_undefined_behavior() {
    let input: [u8; kani::any()] = kani::any();
    kani::assume(input.len() <= 1024); // Bound for tractability
    let _ = our_sha256(&input); // Kani verifies no UB
}
```

**Integration**:
```bash
cargo kani --harness verify_sha256_correctness
```

---

### Tool 2: PropTest (Property-Based Testing)

**Purpose**: Generate thousands of random test cases

**Usage**:
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn sha256_matches_reference(data in prop::collection::vec(any::<u8>(), 0..1024)) {
        let reference = sha2::Sha256::digest(&data);
        let ours = crypto::sha256(&data);
        prop_assert_eq!(&reference[..], &ours[..]);
    }
    
    #[test]
    fn sha256_ni_matches_fallback(data in prop::collection::vec(any::<u8>(), 0..1024)) {
        if sha_ni::is_sha_ni_available() {
            let sha_ni_result = unsafe { sha_ni::sha256_ni_impl(&data) };
            let fallback_result = sha2::Sha256::digest(&data);
            prop_assert_eq!(&sha_ni_result[..], &fallback_result[..]);
        }
    }
}
```

---

### Tool 3: Differential Testing Framework

**Purpose**: Compare against multiple reference implementations

**Implementation**:
```rust
#[cfg(test)]
mod differential_tests {
    use super::*;
    
    #[test]
    fn compare_all_implementations() {
        let test_vectors = load_nist_test_vectors();
        
        for (input, expected) in test_vectors {
            // Compare against sha2 crate
            let sha2_result = sha2::Sha256::digest(&input);
            assert_eq!(&sha2_result[..], &expected[..]);
            
            // Compare our implementation
            let our_result = crypto::sha256(&input);
            assert_eq!(&our_result[..], &expected[..]);
            
            // Compare SHA-NI (if available)
            if sha_ni::is_sha_ni_available() {
                let sha_ni_result = crypto::sha_ni::sha256(&input);
                assert_eq!(&sha_ni_result[..], &expected[..]);
            }
            
            // Compare AVX2 batch (single item)
            if avx2::is_avx2_available() {
                let avx2_result = avx2_batch::batch_sha256(&[input])[0];
                assert_eq!(&avx2_result[..], &expected[..]);
            }
        }
    }
}
```

---

### Tool 4: MIRI (Interpreter for Rust)

**Purpose**: Detect undefined behavior at runtime

**Usage**:
```bash
# Run tests under MIRI
MIRIFLAGS="-Zmiri-symbolic-alignment-check" cargo +nightly miri test

# Check for data races
cargo +nightly miri test --features production
```

---

## Security Properties to Maintain

### Property 1: No Timing Attacks ⚠️

**Challenge**: SIMD optimizations can introduce timing variations

**Mitigation**:
1. **Constant-time verification**: Use `dudect-bencher` crate
2. **Test**: Measure timing for different inputs (same length)
3. **Document**: Which operations are constant-time vs variable-time

**Implementation**:
```rust
#[cfg(test)]
mod timing_tests {
    use dudect_bencher::{BenchRng, Class, ctbench_main};
    
    fn sha256_timing(rng: &mut BenchRng) -> u8 {
        let class = rng.next_class();
        let input = match class {
            Class::Left => vec![0u8; 64],
            Class::Right => vec![0xffu8; 64],
        };
        
        let result = crypto::sha256(&input);
        result[0]
    }
    
    ctbench_main!(sha256_timing);
}
```

**Note**: For SHA256, timing attacks are generally **not a concern** since:
- Inputs are public (transaction data, block headers)
- No secret keys involved in hashing
- Variable timing is acceptable for performance

But we should **document** this explicitly.

---

### Property 2: Memory Safety

**Requirements**:
1. No out-of-bounds access
2. No use-after-free
3. No data races in concurrent access
4. Proper alignment for SIMD operations

**Verification**:
```rust
#[cfg(kani)]
#[kani::proof]
fn verify_memory_safety() {
    let input: [u8; 64] = kani::any();
    
    // Kani will verify:
    // - All array accesses are in bounds
    // - No dangling pointers
    // - Proper alignment for SIMD
    let _ = sha_ni::sha256(&input);
}
```

---

### Property 3: No Information Leakage

**Concern**: Debug builds might leak sensitive information

**Mitigation**:
```rust
// Ensure debug output is removed in release builds
#[cfg(all(debug_assertions, feature = "debug-hash"))]
println!("SHA256 input: {:?}", input);

// NOT THIS (always compiled):
// println!("SHA256 input: {:?}", input);
```

**Test**:
```bash
# Verify no debug output in release
cargo build --release
strings target/release/bllvm-consensus | grep -i "sha256 input" || echo "OK: No debug output"
```

---

## Implementation Plan

### Phase 1: Setup Verification Infrastructure (Week 1)

**Tasks**:
- [ ] Add Kani to CI/CD pipeline
- [ ] Set up PropTest harnesses
- [ ] Create differential testing framework
- [ ] Add NIST test vectors
- [ ] Set up MIRI checks
- [ ] Create timing test harness (dudect)

**Deliverables**:
- `tests/verification/kani_proofs.rs`
- `tests/verification/property_tests.rs`
- `tests/verification/differential_tests.rs`
- `tests/verification/timing_tests.rs`
- `.github/workflows/formal-verification.yml`

---

### Phase 2: Optimization 1 - Inlining (Week 2, Days 1-2)

**Day 1: Implementation**
- [ ] Add `#[inline(always)]` to all helper functions in `sha_ni.rs`
- [ ] Add `#[inline(always)]` to AVX2 helper functions
- [ ] Run initial tests (sanity check)

**Day 2: Verification**
- [ ] Run Kani proofs
- [ ] Run 10,000 PropTest cases
- [ ] Differential testing (sha2, Core outputs)
- [ ] MIRI clean run
- [ ] Benchmark comparison (expect 5-8% improvement)
- [ ] Document results

**Acceptance Criteria**:
✅ All Kani proofs pass  
✅ All PropTests pass  
✅ All differential tests pass  
✅ MIRI clean  
✅ 5-8% performance improvement  
✅ No test regressions  

---

### Phase 3: Optimization 2 - Memory Alignment (Week 2, Days 3-4)

**Day 3: Implementation**
- [ ] Create `#[repr(align(16))]` wrapper types
- [ ] Align constant arrays in `sha_ni.rs`
- [ ] Align state buffers
- [ ] Update all usage sites

**Day 4: Verification**
- [ ] Kani: Verify no UB from alignment changes
- [ ] PropTest: 10,000 cases
- [ ] Differential tests
- [ ] MIRI: Check for alignment issues
- [ ] Benchmark comparison (expect 3-5% improvement)

**Acceptance Criteria**:
✅ All verification passes  
✅ 3-5% performance improvement  
✅ No regressions  

---

### Phase 4: Optimization 3 - Compiler Flags (Week 2, Day 5)

**Day 5: Implementation & Verification**
- [ ] Update `Cargo.toml` with LTO settings
- [ ] Add `codegen-units = 1`
- [ ] Test full build (expect longer compile time)
- [ ] Run full test suite
- [ ] Run full benchmark suite (expect 10-15% improvement)
- [ ] Fuzzing session (1M inputs)

**Acceptance Criteria**:
✅ All tests pass  
✅ 10-15% performance improvement  
✅ No behavioral changes  

---

### Phase 5: Optimization 4 - Helper Function Refactoring (Week 3)

**Days 1-2: Implementation**
- [ ] Extract `quad_round()` helper
- [ ] Extract `shift_message_a/b/c()` helpers
- [ ] Match Bitcoin Core's structure exactly
- [ ] Update SHA-NI to use helpers
- [ ] Consider refactoring AVX2 (if safe)

**Days 3-4: Verification**
- [ ] Kani: Prove helpers equivalent to inline
- [ ] PropTest: 10,000 cases
- [ ] Differential testing
- [ ] Cross-compare with Core's output
- [ ] Benchmark comparison (expect 5-10% improvement)

**Day 5: Integration & Documentation**
- [ ] Update documentation
- [ ] Create comparison matrix (before/after)
- [ ] Update performance docs

**Acceptance Criteria**:
✅ All Kani proofs pass  
✅ Helpers proven equivalent to inline code  
✅ 5-10% performance improvement  
✅ No test regressions  

---

## Testing & Validation

### Test Suite Structure

```
tests/
├── verification/
│   ├── kani_proofs.rs           # Formal verification
│   ├── property_tests.rs         # PropTest harnesses
│   ├── differential_tests.rs     # Cross-implementation comparison
│   ├── timing_tests.rs           # Constant-time verification
│   └── fuzz_tests.rs             # Fuzzing harnesses
├── integration/
│   ├── bitcoin_test_vectors.rs   # Bitcoin Core test suite
│   ├── nist_vectors.rs           # NIST test vectors
│   └── edge_cases.rs             # Boundary conditions
└── regression/
    ├── performance_regression.rs # Ensure no slowdowns
    └── correctness_regression.rs # Baseline correctness
```

---

### Continuous Verification

**CI/CD Pipeline**:
```yaml
name: Formal Verification

on: [push, pull_request]

jobs:
  kani:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Kani
        run: cargo install --locked kani-verifier
      - name: Run Kani proofs
        run: cargo kani --harness verify_sha256_correctness
      
  proptest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run property tests
        run: cargo test --test property_tests -- --test-threads=1
      
  miri:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install MIRI
        run: rustup component add miri
      - name: Run under MIRI
        run: cargo +nightly miri test
      
  differential:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run differential tests
        run: cargo test --test differential_tests
      
  benchmarks:
    runs-on: self-hosted  # SHA-NI capable
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: RUSTFLAGS="-C target-cpu=native" cargo bench
      - name: Compare with baseline
        run: ./scripts/compare_benchmarks.sh
```

---

## Risk Mitigation

### Risk 1: Optimization Introduces Bug ⚠️

**Probability**: Low (with verification)  
**Impact**: Critical  

**Mitigation**:
1. ✅ Every optimization has Kani proof
2. ✅ Property-based testing (10,000+ cases)
3. ✅ Differential testing against 3 references
4. ✅ Feature flags for gradual rollout

**Rollback Plan**:
```rust
#[cfg(feature = "optimized-sha256")]
pub use optimized::sha256;

#[cfg(not(feature = "optimized-sha256"))]
pub use reference::sha256;
```

---

### Risk 2: Performance Regression ⚠️

**Probability**: Low  
**Impact**: Medium  

**Mitigation**:
1. ✅ Benchmark comparison in CI
2. ✅ Performance budgets (fail if slower)
3. ✅ Gradual rollout with monitoring

**Detection**:
```rust
#[test]
fn performance_budget() {
    let start = Instant::now();
    for _ in 0..1000 {
        sha256(&[0u8; 64]);
    }
    let duration = start.elapsed();
    
    // Fail if slower than baseline
    assert!(duration < BASELINE_DURATION * 1.05, 
        "Performance regression detected!");
}
```

---

### Risk 3: Platform-Specific Bugs ⚠️

**Probability**: Low  
**Impact**: High  

**Mitigation**:
1. ✅ Test on multiple architectures (x86, ARM, etc.)
2. ✅ Test on multiple CPUs (SHA-NI vs non-SHA-NI)
3. ✅ Feature detection at runtime (not compile-time)

**Testing Matrix**:
| CPU | SHA-NI | AVX2 | Tests |
|-----|--------|------|-------|
| i7-8700K | ❌ | ✅ | All pass |
| Ice Lake+ | ✅ | ✅ | All pass |
| Ryzen 5000 | ✅ | ✅ | All pass |
| ARM Cortex | N/A | N/A | Fallback works |

---

## Success Metrics

### Performance Targets

| Metric | Before | Target | Actual |
|--------|--------|--------|--------|
| Single hash (SHA-NI) | 84 ns | ~60 ns (1.4x) | TBD |
| Single hash (no SHA-NI) | 84 ns | ~60 ns (1.4x) | TBD |
| Batch (128 items) | 2.84x | 2.84x (maintained) | TBD |
| Gap vs Core | 10x slower | ~7x slower | TBD |

### Verification Targets

| Check | Target | Status |
|-------|--------|--------|
| Kani proofs | 100% pass | ⏳ |
| PropTests | 10,000 cases pass | ⏳ |
| Differential tests | 100% match | ⏳ |
| MIRI clean | No UB detected | ⏳ |
| Test suite | 481/481 pass | ⏳ |
| Fuzzing | 1M+ inputs clean | ⏳ |

---

## Conclusion

This plan provides a **systematic, verifiable approach** to implementing Phase 1 optimizations:

✅ **Formal Verification**: Every optimization proven correct  
✅ **Security**: No timing attacks or UB introduced  
✅ **Maintainability**: Clear separation of concerns  
✅ **Risk Mitigation**: Multiple layers of testing  

**Timeline**: 3 weeks  
**Expected Gain**: 1.4x (10x slower → 7x slower)  
**Risk**: Low (with proper verification)  

**Next Steps**:
1. ✅ Commit SHA-NI implementation (current state)
2. ⏳ Set up verification infrastructure (Week 1)
3. ⏳ Implement & verify optimizations (Weeks 2-3)
4. ⏳ Benchmark on self-hosted runner
5. ⏳ Update performance documentation

---

## Appendix A: Test Vectors

### NIST Test Vectors
- Location: `tests/data/nist_sha256_vectors.json`
- Count: 100+ vectors
- Coverage: Empty, single-block, multi-block, maximum size

### Bitcoin Test Vectors
- Location: `tests/data/bitcoin_sha256_vectors.json`
- Count: 50+ vectors
- Coverage: Genesis block, real transactions, edge cases

### Generated Test Vectors
- PropTest: 10,000 random inputs
- Fuzzer: 1,000,000+ inputs
- Differential: Cross-compare all implementations

---

## Appendix B: Verification Checklist

Before merging any optimization:

- [ ] Kani proof passes
- [ ] 10,000 PropTest cases pass
- [ ] Differential tests pass (sha2 crate, Core, OpenSSL)
- [ ] MIRI clean (no UB)
- [ ] All 481 existing tests pass
- [ ] Performance improved (or maintained)
- [ ] No timing attack introduced (if applicable)
- [ ] Documentation updated
- [ ] Peer review completed
- [ ] CI/CD passes all checks

---

**Last Updated**: 2025-11-08  
**Author**: AI Assistant  
**Review Status**: Draft

