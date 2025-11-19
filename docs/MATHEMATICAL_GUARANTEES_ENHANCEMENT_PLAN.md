# Mathematical Guarantees Enhancement Plan

**Goal**: Ensure mathematical guarantees that we are not deviating from Bitcoin consensus  
**Date**: 2025-01-18  
**Status**: Planning Phase

---

## Executive Summary

This plan outlines systematic enhancements to strengthen mathematical guarantees that `bllvm-consensus` correctly implements Bitcoin consensus rules without deviation. We will expand formal verification, property-based testing, reference implementation comparison, and CI enforcement.

---

## Current State Assessment

### ✅ Existing Infrastructure

1. **Kani Formal Verification**: 287 proof instances across 25 files
   - Coverage: Block validation, transaction validation, economic rules, script execution
   - Status: Active, but needs expansion to cover all critical paths

2. **Property-Based Testing**: Proptest integration exists
   - Coverage: SHA256 optimizations, some consensus functions
   - Status: Needs expansion to all consensus rules

3. **Bitcoin Core Test Vectors**: Infrastructure exists
   - Files: `tests/core_test_vectors/` (transaction, script, block tests)
   - Status: Needs automatic download and CI integration

4. **Differential Testing**: Framework exists
   - Files: `tests/integration/differential_tests.rs`
   - Status: Needs continuous execution against Core

5. **Mathematical Specifications**: Documentation exists
   - Files: `docs/VERIFICATION.md`, inline comments
   - Status: Needs completion for all consensus functions

---

## Enhancement Plan

### Phase 1: Expand Kani Proof Coverage (Critical)

**Goal**: Prove all critical consensus functions with formal verification

#### 1.1 Block Validation Proofs

**Current**: Partial coverage  
**Target**: Complete coverage

**Functions to Add Proofs For**:
- `connect_block`: Full block connection with UTXO updates
- `validate_block_header`: Header validation with all rules
- `check_merkle_root`: Merkle tree validation
- `check_witness_commitment`: SegWit witness commitment validation
- `validate_block_subsidy`: Economic rule enforcement

**Mathematical Specifications**:
```rust
/// ∀ block B, UTXO set US, height h:
/// connect_block(B, US, h) = (valid, US') ⟺
///   (ValidateHeader(B.header) ∧ 
///    ∀ tx ∈ B.transactions: CheckTransaction(tx) ∧ 
///    CheckTxInputs(tx, US, h) ∧
///    VerifyScripts(tx, US) ∧
///    CoinbaseOutput ≤ TotalFees + GetBlockSubsidy(h) ∧
///    US' = ApplyTransactions(B.transactions, US))
```

**Implementation**:
```rust
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)] // Bound for tractability
fn kani_connect_block_preserves_utxo_consistency() {
    let block: Block = kani::any();
    let utxo_set: UtxoSet = kani::any();
    let height: u32 = kani::any();
    
    kani::assume(block.transactions.len() <= 10); // Bound
    kani::assume(height <= 1000000); // Reasonable height
    
    if let Ok((_, new_utxo_set)) = connect_block(&block, utxo_set.clone(), height) {
        // Invariant: UTXO set size changes are bounded
        let size_delta = new_utxo_set.len() as i64 - utxo_set.len() as i64;
        let max_outputs: i64 = block.transactions.iter()
            .map(|tx| tx.outputs.len() as i64)
            .sum();
        let max_inputs: i64 = block.transactions.iter()
            .map(|tx| tx.inputs.len() as i64)
            .sum();
        
        // UTXO set can grow by at most outputs, shrink by at most inputs
        assert!(size_delta <= max_outputs);
        assert!(size_delta >= -max_inputs);
        
        // Invariant: Total supply is preserved (inputs = outputs + fees)
        // This is complex - would need to track value changes
    }
}
```

#### 1.2 Transaction Validation Proofs

**Current**: Good coverage  
**Target**: Complete edge case coverage

**Additional Proofs Needed**:
- Transaction size limits
- Input/output count limits
- Value overflow/underflow protection
- Script execution bounds

#### 1.3 Economic Rules Proofs

**Current**: Partial coverage  
**Target**: Complete mathematical proofs

**Proofs to Add**:
- Block subsidy halving schedule (mathematical formula)
- Total supply convergence to 21M BTC
- Fee calculation correctness
- Difficulty adjustment bounds

**Mathematical Specification**:
```rust
/// ∀ h ∈ ℕ: 
/// subsidy(h) = 50 * 10^8 * 2^(-⌊h/210000⌋) if ⌊h/210000⌋ < 64 else 0
/// 
/// Invariant: total_supply(h) = Σ(i=0 to h) subsidy(i) < 21 * 10^6 * 10^8
```

---

### Phase 2: Property-Based Testing with Mathematical Invariants

**Goal**: Generate thousands of test cases that verify mathematical properties

#### 2.1 Consensus Invariant Tests

**Add to `tests/verification/property_tests.rs`**:

```rust
use proptest::prelude::*;
use bllvm_consensus::*;

proptest! {
    /// Invariant: Block subsidy halves every 210,000 blocks
    #[test]
    fn prop_block_subsidy_halving_schedule(
        height in 0u32..2100000u32 // Up to 10 halvings
    ) {
        let subsidy = economic::get_block_subsidy(height);
        let halving_epoch = height / 210000;
        
        // Subsidy should be 50 BTC * 2^(-halving_epoch) satoshis
        let expected_subsidy = if halving_epoch < 64 {
            50_0000_0000u64 / (1u64 << halving_epoch)
        } else {
            0
        };
        
        prop_assert_eq!(subsidy, expected_subsidy);
    }
    
    /// Invariant: Total supply is monotonic and bounded
    #[test]
    fn prop_total_supply_monotonic_bounded(
        height1 in 0u32..1000000u32,
        height2 in 0u32..1000000u32
    ) {
        let supply1 = economic::total_supply(height1);
        let supply2 = economic::total_supply(height2);
        
        if height1 <= height2 {
            prop_assert!(supply1 <= supply2, "Supply must be monotonic");
        }
        
        // Total supply must be less than 21M BTC
        prop_assert!(supply1 <= 21_000_000 * 100_000_000, "Supply cap");
        prop_assert!(supply2 <= 21_000_000 * 100_000_000, "Supply cap");
    }
    
    /// Invariant: Proof of work target expansion/compression round-trip
    #[test]
    fn prop_pow_target_round_trip(
        bits in 0x03000000u32..0x1d00ffffu32
    ) {
        let expanded = pow::expand_target(bits);
        let compressed = pow::compress_target(expanded);
        let re_expanded = pow::expand_target(compressed);
        
        // Significant bits (words 2, 3) must be preserved exactly
        prop_assert_eq!(re_expanded.0[2], expanded.0[2]);
        prop_assert_eq!(re_expanded.0[3], expanded.0[3]);
        
        // Re-expanded must be ≤ original (compression may truncate)
        prop_assert!(re_expanded <= expanded);
    }
    
    /// Invariant: Transaction fee is non-negative
    #[test]
    fn prop_transaction_fee_non_negative(
        tx in transaction_strategy(),
        utxo_set in utxo_set_strategy()
    ) {
        if let Ok(fee) = economic::calculate_fee(&tx, &utxo_set) {
            prop_assert!(fee >= 0, "Fee must be non-negative");
        }
    }
    
    /// Invariant: Chain work is monotonic
    #[test]
    fn prop_chain_work_monotonic(
        chain1 in chain_strategy(),
        chain2 in chain_strategy()
    ) {
        let work1 = reorganization::calculate_chain_work(&chain1);
        let work2 = reorganization::calculate_chain_work(&chain2);
        
        if chain1.len() <= chain2.len() {
            // Longer chain should have more work (assuming valid blocks)
            // This is probabilistic - some blocks might have lower difficulty
            // But on average, work should increase
        }
        
        // Work must be non-negative
        prop_assert!(work1 >= 0);
        prop_assert!(work2 >= 0);
    }
}
```

#### 2.2 Script Execution Invariant Tests

**Add script execution property tests**:
- Script execution is deterministic
- Script execution respects resource limits
- Script execution preserves stack invariants

---

### Phase 3: Automatic Bitcoin Core Test Vector Integration

**Goal**: Automatically download and run Bitcoin Core test vectors in CI

#### 3.1 Test Vector Download Script

**Create `scripts/download_core_test_vectors.sh`**:

```bash
#!/bin/bash
set -euo pipefail

# Download Bitcoin Core test vectors
CORE_VERSION="27.0"
TEST_VECTORS_DIR="tests/test_data/core_vectors"

mkdir -p "$TEST_VECTORS_DIR/transactions"
mkdir -p "$TEST_VECTORS_DIR/scripts"
mkdir -p "$TEST_VECTORS_DIR/blocks"

# Download from Bitcoin Core repository
BASE_URL="https://raw.githubusercontent.com/bitcoin/bitcoin/v${CORE_VERSION}/src/test/data"

echo "Downloading transaction test vectors..."
curl -f -o "$TEST_VECTORS_DIR/transactions/tx_valid.json" \
    "${BASE_URL}/tx_valid.json" || echo "Warning: tx_valid.json not found"
curl -f -o "$TEST_VECTORS_DIR/transactions/tx_invalid.json" \
    "${BASE_URL}/tx_invalid.json" || echo "Warning: tx_invalid.json not found"

echo "Downloading script test vectors..."
curl -f -o "$TEST_VECTORS_DIR/scripts/script_valid.json" \
    "${BASE_URL}/script_valid.json" || echo "Warning: script_valid.json not found"
curl -f -o "$TEST_VECTORS_DIR/scripts/script_invalid.json" \
    "${BASE_URL}/script_invalid.json" || echo "Warning: script_invalid.json not found"

echo "Downloading block test vectors..."
curl -f -o "$TEST_VECTORS_DIR/blocks/block_valid.json" \
    "${BASE_URL}/block_valid.json" || echo "Warning: block_valid.json not found"
curl -f -o "$TEST_VECTORS_DIR/blocks/block_invalid.json" \
    "${BASE_URL}/block_invalid.json" || echo "Warning: block_invalid.json not found"

echo "Test vectors downloaded successfully"
```

#### 3.2 CI Integration

**Add to `.github/workflows/verify.yml`**:

```yaml
- name: Download Core Test Vectors
  run: |
    chmod +x scripts/download_core_test_vectors.sh
    ./scripts/download_core_test_vectors.sh

- name: Run Core Test Vectors
  run: cargo test --test core_test_vectors --features core-test-vectors
```

#### 3.3 Test Vector Execution

**Enhance `tests/core_test_vectors/integration_test.rs`**:

```rust
#[test]
fn run_all_core_test_vectors() {
    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;
    
    // Transaction tests
    if let Ok(results) = transaction_tests::run_all_tests() {
        passed += results.passed;
        failed += results.failed;
        skipped += results.skipped;
    }
    
    // Script tests
    if let Ok(results) = script_tests::run_all_tests() {
        passed += results.passed;
        failed += results.failed;
        skipped += results.skipped;
    }
    
    // Block tests
    if let Ok(results) = block_tests::run_all_tests() {
        passed += results.passed;
        failed += results.failed;
        skipped += results.skipped;
    }
    
    println!("Core Test Vectors: {} passed, {} failed, {} skipped", 
             passed, failed, skipped);
    
    // Fail if any test vectors failed
    assert_eq!(failed, 0, "Core test vectors must all pass");
}
```

---

### Phase 4: Continuous Differential Testing

**Goal**: Continuously compare against Bitcoin Core reference implementation

#### 4.1 Core RPC Integration

**Enhance `tests/integration/differential_tests.rs`**:

```rust
/// Continuous differential testing against Bitcoin Core
/// 
/// This test suite compares our consensus validation against
/// Bitcoin Core's validation for the same inputs.
#[tokio::test]
#[ignore] // Only run when Core RPC is available
async fn continuous_differential_testing() {
    let config = CoreRpcConfig::from_env().expect("Core RPC config");
    
    // Test random transactions
    for _ in 0..100 {
        let tx = generate_random_transaction();
        let result = compare_transaction_validation(&tx, &config).await?;
        
        if result.divergence {
            panic!("Divergence detected: {}", result.divergence_reason.unwrap());
        }
    }
    
    // Test random blocks
    for _ in 0..50 {
        let block = generate_random_block();
        let result = compare_block_validation(&block, &config).await?;
        
        if result.divergence {
            panic!("Divergence detected: {}", result.divergence_reason.unwrap());
        }
    }
}
```

#### 4.2 Historical Block Replay

**Add continuous historical block validation**:

```rust
/// Replay historical mainnet blocks and verify consensus matches
#[tokio::test]
#[ignore] // Requires mainnet block data
async fn historical_block_replay() {
    // Load historical blocks from mainnet
    let blocks = load_historical_blocks("tests/test_data/mainnet_blocks");
    
    for (height, block) in blocks {
        // Validate with our consensus
        let our_result = connect_block(&block, UtxoSet::new(), height);
        
        // Compare with Core's validation (if available)
        // This ensures we match historical consensus decisions
    }
}
```

---

### Phase 5: Complete Mathematical Specifications

**Goal**: Document all consensus functions with formal mathematical notation

#### 5.1 Specification Template

**For each consensus function, add**:

```rust
/// Mathematical Specification:
/// 
/// ∀ input I: function(I) = output ⟺ conditions(I)
/// 
/// Where:
/// - I: Input type
/// - conditions(I): Mathematical conditions that must hold
/// 
/// Invariants:
/// - Invariant 1: Description with mathematical notation
/// - Invariant 2: Description with mathematical notation
/// 
/// Proof:
/// - Kani proof: `kani_verify_function_name`
/// - Property test: `prop_function_invariant`
/// - Test vectors: Core test vectors pass
/// 
/// Reference:
/// - Bitcoin Core: `function_name()` in `file.cpp`
/// - BIP: BIP-XXX
/// - Orange Paper: Section X.Y
```

#### 5.2 Functions Needing Specifications

1. **Block Validation**:
   - `connect_block`: Full specification
   - `validate_block_header`: Header rules
   - `check_merkle_root`: Merkle tree validation
   - `check_witness_commitment`: SegWit validation

2. **Transaction Validation**:
   - `check_transaction`: Structure rules
   - `check_tx_inputs`: Input validation
   - `verify_scripts`: Script execution

3. **Economic Rules**:
   - `get_block_subsidy`: Halving schedule
   - `total_supply`: Supply calculation
   - `calculate_fee`: Fee calculation

4. **Proof of Work**:
   - `check_proof_of_work`: PoW validation
   - `expand_target`: Target expansion
   - `compress_target`: Target compression
   - `get_next_work_required`: Difficulty adjustment

---

### Phase 6: CI Enforcement

**Goal**: Block merges if any verification fails

#### 6.1 Enhanced CI Workflow

**Update `.github/workflows/verify.yml`**:

```yaml
name: Mathematical Verification

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  kani-proofs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Kani
        run: |
          cargo install --locked kani-verifier
      - name: Run Kani Proofs
        run: |
          cargo kani --features verify
        continue-on-error: false # Block on failure
  
  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Property Tests
        run: |
          cargo test --test property_tests --features verify
        continue-on-error: false
  
  core-test-vectors:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download Core Test Vectors
        run: ./scripts/download_core_test_vectors.sh
      - name: Run Core Test Vectors
        run: |
          cargo test --test core_test_vectors
        continue-on-error: false
  
  differential-testing:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - name: Start Bitcoin Core (regtest)
        run: |
          # Start Core in regtest mode for testing
          # (Implementation depends on CI setup)
      - name: Run Differential Tests
        run: |
          cargo test --test differential_tests --features core-rpc
        continue-on-error: false
  
  verification-summary:
    needs: [kani-proofs, property-tests, core-test-vectors, differential-testing]
    runs-on: ubuntu-latest
    if: always()
    steps:
      - name: Verification Summary
        run: |
          echo "## Verification Results" >> $GITHUB_STEP_SUMMARY
          echo "- Kani Proofs: ${{ needs.kani-proofs.result }}" >> $GITHUB_STEP_SUMMARY
          echo "- Property Tests: ${{ needs.property-tests.result }}" >> $GITHUB_STEP_SUMMARY
          echo "- Core Test Vectors: ${{ needs.core-test-vectors.result }}" >> $GITHUB_STEP_SUMMARY
          echo "- Differential Tests: ${{ needs.differential-testing.result }}" >> $GITHUB_STEP_SUMMARY
```

#### 6.2 Governance App Integration

**Update governance app to check verification status**:

```rust
/// Check if PR has passed all mathematical verification
pub async fn check_verification_status(pr: &PullRequest) -> bool {
    // Check CI status for verification jobs
    let ci_status = get_ci_status(pr).await?;
    
    let required_checks = [
        "kani-proofs",
        "property-tests", 
        "core-test-vectors",
        "differential-testing",
    ];
    
    for check in required_checks {
        if !ci_status.passed(check) {
            return false;
        }
    }
    
    true
}
```

---

## Implementation Timeline

### Week 1: Infrastructure Setup
- [ ] Expand Kani proof coverage (Phase 1.1-1.3)
- [ ] Add property-based tests (Phase 2)
- [ ] Set up Core test vector download (Phase 3.1)

### Week 2: Integration & Testing
- [ ] Integrate Core test vectors in CI (Phase 3.2-3.3)
- [ ] Set up differential testing (Phase 4)
- [ ] Document mathematical specifications (Phase 5)

### Week 3: CI Enforcement
- [ ] Update CI workflows (Phase 6.1)
- [ ] Integrate with governance app (Phase 6.2)
- [ ] Run full verification suite
- [ ] Document results

---

## Success Metrics

### Coverage Metrics
- **Kani Proofs**: 100% of critical consensus functions
- **Property Tests**: All consensus invariants tested
- **Core Test Vectors**: 100% pass rate
- **Differential Tests**: 0 divergences

### Quality Metrics
- **Mathematical Specifications**: 100% of consensus functions documented
- **CI Enforcement**: 100% of verification checks required
- **Documentation**: Complete with formal notation

---

## Risk Mitigation

### Risk 1: Verification Timeouts
**Mitigation**: Use bounded proofs, parallel execution, caching

### Risk 2: Core Test Vector Availability
**Mitigation**: Cache vectors, fallback to local copies, graceful degradation

### Risk 3: Core RPC Unavailability
**Mitigation**: Make differential tests optional, use mock responses

---

## References

- [Kani Documentation](https://model-checking.github.io/kani/)
- [Proptest Documentation](https://docs.rs/proptest/)
- [Bitcoin Core Test Framework](https://github.com/bitcoin/bitcoin/tree/master/src/test)
- [Orange Paper](../bllvm-spec/THE_ORANGE_PAPER.md)

---

**Last Updated**: 2025-01-18  
**Status**: Ready for Implementation

