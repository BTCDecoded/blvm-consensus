# Formal Verification Documentation

## Overview

The `consensus-proof` crate implements formal verification for Bitcoin consensus rules using a multi-layered approach that combines mathematical specifications, symbolic verification, and property-based testing. This creates a mathematically locked consensus layer that is auditable and tamper-evident.

## Verification Stack

Our verification approach follows the principle: **"Rust + Tests + Math Specs = Source of Truth"**

### Layer 1: Empirical Testing (Required, Must Pass)
- **Unit tests**: Comprehensive test coverage for all consensus functions
- **Property-based tests**: Randomized testing with `proptest` to discover edge cases
- **Integration tests**: Cross-system validation between consensus components

### Layer 2: Symbolic Verification (Required, Must Pass)
- **Kani model checking**: Bounded symbolic verification with mathematical invariants
- **Mathematical specifications**: Formal documentation of consensus rules
- **State space exploration**: Verification of all possible execution paths

### Layer 3: CI Enforcement (Required, Blocks Merge)
- **Automated verification**: All tests and proofs must pass before merge
- **OpenTimestamps audit logging**: Immutable proof of verification artifacts
- **No human override**: Technical correctness is non-negotiable

## Mathematical Specifications

### Chain Selection (`src/reorganization.rs`)

**Mathematical Specification:**
```
∀ chains C₁, C₂: work(C₁) > work(C₂) ⇒ select(C₁)
```

**Invariants:**
- Selected chain has maximum cumulative work
- Work calculation is deterministic
- Empty chains are rejected
- Chain work is always non-negative

**Verified Functions:**
- `should_reorganize`: Proves longest chain selection
- `calculate_chain_work`: Verifies cumulative work calculation
- `expand_target`: Handles difficulty target edge cases

### Block Subsidy (`src/economic.rs`)

**Mathematical Specification:**
```
∀ h ∈ ℕ: subsidy(h) = 50 * 10^8 * 2^(-⌊h/210000⌋) if ⌊h/210000⌋ < 64 else 0
```

**Invariants:**
- Subsidy halves every 210,000 blocks
- After 64 halvings, subsidy becomes 0
- Subsidy is always non-negative
- Total supply approaches 21M BTC asymptotically

**Verified Functions:**
- `get_block_subsidy`: Verifies halving schedule
- `total_supply`: Proves monotonic increase
- `validate_supply_limit`: Ensures supply cap compliance

### Proof of Work (`src/pow.rs`)

**Mathematical Specification:**
```
∀ header H: CheckProofOfWork(H) = SHA256(SHA256(H)) < ExpandTarget(H.bits)
```

**Invariants:**
- Hash must be less than target for valid proof of work
- Target expansion handles edge cases correctly
- Difficulty adjustment respects bounds [0.25, 4.0]
- Work calculation is deterministic

**Verified Functions:**
- `check_proof_of_work`: Verifies hash < target
- `expand_target`: Handles compact target representation
- `get_next_work_required`: Respects difficulty bounds

### Transaction Validation (`src/transaction.rs`)

**Mathematical Specification:**
```
∀ tx ∈ 𝒯𝒳: CheckTransaction(tx) = valid ⟺ 
  (|tx.inputs| > 0 ∧ |tx.outputs| > 0 ∧ 
   ∀o ∈ tx.outputs: 0 ≤ o.value ≤ M_max ∧
   |tx.inputs| ≤ M_max_inputs ∧ |tx.outputs| ≤ M_max_outputs ∧
   |tx| ≤ M_max_tx_size)
```

**Invariants:**
- Valid transactions have non-empty inputs and outputs
- Output values are bounded [0, MAX_MONEY]
- Input/output counts respect limits
- Transaction size respects limits
- Coinbase transactions have special validation rules

**Verified Functions:**
- `check_transaction`: Validates structure rules
- `check_tx_inputs`: Handles coinbase correctly
- `is_coinbase`: Correctly identifies coinbase transactions

### Block Connection (`src/block.rs`)

**Mathematical Specification:**
```
∀ block B, UTXO set US, height h: ConnectBlock(B, US, h) = (valid, US') ⟺
  (ValidateHeader(B.header) ∧ 
   ∀ tx ∈ B.transactions: CheckTransaction(tx) ∧ CheckTxInputs(tx, US, h) ∧
   VerifyScripts(tx, US) ∧
   CoinbaseOutput ≤ TotalFees + GetBlockSubsidy(h) ∧
   US' = ApplyTransactions(B.transactions, US))
```

**Invariants:**
- Valid blocks have valid headers and transactions
- UTXO set consistency is preserved
- Coinbase output respects economic rules
- Transaction application is atomic

**Verified Functions:**
- `connect_block`: Validates complete block
- `apply_transaction`: Preserves UTXO consistency
- `calculate_tx_id`: Deterministic transaction identification

## Verification Tools

### Kani Model Checker

**Purpose**: Symbolic verification with bounded model checking
**Usage**: `cargo kani --features verify`
**Coverage**: All `#[kani::proof]` functions
**Bounded**: Uses `#[kani::unwind(N)]` for tractability

**Example:**
```rust
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn kani_verify_function() {
    let input: InputType = kani::any();
    kani::assume(input.is_valid());
    
    let result = function_under_test(input);
    
    assert!(result.property_holds());
}
```

### Proptest Property Testing

**Purpose**: Randomized testing to discover edge cases
**Usage**: `cargo test` (runs automatically)
**Coverage**: All `proptest!` macros
**Strategy**: Generates random inputs within specified bounds

**Example:**
```rust
proptest! {
    #[test]
    fn prop_function_invariant(input in strategy) {
        let result = function_under_test(input);
        prop_assert!(result.property_holds());
    }
}
```

### OpenTimestamps Audit Trail

**Purpose**: Immutable proof of verification artifacts
**Usage**: Automatic in CI via `ots stamp`
**Coverage**: All verification artifacts
**Verification**: `ots verify proof-artifacts.tar.gz.ots`

## CI Integration

### Verification Workflow

The `.github/workflows/verify.yml` workflow enforces verification:

1. **Unit & Property Tests** (required)
   - `cargo test --all-features`
   - Must pass for CI to succeed

2. **Kani Model Checking** (required)
   - `cargo kani --features verify`
   - Verifies all `#[kani::proof]` functions
   - Must pass for CI to succeed

3. **OpenTimestamps Audit** (non-blocking)
   - Collect verification artifacts
   - Timestamp proof bundle with `ots stamp`
   - Upload artifacts for transparency

### Local Development

**Run all tests:**
```bash
cargo test --all-features
```

**Run Kani proofs:**
```bash
cargo kani --features verify
```

**Run specific verification:**
```bash
cargo test --test property_tests
cargo kani --features verify --harness kani_verify_function
```

## Adding New Verification

### Step 1: Add Mathematical Specification

Document the mathematical invariant as a comment:

```rust
/// Mathematical Specification:
/// ∀ input I: property(I) ⟺ invariant_holds(I)
/// 
/// Invariants:
/// - Invariant 1: Description
/// - Invariant 2: Description
```

### Step 2: Add Kani Proof

Create a bounded proof with assumptions:

```rust
#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    #[kani::proof]
    #[kani::unwind(10)]
    fn kani_verify_function_name() {
        let input: InputType = kani::any();
        
        // Bound for tractability
        kani::assume(input.len() <= 10);
        
        let result = function_under_test(input);
        
        // Assert invariants
        assert!(result.property_holds());
    }
}
```

### Step 3: Add Property Tests

Create randomized tests with proptest:

```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_function_invariant(
            input in strategy
        ) {
            let result = function_under_test(input);
            prop_assert!(result.property_holds());
        }
    }
}
```

### Step 4: Update CI

The CI workflow automatically picks up new verification. No changes needed.

## Governance Integration

### Multi-Level Enforcement

The verification system integrates with BTCDecoded governance at multiple levels:

1. **Technical Enforcement (CI/CD)**: Automated verification blocks merge
2. **Governance Policy (Layer System)**: Layer 2 requires 6-of-7 signatures
3. **Governance App Enforcement**: Validates CI passed before signatures
4. **Cross-Layer Separation**: Prevents governance capture of consensus

### Ostrom Principles Compliance

- **Monitoring (#4)**: Automated CI verification enforces rules
- **Proportional Equivalence (#2)**: Constitutional changes require maximum consensus
- **Graduated Sanctions (#5)**: PRs without verification cannot progress
- **Clearly Defined Boundaries (#1)**: Separation preserved via module boundaries

## Security Considerations

### Attack Vectors Mitigated

1. **Consensus Rule Violations**: Kani proofs prevent invalid consensus
2. **Edge Case Exploits**: Property tests discover boundary conditions
3. **Implementation Bugs**: Mathematical specs document correct behavior
4. **Governance Capture**: Multi-level enforcement prevents override

### Defense Mechanisms

1. **Mathematical Proofs**: Kani verifies invariants symbolically
2. **Randomized Testing**: Proptest discovers unexpected behavior
3. **Audit Trail**: OpenTimestamps provides immutable proof
4. **CI Enforcement**: No human override of verification results

## Performance Considerations

### Verification Bounds

- **Kani proofs**: Bounded with `#[kani::unwind(N)]` for tractability
- **Property tests**: Limited input ranges to prevent timeouts
- **CI timeouts**: 5-minute limit per verification step

### Optimization Strategies

1. **Bounded verification**: Focus on critical paths
2. **Parallel execution**: CI runs tests in parallel
3. **Incremental verification**: Only verify changed functions
4. **Caching**: CI caches dependencies and artifacts

## Future Enhancements

### Planned Improvements

1. **Expanded Coverage**: Add verification to all consensus functions
2. **Cross-Layer Verification**: Verify protocol-engine and reference-node
3. **Performance Optimization**: Reduce verification time
4. **Documentation**: Add more mathematical specifications

### Optional Integrations

1. **Coq Integration**: Via `coq-of-rust` (non-blocking)
2. **Additional Tools**: ESBMC, CBMC for redundancy
3. **Formal Specs**: TLA+ or Alloy specifications
4. **Interactive Proofs**: Manual verification assistance

## Troubleshooting

### Common Issues

**Kani proof fails:**
- Check bounds with `#[kani::unwind(N)]`
- Add assumptions with `kani::assume()`
- Verify function is deterministic

**Property test fails:**
- Check input strategy bounds
- Add filtering with `prop_filter!`
- Verify property is actually true

**CI verification fails:**
- Check all tests pass locally
- Verify Kani proofs complete
- Check OpenTimestamps installation

### Debug Commands

```bash
# Debug Kani proof
cargo kani --features verify --harness kani_verify_function --debug

# Debug property test
RUST_LOG=debug cargo test prop_function_invariant

# Check verification artifacts
ots verify proof-artifacts.tar.gz.ots
```

## References

- [Kani Documentation](https://model-checking.github.io/kani/)
- [Proptest Documentation](https://docs.rs/proptest/)
- [OpenTimestamps Protocol](https://opentimestamps.org/)
- [BTCDecoded Governance](../governance/GOVERNANCE.md)
- [Orange Paper](../the-orange-paper/THE_ORANGE_PAPER.md)









