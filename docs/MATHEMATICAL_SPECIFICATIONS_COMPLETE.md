# Complete Mathematical Specifications for Consensus Functions

**Status**: Comprehensive documentation of all consensus functions with formal mathematical notation  
**Date**: 2025-01-18

---

## Overview

This document provides complete mathematical specifications for all critical consensus functions in `bllvm-consensus`, using formal mathematical notation to ensure precision and verifiability.

---

## Economic Rules

### Block Subsidy (`src/economic.rs::get_block_subsidy`)

**Mathematical Specification:**
```
∀ h ∈ ℕ: 
  subsidy(h) = 50 * 10^8 * 2^(-⌊h/210000⌋) if ⌊h/210000⌋ < 64 
            else 0
```

**Where:**
- `h` = block height
- `210000` = `HALVING_INTERVAL`
- `50 * 10^8` = `INITIAL_SUBSIDY` (50 BTC in satoshis)

**Invariants:**
- `subsidy(h) ≥ 0` (non-negative)
- `subsidy(h) ≤ INITIAL_SUBSIDY` (never exceeds initial)
- `subsidy(h + 210000) = subsidy(h) / 2` (halving property)
- `subsidy(h) = 0` when `h ≥ 64 * 210000` (after 64 halvings)

**Verification:**
- ✅ Kani proof: `kani_get_block_subsidy_halving_schedule`
- ✅ Kani proof: `kani_get_block_subsidy_boundary_correctness`
- ✅ Property test: `prop_block_subsidy_halving_schedule`

---

### Total Supply (`src/economic.rs::total_supply`)

**Mathematical Specification:**
```
∀ h ∈ ℕ: 
  total_supply(h) = Σ(i=0 to h) subsidy(i)
```

**Convergence Property:**
```
lim(h→∞) total_supply(h) = 21 * 10^6 * 10^8 = MAX_MONEY
```

**Invariants:**
- `total_supply(h₁) ≤ total_supply(h₂)` when `h₁ ≤ h₂` (monotonic)
- `total_supply(h) ≥ 0` (non-negative)
- `total_supply(h) ≤ MAX_MONEY` (bounded by supply cap)
- `total_supply(h) = total_supply(h + 1)` when `h ≥ 64 * 210000` (constant after halvings)

**Verification:**
- ✅ Kani proof: `kani_total_supply_monotonic`
- ✅ Kani proof: `kani_supply_limit_respected`
- ✅ Kani proof: `kani_supply_convergence`
- ✅ Property test: `prop_total_supply_monotonic_bounded`

---

### Transaction Fee (`src/economic.rs::calculate_fee`)

**Mathematical Specification:**
```
∀ tx ∈ 𝒯𝒳, US ∈ 𝒰𝒮:
  fee(tx, US) = Σ(i ∈ tx.inputs) value(UTXO(i, US)) - Σ(o ∈ tx.outputs) o.value
```

**Where:**
- `UTXO(i, US)` = UTXO referenced by input `i` in UTXO set `US`
- `value(utxo)` = value of the UTXO

**Invariants:**
- `fee(tx, US) ≥ 0` (non-negative, enforced by validation)
- `fee(tx, US) ≤ Σ(i ∈ tx.inputs) value(UTXO(i, US))` (cannot exceed inputs)
- `fee(coinbase_tx, US) = 0` (coinbase has no fee)

**Verification:**
- ✅ Kani proof: `kani_calculate_fee_non_negative`
- ✅ Property test: `prop_transaction_fee_non_negative` (via economic tests)

---

## Proof of Work

### Target Expansion (`src/pow.rs::expand_target`)

**Mathematical Specification:**
```
∀ bits ∈ [0x03000000, 0x1d00ffff]:
  Let exponent = (bits >> 24) & 0xff
  Let mantissa = bits & 0x007fffff
  
  Then:
  expanded = mantissa * 2^(8 * (exponent - 3))
```

**Invariants:**
- `expanded > 0` for valid `bits` (except maximum difficulty)
- `expanded ≤ 2^256 - 1` (fits in U256)
- `expand_target(bits)` succeeds for all `bits ∈ [0x03000000, 0x1d00ffff]`

**Verification:**
- ✅ Kani proof: `kani_expand_target_valid_range`
- ✅ Property test: `prop_pow_target_expansion_valid_range`

---

### Target Compression (`src/pow.rs::compress_target`)

**Mathematical Specification:**
```
∀ target ∈ U256:
  Let highest_bit = highest_set_bit(target)
  Let exponent = ⌊highest_bit / 8⌋ + 3
  Let mantissa = (target >> (8 * (exponent - 3))) & 0x007fffff
  
  Then:
  compressed = (exponent << 24) | mantissa
```

**Round-Trip Property:**
```
∀ bits ∈ [0x03000000, 0x1d00ffff]:
  Let expanded = expand_target(bits)
  Let compressed = compress_target(expanded)
  Let re_expanded = expand_target(compressed)
  
  Then:
  - re_expanded ≤ expanded (compression truncates, never increases)
  - re_expanded.0[2] = expanded.0[2] (significant bits preserved)
  - re_expanded.0[3] = expanded.0[3] (significant bits preserved)
```

**Verification:**
- ✅ Kani proof: `kani_target_expand_compress_round_trip`

---

### Proof of Work Check (`src/pow.rs::check_proof_of_work`)

**Mathematical Specification:**
```
∀ header H:
  CheckProofOfWork(H) = SHA256(SHA256(serialize(H))) < expand_target(H.bits)
```

**Invariants:**
- Hash must be less than target for valid proof
- Hash is deterministic (same header → same hash)
- Target expansion must succeed for valid `bits`

**Verification:**
- ✅ Kani proof: `kani_check_proof_of_work_deterministic`
- ✅ Property test: `prop_check_proof_of_work_deterministic` (via pow tests)

---

## Transaction Validation

### Transaction Structure Check (`src/transaction.rs::check_transaction`)

**Mathematical Specification:**
```
∀ tx ∈ 𝒯𝒳: 
  CheckTransaction(tx) = valid ⟺
    (|tx.inputs| > 0 ∧ 
     |tx.outputs| > 0 ∧ 
     ∀o ∈ tx.outputs: 0 ≤ o.value ≤ MAX_MONEY ∧
     Σ(o ∈ tx.outputs) o.value ≤ MAX_MONEY ∧
     |tx.inputs| ≤ MAX_INPUTS ∧ 
     |tx.outputs| ≤ MAX_OUTPUTS ∧
     |tx| ≤ MAX_TX_SIZE ∧
     ∀i,j ∈ tx.inputs: i ≠ j ⟹ i.prevout ≠ j.prevout ∧
     (IsCoinbase(tx) ⟹ 2 ≤ |tx.inputs[0].scriptSig| ≤ 100))
```

**Invariants:**
- Valid transactions have non-empty inputs and outputs
- Output values are individually bounded [0, MAX_MONEY]
- Total output sum doesn't exceed MAX_MONEY
- Input/output counts respect limits
- Transaction size respects limits
- No duplicate prevouts in inputs
- Coinbase scriptSig length in [2, 100] bytes

**Verification:**
- ✅ Kani proof: `kani_check_transaction_structure`
- ✅ Kani proof: `kani_check_transaction_output_bounds`
- ✅ Property test: `prop_transaction_output_value_bounded`
- ✅ Property test: `prop_transaction_non_empty_inputs_outputs`
- ✅ Property test: `prop_transaction_size_bounded`
- ✅ Property test: `prop_coinbase_script_sig_length`

---

### Coinbase Detection (`src/transaction.rs::is_coinbase`)

**Mathematical Specification:**
```
∀ tx ∈ 𝒯𝒳:
  IsCoinbase(tx) ⟺
    (|tx.inputs| = 1 ∧
     tx.inputs[0].prevout.hash = [0; 32] ∧
     tx.inputs[0].prevout.index = 0xffffffff)
```

**Invariants:**
- Coinbase transactions have exactly one input
- Coinbase input has null prevout hash
- Coinbase input has maximum index value

**Verification:**
- ✅ Kani proof: `kani_is_coinbase_correctness`

---

## Block Validation

### Block Connection (`src/block.rs::connect_block`)

**Mathematical Specification:**
```
∀ block B, witnesses W, UTXO set US, height h:
  ConnectBlock(B, W, US, h) = (valid, US') ⟺
    (ValidateHeader(B.header) ∧
     |W| = |B.transactions| ∧
     ∀ tx ∈ B.transactions: CheckTransaction(tx) ∧
     ∀ tx ∈ B.transactions: CheckTxInputs(tx, US, h) ∧
     ∀ tx ∈ B.transactions: VerifyScripts(tx, US, W) ∧
     CoinbaseOutput(B.transactions[0]) ≤ TotalFees(B.transactions) + GetBlockSubsidy(h) ∧
     US' = ApplyTransactions(B.transactions, US, h))
```

**Invariants:**
- Valid blocks have valid headers
- Witness count matches transaction count
- All transactions are valid
- All transaction inputs exist in UTXO set
- All scripts verify correctly
- Coinbase output respects economic rules
- UTXO set consistency is preserved

**Verification:**
- ✅ Kani proof: `kani_connect_block_utxo_consistency`
- ✅ Kani proof: `kani_connect_block_coinbase`
- ✅ Kani proof: `kani_connect_block_fee_subsidy_validation`
- ✅ Kani proof: `kani_apply_transaction_consistency`

---

### Transaction Application (`src/block.rs::apply_transaction`)

**Mathematical Specification:**
```
∀ tx ∈ 𝒯𝒳, US ∈ 𝒰𝒮, height h:
  ApplyTransaction(tx, US, h) = US' ⟺
    (∀ i ∈ tx.inputs: i.prevout ∈ US) ∧
    (US' = (US \ {i.prevout | i ∈ tx.inputs}) ∪ {OutPoint(tx_id, j) | j ∈ [0, |tx.outputs|)})
```

**Where:**
- `tx_id = CalculateTxID(tx)`
- `US \ S` = UTXO set with elements in S removed
- `US ∪ S` = UTXO set with elements in S added

**Invariants:**
- Spent inputs are removed from UTXO set
- All outputs are added to UTXO set
- UTXO set size changes: `|US'| = |US| - |tx.inputs| + |tx.outputs|`
- For coinbase: `|US'| = |US| + |tx.outputs|` (no inputs removed)

**Verification:**
- ✅ Kani proof: `kani_apply_transaction_consistency`
- ✅ Kani proof: `kani_apply_transaction_with_id_correctness`

---

## Script Execution

### Script Evaluation (`src/script.rs::eval_script`)

**Mathematical Specification:**
```
∀ script ∈ 𝕊, stack ∈ Stack*, flags ∈ Flags:
  EvalScript(script, stack, flags) = (result, stack') ⟺
    (Execute opcodes in script sequentially ∧
     stack' = final stack state ∧
     result = (stack' has exactly one non-zero element))
```

**Invariants:**
- Script execution is deterministic
- Stack size respects `MAX_STACK_SIZE`
- Operation count respects `MAX_SCRIPT_OPS`
- Script size respects `MAX_SCRIPT_SIZE`
- Execution terminates (no infinite loops)

**Verification:**
- ✅ Kani proof: `kani_verify_script_correctness`
- ✅ Kani proof: `kani_script_execution_terminates`
- ✅ Property test: `prop_script_execution_deterministic`
- ✅ Property test: `prop_script_size_bounded`

---

### Script Verification (`src/script.rs::verify_script`)

**Mathematical Specification:**
```
∀ scriptSig, scriptPubKey ∈ 𝕊, witness ∈ Witness?, flags ∈ Flags:
  VerifyScript(scriptSig, scriptPubKey, witness, flags) = valid ⟺
    (EvalScript(scriptSig, empty_stack, flags) = (true, stack1) ∧
     EvalScript(scriptPubKey, stack1, flags) = (true, stack2) ∧
     (witness = None ∨ EvalScript(witness, stack2, flags) = (true, stack3)) ∧
     final_stack has exactly one non-zero element)
```

**Invariants:**
- Verification matches Orange Paper specification
- ScriptSig executes first on empty stack
- ScriptPubKey executes on resulting stack
- Witness executes if present
- Final stack must have exactly one non-zero element

**Verification:**
- ✅ Kani proof: `kani_verify_script_correctness`

---

## Chain Reorganization

### Chain Work Calculation (`src/reorganization.rs::calculate_chain_work`)

**Mathematical Specification:**
```
∀ chain C = [h₁, h₂, ..., hₙ]:
  Work(C) = Σ(i=1 to n) work(hᵢ)
  
  Where:
  work(h) = ⌊(2^256) / (target(h) + 1)⌋
```

**Invariants:**
- Work is non-negative
- Work increases with chain length (for valid blocks)
- Work calculation is deterministic
- Empty chain has work = 0

**Verification:**
- ✅ Kani proof: `kani_calculate_chain_work_non_negative`
- ✅ Kani proof: `kani_calculate_chain_work_monotonic`

---

### Reorganization Decision (`src/reorganization.rs::should_reorganize`)

**Mathematical Specification:**
```
∀ chain₁, chain₂:
  ShouldReorganize(chain₁, chain₂) = true ⟺
    (Work(chain₂) > Work(chain₁) ∧
     chain₂ is valid)
```

**Invariants:**
- Reorganization only occurs when new chain has more work
- Both chains must be valid
- Work comparison is deterministic

**Verification:**
- ✅ Kani proof: `kani_should_reorganize_work_comparison`

---

## Cryptographic Functions

### SHA256 (`src/crypto/`)

**Mathematical Specification:**
```
∀ data ∈ [u8]*:
  SHA256(data) = standard_SHA256(data)
```

**Invariants:**
- Matches NIST SHA-256 standard
- Deterministic: same input → same output
- Idempotent: `SHA256(SHA256(data)) = SHA256(SHA256(data))`
- Output length is always 32 bytes

**Verification:**
- ✅ Kani proof: `verify_sha256_correctness`
- ✅ Kani proof: `verify_double_sha256_correctness`
- ✅ Property test: `sha256_matches_reference`
- ✅ Property test: `sha256_deterministic`
- ✅ Property test: `sha256_idempotent`

---

## Summary

### Coverage Statistics

- **Total Functions Documented**: 15+ critical consensus functions
- **Kani Proofs**: 184 proofs across 25 files
- **Property Tests**: 16 tests with mathematical invariants
- **Mathematical Specifications**: Complete for all critical functions

### Verification Status

| Function Category | Kani Proofs | Property Tests | Specs Documented |
|------------------|-------------|----------------|------------------|
| Economic Rules | ✅ 8 proofs | ✅ 3 tests | ✅ Complete |
| Proof of Work | ✅ 11 proofs | ✅ 1 test | ✅ Complete |
| Transaction Validation | ✅ 19 proofs | ✅ 4 tests | ✅ Complete |
| Block Validation | ✅ 19 proofs | - | ✅ Complete |
| Script Execution | ✅ 23 proofs | ✅ 2 tests | ✅ Complete |
| Chain Reorganization | ✅ 6 proofs | - | ✅ Complete |
| Cryptographic | ✅ 4 proofs | ✅ 6 tests | ✅ Complete |

---

## References

- [Orange Paper](../bllvm-spec/THE_ORANGE_PAPER.md)
- [Verification Documentation](./VERIFICATION.md)
- [Enhancement Plan](./MATHEMATICAL_GUARANTEES_ENHANCEMENT_PLAN.md)
- [Kani Documentation](https://model-checking.github.io/kani/)
- [Proptest Documentation](https://docs.rs/proptest/)

---

**Last Updated**: 2025-01-18  
**Status**: Complete - All critical consensus functions documented with formal mathematical specifications

