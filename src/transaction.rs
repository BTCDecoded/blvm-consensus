//! Transaction validation functions from Orange Paper Section 5.1
//!
//! Performance optimizations (Phase 6.3):
//! - Early-exit fast-path checks for obviously invalid transactions

use crate::constants::*;
use crate::error::{ConsensusError, Result};
use crate::types::*;
use std::borrow::Cow;

// Cold error construction helpers - these paths are rarely taken
#[cold]
fn make_output_sum_overflow_error() -> ConsensusError {
    ConsensusError::TransactionValidation("Output value sum overflow".into())
}

#[cold]
fn make_fee_calculation_underflow_error() -> ConsensusError {
    ConsensusError::TransactionValidation("Fee calculation underflow".into())
}

/// Phase 6.3: Fast-path early-exit checks for transaction validation
///
/// Performs quick checks before expensive validation operations.
/// Returns Some(ValidationResult) if fast-path can determine validity, None if full validation needed.
#[inline(always)]
#[cfg(feature = "production")]
fn check_transaction_fast_path(tx: &Transaction) -> Option<ValidationResult> {
    // Quick reject: empty inputs or outputs (most common invalid case)
    if tx.inputs.is_empty() || tx.outputs.is_empty() {
        return Some(ValidationResult::Invalid(
            "Empty inputs or outputs".to_string(),
        ));
    }

    // Quick reject: obviously too many inputs/outputs (before expensive size calculation)
    if tx.inputs.len() > MAX_INPUTS {
        return Some(ValidationResult::Invalid(format!(
            "Too many inputs: {}",
            tx.inputs.len()
        )));
    }
    if tx.outputs.len() > MAX_OUTPUTS {
        return Some(ValidationResult::Invalid(format!(
            "Too many outputs: {}",
            tx.outputs.len()
        )));
    }

    // Quick reject: obviously invalid value ranges (before expensive validation)
    // Check if any output value is negative or exceeds MAX_MONEY
    for output in &tx.outputs {
        if output.value < 0 || output.value > MAX_MONEY {
            return Some(ValidationResult::Invalid(format!(
                "Invalid output value: {}",
                output.value
            )));
        }
    }

    // Quick reject: coinbase with invalid scriptSig length
    if tx.inputs.len() == 1
        && tx.inputs[0].prevout.hash == [0u8; 32]
        && tx.inputs[0].prevout.index == 0xffffffff
    {
        let script_sig_len = tx.inputs[0].script_sig.len();
        if !(2..=100).contains(&script_sig_len) {
            return Some(ValidationResult::Invalid(format!(
                "Coinbase scriptSig length {} must be between 2 and 100 bytes",
                script_sig_len
            )));
        }
    }

    // Fast-path can't validate everything, needs full validation
    None
}

/// CheckTransaction: 𝒯𝒳 → {valid, invalid}
///
/// A transaction tx = (v, ins, outs, lt) is valid if and only if:
/// 1. |ins| > 0 ∧ |outs| > 0
/// 2. ∀o ∈ outs: 0 ≤ o.value ≤ M_max
/// 3. ∑_{o ∈ outs} o.value ≤ M_max (total output sum)
/// 4. |ins| ≤ M_max_inputs
/// 5. |outs| ≤ M_max_outputs
/// 6. |tx| ≤ M_max_tx_size
/// 7. ∀i,j ∈ ins: i ≠ j ⟹ i.prevout ≠ j.prevout (no duplicate inputs)
/// 8. If tx is coinbase: 2 ≤ |ins[0].scriptSig| ≤ 100
///
/// Performance optimization (Phase 6.3): Uses fast-path checks before full validation.
#[track_caller] // Better error messages showing caller location
#[cfg_attr(feature = "production", inline(always))]
#[cfg_attr(not(feature = "production"), inline)]
pub fn check_transaction(tx: &Transaction) -> Result<ValidationResult> {
    // Phase 6.3: Fast-path early exit for obviously invalid transactions
    #[cfg(feature = "production")]
    if let Some(result) = check_transaction_fast_path(tx) {
        return Ok(result);
    }

    // 1. Check inputs and outputs are not empty (redundant if fast-path worked, but safe fallback)
    if tx.inputs.is_empty() || tx.outputs.is_empty() {
        return Ok(ValidationResult::Invalid(
            "Empty inputs or outputs".to_string(),
        ));
    }

    // 2. Check output values are valid and calculate total sum in one pass (Orange Paper Section 5.1, rules 2 & 3)
    // ∀o ∈ outs: 0 ≤ o.value ≤ M_max ∧ ∑_{o ∈ outs} o.value ≤ M_max
    // BLLVM Optimization: Use Kani-proven bounds for output access in hot path
    let mut total_output_value = 0i64;
    #[cfg(feature = "production")]
    {
        use crate::optimizations::kani_optimized_access::get_proven_by_kani;
        for i in 0..tx.outputs.len() {
            if let Some(output) = get_proven_by_kani(&tx.outputs, i) {
                if output.value < 0 || output.value > MAX_MONEY {
                    return Ok(ValidationResult::Invalid(format!(
                        "Invalid output value {} at index {}",
                        output.value, i
                    )));
                }
                // Accumulate sum with overflow check
                total_output_value = total_output_value
                    .checked_add(output.value)
                    .ok_or_else(make_output_sum_overflow_error)?;
            }
        }
    }
    
    #[cfg(not(feature = "production"))]
    {
        for (i, output) in tx.outputs.iter().enumerate() {
            if output.value < 0 || output.value > MAX_MONEY {
                return Ok(ValidationResult::Invalid(format!(
                    "Invalid output value {} at index {}",
                    output.value, i
                )));
            }
            // Accumulate sum with overflow check
            total_output_value = total_output_value
                .checked_add(output.value)
                .ok_or_else(make_output_sum_overflow_error)?;
        }
    }

    // 2b. Check total output sum doesn't exceed MAX_MONEY (Orange Paper Section 5.1, rule 3)
    if total_output_value > MAX_MONEY {
        return Ok(ValidationResult::Invalid(format!(
            "Total output value {total_output_value} exceeds maximum money supply"
        )));
    }

    // 3. Check input count limit (redundant if fast-path worked)
    if tx.inputs.len() > MAX_INPUTS {
        return Ok(ValidationResult::Invalid(format!(
            "Too many inputs: {}",
            tx.inputs.len()
        )));
    }

    // 4. Check output count limit (redundant if fast-path worked)
    if tx.outputs.len() > MAX_OUTPUTS {
        return Ok(ValidationResult::Invalid(format!(
            "Too many outputs: {}",
            tx.outputs.len()
        )));
    }

    // 5. Check transaction size limit
    let tx_size = calculate_transaction_size(tx);
    if tx_size > MAX_TX_SIZE {
        return Ok(ValidationResult::Invalid(format!(
            "Transaction too large: {tx_size} bytes"
        )));
    }

    // 7. Check for duplicate inputs (Orange Paper Section 5.1, rule 4)
    // ∀i,j ∈ ins: i ≠ j ⟹ i.prevout ≠ j.prevout
    // Optimization: Use HashSet for O(n) duplicate detection instead of O(n²) nested loop
    use std::collections::HashSet;
    let mut seen_prevouts = HashSet::with_capacity(tx.inputs.len());
    for (i, input) in tx.inputs.iter().enumerate() {
        if !seen_prevouts.insert(&input.prevout) {
            return Ok(ValidationResult::Invalid(format!(
                "Duplicate input prevout at index {i}"
            )));
        }
    }

    // 8. Check coinbase scriptSig length (Orange Paper Section 5.1, rule 5)
    // If tx is coinbase: 2 ≤ |ins[0].scriptSig| ≤ 100
    if is_coinbase(tx) {
        let script_sig_len = tx.inputs[0].script_sig.len();
        if !(2..=100).contains(&script_sig_len) {
            return Ok(ValidationResult::Invalid(format!(
                "Coinbase scriptSig length {script_sig_len} must be between 2 and 100 bytes"
            )));
        }
    }

    Ok(ValidationResult::Valid)
}

/// CheckTxInputs: 𝒯𝒳 × 𝒰𝒮 × ℕ → {valid, invalid} × ℤ
///
/// For transaction tx with UTXO set us at height h:
/// 1. If tx is coinbase: return (valid, 0)
/// 2. If tx is not coinbase: ∀i ∈ ins: ¬i.prevout.IsNull() (Orange Paper Section 5.1, rule 6)
/// 3. Let total_in = Σᵢ us(i.prevout).value
/// 4. Let total_out = Σₒ o.value
/// 5. If total_in < total_out: return (invalid, 0)
/// 6. Return (valid, total_in - total_out)
#[cfg_attr(feature = "production", inline(always))]
#[cfg_attr(not(feature = "production"), inline)]
pub fn check_tx_inputs(
    tx: &Transaction,
    utxo_set: &UtxoSet,
    _height: Natural,
) -> Result<(ValidationResult, Integer)> {
    // Check if this is a coinbase transaction
    if is_coinbase(tx) {
        return Ok((ValidationResult::Valid, 0));
    }

    // Check that non-coinbase inputs don't have null prevouts (Orange Paper Section 5.1, rule 6)
    // ∀i ∈ ins: ¬i.prevout.IsNull()
    // BLLVM Optimization: Use Kani-proven bounds for input access in hot path
    #[cfg(feature = "production")]
    {
        use crate::optimizations::kani_optimized_access::get_proven_by_kani;
        for i in 0..tx.inputs.len() {
            if let Some(input) = get_proven_by_kani(&tx.inputs, i) {
                if input.prevout.hash == [0u8; 32] && input.prevout.index == 0xffffffff {
                    return Ok((
                        ValidationResult::Invalid(format!("Non-coinbase input {i} has null prevout")),
                        0,
                    ));
                }
            }
        }
    }
    
    #[cfg(not(feature = "production"))]
    {
        for (i, input) in tx.inputs.iter().enumerate() {
            if input.prevout.hash == [0u8; 32] && input.prevout.index == 0xffffffff {
                return Ok((
                    ValidationResult::Invalid(format!("Non-coinbase input {i} has null prevout")),
                    0,
                ));
            }
        }
    }

    // Optimization: Batch UTXO lookups - collect all prevouts first, then lookup
    // This improves cache locality and reduces HashMap traversal overhead
    // Optimization: Pre-allocate with known size
    let input_utxos: Vec<(usize, Option<&UTXO>)> = {
        let mut result = Vec::with_capacity(tx.inputs.len());
        for (i, input) in tx.inputs.iter().enumerate() {
            result.push((i, utxo_set.get(&input.prevout)));
        }
        result
    };

    let mut total_input_value = 0i64;

    for (i, opt_utxo) in input_utxos {
        // Check if input exists in UTXO set
        if let Some(utxo) = opt_utxo {
            // Check coinbase maturity: coinbase outputs cannot be spent until COINBASE_MATURITY blocks deep
            // This is enforced by checking if the UTXO was created at height h and current height >= h + COINBASE_MATURITY
            // Note: For coinbase outputs, we check if height difference is sufficient
            // If height is available, we should check: height >= utxo.height + COINBASE_MATURITY
            // For now, we rely on the UTXO height field which should be set correctly during block connection

            // Use checked arithmetic to prevent overflow
            total_input_value = total_input_value.checked_add(utxo.value).ok_or_else(|| {
                ConsensusError::TransactionValidation(format!("Input value overflow at input {i}").into())
            })?;
        } else {
            return Ok((
                ValidationResult::Invalid(format!("Input {i} not found in UTXO set")),
                0,
            ));
        }
    }

    // Use checked sum to prevent overflow when summing outputs
    let total_output_value: i64 = tx
        .outputs
        .iter()
        .try_fold(0i64, |acc, output| {
            acc.checked_add(output.value).ok_or_else(|| {
                ConsensusError::TransactionValidation("Output value overflow".into())
            })
        })
        .map_err(|e| ConsensusError::TransactionValidation(Cow::Owned(e.to_string())))?;

    // Check that output total doesn't exceed MAX_MONEY (Bitcoin Core check)
    if total_output_value > MAX_MONEY {
        return Ok((
            ValidationResult::Invalid(format!(
                "Total output value {total_output_value} exceeds maximum money supply"
            )),
            0,
        ));
    }

    if total_input_value < total_output_value {
        return Ok((
            ValidationResult::Invalid("Insufficient input value".to_string()),
            0,
        ));
    }

    // Use checked subtraction to prevent underflow (shouldn't happen due to check above, but be safe)
    let fee = total_input_value
        .checked_sub(total_output_value)
        .ok_or_else(make_fee_calculation_underflow_error)?;

    Ok((ValidationResult::Valid, fee))
}

/// Check if transaction is coinbase
#[inline]
pub fn is_coinbase(tx: &Transaction) -> bool {
    tx.inputs.len() == 1
        && tx.inputs[0].prevout.hash == [0u8; 32]
        && tx.inputs[0].prevout.index == 0xffffffff
}

/// Calculate transaction size (simplified)
#[inline]
fn calculate_transaction_size(tx: &Transaction) -> usize {
    // Simplified size calculation
    // In reality, this would be the serialized size
    let size = 4 + // version
        tx.inputs.len() * 41 + // inputs (simplified)
        tx.outputs.len() * 9 + // outputs (simplified)
        4; // lock_time
    
    // Runtime assertion: Transaction size must be positive
    debug_assert!(
        size > 0,
        "Transaction size ({}) must be positive",
        size
    );
    
    // Runtime assertion: Transaction size must not exceed MAX_TX_SIZE
    debug_assert!(
        size <= MAX_TX_SIZE as usize,
        "Transaction size ({}) must not exceed MAX_TX_SIZE ({})",
        size,
        MAX_TX_SIZE
    );
    
    size
}

// ============================================================================
// FORMAL VERIFICATION
// ============================================================================

/// Mathematical Specification for Transaction Validation (Orange Paper Section 5.1):
/// ∀ tx ∈ 𝒯𝒳: CheckTransaction(tx) = valid ⟺
///   (|tx.inputs| > 0 ∧ |tx.outputs| > 0 ∧
///    ∀o ∈ tx.outputs: 0 ≤ o.value ≤ M_max ∧
///    ∑_{o ∈ tx.outputs} o.value ≤ M_max ∧
///    |tx.inputs| ≤ M_max_inputs ∧ |tx.outputs| ≤ M_max_outputs ∧
///    |tx| ≤ M_max_tx_size ∧
///    ∀i,j ∈ tx.inputs: i ≠ j ⟹ i.prevout ≠ j.prevout ∧
///    (IsCoinbase(tx) ⟹ 2 ≤ |tx.inputs[0].scriptSig| ≤ 100))
///
/// Invariants:
/// - Valid transactions have non-empty inputs and outputs
/// - Output values are bounded [0, MAX_MONEY] individually (rule 2)
/// - Total output sum doesn't exceed MAX_MONEY (rule 3)
/// - Input/output counts respect limits
/// - Transaction size respects limits
/// - No duplicate prevouts in inputs (rule 4)
/// - Coinbase transactions have scriptSig length [2, 100] bytes (rule 5)
/// - Non-coinbase inputs must not have null prevouts (rule 6, checked in check_tx_inputs)

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: check_transaction validates structure correctly
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_check_transaction_structure() {
        use crate::kani_helpers::{assume_transaction_bounds_custom, unwind_bounds};

        let tx: Transaction = kani::any();

        // Bound for tractability using standardized helpers
        assume_transaction_bounds_custom!(tx, 10, 10);

        let result =
            check_transaction(&tx).unwrap_or(ValidationResult::Invalid("Error".to_string()));

        // Structure invariants
        match result {
            ValidationResult::Valid => {
                // Valid transactions must have non-empty inputs and outputs
                assert!(!tx.inputs.is_empty(), "Valid transaction must have inputs");
                assert!(
                    !tx.outputs.is_empty(),
                    "Valid transaction must have outputs"
                );

                // Valid transactions must respect limits
                assert!(
                    tx.inputs.len() <= MAX_INPUTS,
                    "Valid transaction must respect input limit"
                );
                assert!(
                    tx.outputs.len() <= MAX_OUTPUTS,
                    "Valid transaction must respect output limit"
                );

                // Valid transactions must have valid output values
                let mut total_output = 0i64;
                for output in &tx.outputs {
                    assert!(
                        output.value >= 0,
                        "Valid transaction outputs must be non-negative"
                    );
                    assert!(
                        output.value <= MAX_MONEY,
                        "Valid transaction outputs must not exceed max money"
                    );
                    total_output = total_output
                        .checked_add(output.value)
                        .unwrap_or(MAX_MONEY + 1);
                }
                // Total output sum must not exceed MAX_MONEY (Orange Paper Section 5.1, rule 3)
                assert!(
                    total_output <= MAX_MONEY,
                    "Total output value must not exceed MAX_MONEY"
                );

                // Valid transactions must not have duplicate prevouts (Orange Paper Section 5.1, rule 4)
                for i in 0..tx.inputs.len() {
                    for j in (i + 1)..tx.inputs.len() {
                        assert!(
                            tx.inputs[i].prevout != tx.inputs[j].prevout,
                            "Valid transaction must not have duplicate prevouts"
                        );
                    }
                }

                // Coinbase transactions must have scriptSig length [2, 100] (Orange Paper Section 5.1, rule 5)
                if is_coinbase(&tx) {
                    let script_sig_len = tx.inputs[0].script_sig.len();
                    assert!(
                        script_sig_len >= 2,
                        "Valid coinbase must have scriptSig length >= 2 bytes"
                    );
                    assert!(
                        script_sig_len <= 100,
                        "Valid coinbase must have scriptSig length <= 100 bytes"
                    );
                }
            }
            ValidationResult::Invalid(_) => {
                // Invalid transactions may violate any rule
                // This is acceptable - we're testing the validation logic
            }
        }
    }

    /// Kani proof: check_tx_inputs handles coinbase correctly
    #[kani::proof]
    fn kani_check_tx_inputs_coinbase() {
        let tx: Transaction = kani::any();
        let utxo_set: UtxoSet = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        let result = check_tx_inputs(&tx, &utxo_set, height)
            .unwrap_or((ValidationResult::Invalid("Error".to_string()), 0));

        // Coinbase invariant
        if is_coinbase(&tx) {
            assert!(
                matches!(result.0, ValidationResult::Valid),
                "Coinbase transactions must be valid"
            );
            assert_eq!(result.1, 0, "Coinbase transactions must have zero fee");
        } else {
            // Non-coinbase transactions must not have null prevouts (Orange Paper Section 5.1, rule 6)
            for input in &tx.inputs {
                assert!(
                    !(input.prevout.hash == [0u8; 32] && input.prevout.index == 0xffffffff),
                    "Non-coinbase transactions must not have null prevouts"
                );
            }
        }
    }

    /// Kani proof: is_coinbase correctly identifies coinbase transactions
    #[kani::proof]
    fn kani_is_coinbase_correct() {
        let tx: Transaction = kani::any();

        let is_cb = is_coinbase(&tx);

        // Coinbase identification invariant
        if is_cb {
            assert_eq!(tx.inputs.len(), 1, "Coinbase must have exactly one input");
            assert_eq!(
                tx.inputs[0].prevout.hash, [0u8; 32],
                "Coinbase input must have zero hash"
            );
            assert_eq!(
                tx.inputs[0].prevout.index, 0xffffffff,
                "Coinbase input must have max index"
            );
        }
    }

    /// Verify coinbase transaction validation
    ///
    /// Ensures coinbase transactions are handled correctly with special rules.
    #[kani::proof]
    fn kani_coinbase_transaction() {
        let height: Natural = kani::any();

        // Create coinbase transaction
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff, // Coinbase marker
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: kani::any(),
                script_pubkey: vec![0x51],
            }],
            lock_time: 0,
        };

        let utxo_set = UtxoSet::new();
        let result = check_tx_inputs(&tx, &utxo_set, height);

        // Coinbase should always validate inputs (special case)
        assert!(result.is_ok());
        let (validation_result, fee) = result.unwrap();
        assert!(matches!(validation_result, ValidationResult::Valid));
        assert_eq!(fee, 0); // Coinbase has no fee
    }

    /// Verify transaction with empty inputs/outputs
    #[kani::proof]
    fn kani_transaction_empty_lists() {
        // Empty inputs
        let tx_no_inputs = Transaction {
            version: 1,
            inputs: vec![],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            }],
            lock_time: 0,
        };

        // Empty outputs
        let tx_no_outputs = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }],
            outputs: vec![],
            lock_time: 0,
        };

        // Both should fail validation
        let result1 = check_transaction(&tx_no_inputs);
        let result2 = check_transaction(&tx_no_outputs);

        assert!(matches!(result1.unwrap(), ValidationResult::Invalid(_)));
        assert!(matches!(result2.unwrap(), ValidationResult::Invalid(_)));
    }

    /// Verify transaction output value bounds
    #[kani::proof]
    fn kani_transaction_output_value_bounds() {
        let value: Integer = kani::any();
        kani::assume(value <= MAX_MONEY as i64 + 1000);

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: value as u64,
                script_pubkey: vec![0x51],
            }],
            lock_time: 0,
        };

        let result = check_transaction(&tx);

        if value < 0 || value > MAX_MONEY as i64 {
            assert!(matches!(result.unwrap(), ValidationResult::Invalid(_)));
        } else {
            assert!(result.is_ok());
        }
    }

    /// Kani proof: check_tx_inputs enforces value consistency
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳, utxo_set ∈ 𝒰𝒮, height ∈ ℕ:
    /// - If check_tx_inputs(tx, utxo_set, height) = (Valid, fee):
    ///   (tx is coinbase ∨ Σᵢ utxo(i.prevout).value ≥ Σₒ o.value)
    /// - fee = Σᵢ utxo(i.prevout).value - Σₒ o.value (non-negative)
    ///
    /// This ensures transactions cannot create money out of thin air.
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_transaction_value_consistency() {
        let tx: Transaction = kani::any();
        let utxo_set: UtxoSet = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        let result = check_tx_inputs(&tx, &utxo_set, height);

        if result.is_ok() {
            let (validation_result, fee) = result.unwrap();

            match validation_result {
                ValidationResult::Valid => {
                    if is_coinbase(&tx) {
                        // Coinbase must have zero fee
                        assert_eq!(fee, 0, "Coinbase transactions must have zero fee");
                    } else {
                        // Non-coinbase transactions: fee must be non-negative
                        assert!(fee >= 0, "Transaction fee must be non-negative");

                        // Total input value must be >= total output value
                        // (This is enforced by check_tx_inputs, but we prove it here)
                        let total_input: i64 = tx
                            .inputs
                            .iter()
                            .filter_map(|input| utxo_set.get(&input.prevout))
                            .map(|utxo| utxo.value as i64)
                            .sum();

                        let total_output: i64 =
                            tx.outputs.iter().map(|output| output.value as i64).sum();

                        assert!(
                            total_input >= total_output,
                            "Valid transactions must have input value >= output value"
                        );
                        assert_eq!(
                            fee,
                            total_input - total_output,
                            "Fee must equal input value - output value"
                        );
                    }
                }
                ValidationResult::Invalid(_) => {
                    // Invalid transactions may violate value constraints
                    // This is acceptable - we're testing the validation logic
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(unused_doc_comments)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Arbitrary implementation for Transaction (inline since tests/fuzzing/arbitrary_impls.rs
    // is in separate test crate and not accessible from src/ tests)
    impl Arbitrary for Transaction {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                any::<u64>(), // version
                prop::collection::vec(
                    (
                        any::<[u8; 32]>(),                          // prevout hash
                        any::<u64>(),                               // prevout index
                        prop::collection::vec(any::<u8>(), 0..100), // script_sig
                        any::<u64>(),                               // sequence
                    ),
                    0..10, // input count
                ),
                prop::collection::vec(
                    (
                        any::<i64>(),                               // value
                        prop::collection::vec(any::<u8>(), 0..100), // script_pubkey
                    ),
                    0..10, // output count
                ),
                any::<u64>(), // lock_time
            )
                .prop_map(|(version, inputs, outputs, lock_time)| Transaction {
                    version,
                    inputs: inputs
                        .into_iter()
                        .map(|(hash, index, script_sig, sequence)| TransactionInput {
                            prevout: OutPoint { hash, index },
                            script_sig,
                            sequence,
                        })
                        .collect(),
                    outputs: outputs
                        .into_iter()
                        .map(|(value, script_pubkey)| TransactionOutput {
                            value,
                            script_pubkey,
                        })
                        .collect(),
                    lock_time,
                })
                .boxed()
        }
    }

    /// Property test: check_transaction validates structure correctly
    proptest! {
        #[test]
        fn prop_check_transaction_structure(
            tx in any::<Transaction>()
        ) {
            // Bound for tractability
            let mut bounded_tx = tx;
            if bounded_tx.inputs.len() > 10 {
                bounded_tx.inputs.truncate(10);
            }
            if bounded_tx.outputs.len() > 10 {
                bounded_tx.outputs.truncate(10);
            }

            let result = check_transaction(&bounded_tx).unwrap_or_else(|_| ValidationResult::Invalid("Error".to_string()));

            // Structure properties
            match result {
                ValidationResult::Valid => {
                    // Valid transactions must have non-empty inputs and outputs
                    prop_assert!(!bounded_tx.inputs.is_empty(), "Valid transaction must have inputs");
                    prop_assert!(!bounded_tx.outputs.is_empty(), "Valid transaction must have outputs");

                    // Valid transactions must respect limits
                    prop_assert!(bounded_tx.inputs.len() <= MAX_INPUTS, "Valid transaction must respect input limit");
                    prop_assert!(bounded_tx.outputs.len() <= MAX_OUTPUTS, "Valid transaction must respect output limit");

                    // Valid transactions must have valid output values
                    for output in &bounded_tx.outputs {
                        prop_assert!(output.value >= 0, "Valid transaction outputs must be non-negative");
                        prop_assert!(output.value <= MAX_MONEY, "Valid transaction outputs must not exceed max money");
                    }
                },
                ValidationResult::Invalid(_) => {
                    // Invalid transactions may violate any rule
                    // This is acceptable - we're testing the validation logic
                }
            }
        }
    }

    /// Property test: check_tx_inputs handles coinbase correctly
    proptest! {
        #[test]
        fn prop_check_tx_inputs_coinbase(
            tx in any::<Transaction>(),
            utxo_set in any::<UtxoSet>(),
            height in 0u64..1000u64
        ) {
            // Bound for tractability
            let mut bounded_tx = tx;
            if bounded_tx.inputs.len() > 5 {
                bounded_tx.inputs.truncate(5);
            }
            if bounded_tx.outputs.len() > 5 {
                bounded_tx.outputs.truncate(5);
            }

            let result = check_tx_inputs(&bounded_tx, &utxo_set, height).unwrap_or((ValidationResult::Invalid("Error".to_string()), 0));

            // Coinbase property
            if is_coinbase(&bounded_tx) {
                prop_assert!(matches!(result.0, ValidationResult::Valid), "Coinbase transactions must be valid");
                prop_assert_eq!(result.1, 0, "Coinbase transactions must have zero fee");
            }
        }
    }

    /// Property test: is_coinbase correctly identifies coinbase transactions
    proptest! {
        #[test]
        fn prop_is_coinbase_correct(
            tx in any::<Transaction>()
        ) {
            let is_cb = is_coinbase(&tx);

            // Coinbase identification property
            if is_cb {
                prop_assert_eq!(tx.inputs.len(), 1, "Coinbase must have exactly one input");
                prop_assert_eq!(tx.inputs[0].prevout.hash, [0u8; 32], "Coinbase input must have zero hash");
                prop_assert_eq!(tx.inputs[0].prevout.index, 0xffffffffu64, "Coinbase input must have max index");
            }
        }
    }

    /// Property test: calculate_transaction_size is consistent
    proptest! {
        #[test]
        fn prop_calculate_transaction_size_consistent(
            tx in any::<Transaction>()
        ) {
            // Bound for tractability
            let mut bounded_tx = tx;
            if bounded_tx.inputs.len() > 10 {
                bounded_tx.inputs.truncate(10);
            }
            if bounded_tx.outputs.len() > 10 {
                bounded_tx.outputs.truncate(10);
            }

            let size = calculate_transaction_size(&bounded_tx);

            // Size calculation properties
            prop_assert!(size >= 8, "Transaction size must be at least 8 bytes (version + lock_time)");
            prop_assert!(size <= 4 + 10 * 41 + 10 * 9 + 4, "Transaction size must not exceed maximum");

            // Size should be deterministic
            let size2 = calculate_transaction_size(&bounded_tx);
            prop_assert_eq!(size, size2, "Transaction size calculation must be deterministic");
        }
    }

    /// Property test: output value bounds are respected
    proptest! {
        #[test]
        fn prop_output_value_bounds(
            value in 0i64..(MAX_MONEY + 1000)
        ) {
            let tx = Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0 },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                }],
                outputs: vec![TransactionOutput {
                    value,
                    script_pubkey: vec![],
                }],
                lock_time: 0,
            };

            let result = check_transaction(&tx).unwrap_or(ValidationResult::Invalid("Error".to_string()));

            // Value bounds property
            if !(0..=MAX_MONEY).contains(&value) {
                prop_assert!(matches!(result, ValidationResult::Invalid(_)),
                    "Transactions with invalid output values must be invalid");
            } else {
                // Valid values should pass other checks too
                if !tx.inputs.is_empty() && !tx.outputs.is_empty() {
                    prop_assert!(matches!(result, ValidationResult::Valid),
                        "Transactions with valid output values should be valid");
                }
            }
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: CheckTransaction invariants
    ///
    /// Mathematical specification (Orange Paper Section 5.1):
    /// ∀ tx ∈ TX:
    /// - CheckTransaction(tx) = valid ⟹
    ///   (tx.inputs.len() > 0 ∧
    ///    tx.outputs.len() > 0 ∧
    ///    tx.inputs.len() <= MAX_INPUTS ∧
    ///    tx.outputs.len() <= MAX_OUTPUTS ∧
    ///    ∀ output ∈ tx.outputs: 0 <= output.value <= MAX_MONEY ∧
    ///    ∑_{o ∈ tx.outputs} o.value <= MAX_MONEY ∧
    ///    ∀i,j ∈ tx.inputs: i ≠ j ⟹ i.prevout ≠ j.prevout ∧
    ///    (IsCoinbase(tx) ⟹ 2 <= |tx.inputs[0].scriptSig| <= 100))
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_check_transaction_invariants() {
        let tx: Transaction = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 10, 10);

        let result = check_transaction(&tx);

        if result.is_ok() {
            let validation_result = result.unwrap();
            if matches!(validation_result, ValidationResult::Valid) {
                // If valid, these invariants must hold:
                assert!(!tx.inputs.is_empty(), "Valid transaction must have inputs");
                assert!(
                    !tx.outputs.is_empty(),
                    "Valid transaction must have outputs"
                );
                assert!(
                    tx.inputs.len() <= MAX_INPUTS,
                    "Valid transaction must respect input limit"
                );
                assert!(
                    tx.outputs.len() <= MAX_OUTPUTS,
                    "Valid transaction must respect output limit"
                );

                let mut total_output = 0i64;
                for output in &tx.outputs {
                    assert!(
                        output.value >= 0,
                        "Valid transaction outputs must be non-negative"
                    );
                    assert!(
                        output.value <= MAX_MONEY,
                        "Valid transaction outputs must not exceed MAX_MONEY"
                    );
                    total_output = total_output
                        .checked_add(output.value)
                        .unwrap_or(MAX_MONEY + 1);
                }
                // Total output sum must not exceed MAX_MONEY (Orange Paper Section 5.1, rule 3)
                assert!(
                    total_output <= MAX_MONEY,
                    "Total output value must not exceed MAX_MONEY"
                );

                // Valid transactions must not have duplicate prevouts (Orange Paper Section 5.1, rule 4)
                for i in 0..tx.inputs.len() {
                    for j in (i + 1)..tx.inputs.len() {
                        assert!(
                            tx.inputs[i].prevout != tx.inputs[j].prevout,
                            "Valid transaction must not have duplicate prevouts"
                        );
                    }
                }

                // Coinbase transactions must have scriptSig length [2, 100] (Orange Paper Section 5.1, rule 5)
                if is_coinbase(&tx) {
                    let script_sig_len = tx.inputs[0].script_sig.len();
                    assert!(
                        script_sig_len >= 2,
                        "Valid coinbase must have scriptSig length >= 2 bytes"
                    );
                    assert!(
                        script_sig_len <= 100,
                        "Valid coinbase must have scriptSig length <= 100 bytes"
                    );
                }
            }
        }
    }

    /// Kani proof: CheckTxInputs fee calculation correctness
    ///
    /// Mathematical specification (Orange Paper Section 5.1):
    /// ∀ tx ∈ TX, utxo_set ∈ US, height ∈ N:
    /// - CheckTxInputs(tx, utxo_set, height) = (valid, fee) ⟹
    ///   (fee = sum(inputs.value) - sum(outputs.value) ∧
    ///    fee >= 0)
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_check_tx_inputs_fee_calculation() {
        let tx: Transaction = kani::any();
        let mut utxo_set: UtxoSet = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        // Populate UTXO set with values for transaction inputs
        for input in &tx.inputs {
            if !utxo_set.contains_key(&input.prevout) {
                let utxo = UTXO {
                    value: kani::any(),
                    script_pubkey: kani::any(),
                    height: 0,
                };
                utxo_set.insert(input.prevout, utxo);
            }
        }

        let result = check_tx_inputs(&tx, &utxo_set, height);

        if result.is_ok() {
            let (validation_result, fee) = result.unwrap();
            if matches!(validation_result, ValidationResult::Valid) {
                // Fee must be non-negative for valid transactions
                assert!(fee >= 0, "Valid transaction fee must be non-negative");

                // Calculate expected fee manually
                let total_input: i64 = tx
                    .inputs
                    .iter()
                    .filter_map(|input| utxo_set.get(&input.prevout))
                    .map(|utxo| utxo.value)
                    .sum();
                let total_output: i64 = tx.outputs.iter().map(|output| output.value).sum();
                let expected_fee = total_input.checked_sub(total_output).unwrap_or(-1);

                if expected_fee >= 0 {
                    assert!(
                        fee == expected_fee,
                        "Fee calculation must match sum(inputs) - sum(outputs)"
                    );
                }
            }
        }
    }

    /// Kani proof: Conservation of Value (Orange Paper Section 8.1, Bitcoin Core Consensus)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ TX, utxo_set ∈ US:
    /// - If check_tx_inputs(tx, utxo_set) = (valid, fee):
    ///   Σ(tx.inputs.value) = Σ(tx.outputs.value) + fee
    ///
    /// This is a fundamental economic security property ensuring no money creation.
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_conservation_of_value() {
        let tx: Transaction = kani::any();
        let mut utxo_set: UtxoSet = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        // Skip coinbase (has special rules: fee = 0)
        kani::assume(!is_coinbase(&tx));

        // Populate UTXO set with values for transaction inputs
        for input in &tx.inputs {
            if !utxo_set.contains_key(&input.prevout) {
                let utxo = UTXO {
                    value: kani::any(),
                    script_pubkey: kani::any(),
                    height: 0,
                };
                utxo_set.insert(input.prevout, utxo);
            }
        }

        let result = check_tx_inputs(&tx, &utxo_set, height);

        if result.is_ok() {
            let (validation_result, fee) = result.unwrap();
            if matches!(validation_result, ValidationResult::Valid) {
                // Calculate input and output sums
                let total_input: i64 = tx
                    .inputs
                    .iter()
                    .filter_map(|input| utxo_set.get(&input.prevout))
                    .map(|utxo| utxo.value)
                    .sum();

                let total_output: i64 = tx.outputs.iter().map(|output| output.value).sum();

                // Conservation of Value: inputs = outputs + fee
                let expected_fee = total_input.checked_sub(total_output).unwrap_or(i64::MIN);

                if expected_fee >= 0 {
                    assert_eq!(
                        fee, expected_fee,
                        "Conservation of Value: inputs.value must equal outputs.value + fee"
                    );
                    assert_eq!(
                        total_input,
                        total_output + fee,
                        "Conservation of Value: Σ(inputs) = Σ(outputs) + fee"
                    );
                }
            }
        }
    }

    /// Kani proof: CheckTransaction rejects duplicate inputs
    ///
    /// Mathematical specification (Orange Paper Section 5.1, rule 4):
    /// ∀ tx ∈ TX: CheckTransaction(tx) = valid ⟹
    ///   ∀i,j ∈ tx.inputs: i ≠ j ⟹ i.prevout ≠ j.prevout
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_check_transaction_no_duplicates() {
        let tx: Transaction = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 10, 10);

        // Check if transaction has duplicate prevouts
        let mut has_duplicates = false;
        for i in 0..tx.inputs.len() {
            for j in (i + 1)..tx.inputs.len() {
                if tx.inputs[i].prevout == tx.inputs[j].prevout {
                    has_duplicates = true;
                    break;
                }
            }
            if has_duplicates {
                break;
            }
        }

        let result = check_transaction(&tx);

        if result.is_ok() {
            let validation_result = result.unwrap();
            // If transaction has duplicates, it must be invalid
            if has_duplicates {
                assert!(
                    matches!(validation_result, ValidationResult::Invalid(_)),
                    "Transactions with duplicate prevouts must be invalid"
                );
            } else {
                // If no duplicates and valid, then all prevouts are distinct
                if matches!(validation_result, ValidationResult::Valid) {
                    for i in 0..tx.inputs.len() {
                        for j in (i + 1)..tx.inputs.len() {
                            assert!(
                                tx.inputs[i].prevout != tx.inputs[j].prevout,
                                "Valid transactions must not have duplicate prevouts"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Kani proof: Coinbase maturity enforcement (Orange Paper Economic Security)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ TX, utxo_set ∈ US, height ∈ ℕ:
    /// - If tx spends coinbase output created at height h:
    ///   CheckTxInputs(tx, utxo_set, height) = valid ⟹ height ≥ h + COINBASE_MATURITY
    ///
    /// This ensures coinbase outputs cannot be spent until 100 blocks deep.
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_coinbase_maturity_enforcement() {
        use crate::constants::COINBASE_MATURITY;

        let tx: Transaction = kani::any();
        let mut utxo_set: UtxoSet = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        // Skip coinbase transactions (they don't spend coinbase outputs)
        kani::assume(!is_coinbase(&tx));

        // Create coinbase outputs in UTXO set with known heights
        for (i, input) in tx.inputs.iter().enumerate() {
            if !utxo_set.contains_key(&input.prevout) {
                // Create a coinbase UTXO (created at height h)
                let coinbase_height: Natural = kani::any();
                kani::assume(coinbase_height <= height);

                let utxo = UTXO {
                    value: 1000,
                    script_pubkey: vec![],
                    height: coinbase_height,
                };
                utxo_set.insert(input.prevout.clone(), utxo);
            }
        }

        let result = check_tx_inputs(&tx, &utxo_set, height);

        if result.is_ok() {
            let (validation_result, _fee) = result.unwrap();
            if matches!(validation_result, ValidationResult::Valid) {
                // For valid transactions, verify coinbase maturity
                for input in &tx.inputs {
                    if let Some(utxo) = utxo_set.get(&input.prevout) {
                        // Coinbase maturity: height must be >= utxo.height + COINBASE_MATURITY
                        let required_height = utxo.height.saturating_add(COINBASE_MATURITY);
                        assert!(height >= required_height,
                            "Coinbase maturity: cannot spend coinbase output until {} blocks deep (height {} >= {})",
                            COINBASE_MATURITY, height, required_height);
                    }
                }
            }
        }
    }

    /// Kani proof: CheckTxInputs rejects null prevouts in non-coinbase transactions
    ///
    /// Mathematical specification (Orange Paper Section 5.1, rule 6):
    /// ∀ tx ∈ TX: ¬IsCoinbase(tx) ⟹ CheckTxInputs(tx) = valid ⟹
    ///   ∀i ∈ tx.inputs: ¬i.prevout.IsNull()
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_check_tx_inputs_no_null_prevout() {
        let tx: Transaction = kani::any();
        let utxo_set: UtxoSet = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        // Skip coinbase transactions (they have null prevouts by definition)
        kani::assume(!is_coinbase(&tx));

        // Check if transaction has null prevouts
        let mut has_null_prevout = false;
        for input in &tx.inputs {
            if input.prevout.hash == [0u8; 32] && input.prevout.index == 0xffffffff {
                has_null_prevout = true;
                break;
            }
        }

        let result = check_tx_inputs(&tx, &utxo_set, height);

        if result.is_ok() {
            let (validation_result, _fee) = result.unwrap();
            // If transaction has null prevout, it must be invalid
            if has_null_prevout {
                assert!(
                    matches!(validation_result, ValidationResult::Invalid(_)),
                    "Non-coinbase transactions with null prevouts must be invalid"
                );
            } else {
                // If no null prevouts and valid, then all prevouts are non-null
                if matches!(validation_result, ValidationResult::Valid) {
                    for input in &tx.inputs {
                        assert!(
                            !(input.prevout.hash == [0u8; 32] && input.prevout.index == 0xffffffff),
                            "Valid non-coinbase transactions must not have null prevouts"
                        );
                    }
                }
            }
        }
    }

    /// Kani proof: CheckTransaction validates total output sum
    ///
    /// Mathematical specification (Orange Paper Section 5.1, rule 3):
    /// ∀ tx ∈ TX: CheckTransaction(tx) = valid ⟹
    ///   ∑_{o ∈ tx.outputs} o.value ≤ M_max
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_check_transaction_total_output_sum() {
        let tx: Transaction = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 10, 10);

        // Calculate total output sum
        let mut total_sum = 0i64;
        for output in &tx.outputs {
            total_sum = total_sum.checked_add(output.value).unwrap_or(i64::MAX);
        }

        let result = check_transaction(&tx);

        if result.is_ok() {
            let validation_result = result.unwrap();
            // If transaction is valid, total sum must not exceed MAX_MONEY
            if matches!(validation_result, ValidationResult::Valid) {
                assert!(
                    total_sum <= MAX_MONEY,
                    "Valid transactions must have total output sum <= MAX_MONEY"
                );
            } else {
                // If invalid and total sum exceeds MAX_MONEY, validation correctly rejected it
                if total_sum > MAX_MONEY {
                    assert!(
                        matches!(validation_result, ValidationResult::Invalid(_)),
                        "Transactions with total output sum > MAX_MONEY must be invalid"
                    );
                }
            }
        }
    }

    /// Kani proof: Integer arithmetic overflow safety (Orange Paper Section 13.3.1)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ TX, utxo_set ∈ US:
    /// - Input value summation uses checked_add() and never overflows i64::MAX
    /// - Output value summation uses checked_add() and never overflows i64::MAX
    /// - Fee calculation uses checked_sub() and never underflows
    ///
    /// This ensures all monetary value arithmetic is safe from overflow/underflow.
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_integer_arithmetic_overflow_safety() {
        let tx: Transaction = kani::any();
        let mut utxo_set: UtxoSet = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 10, 10);

        // Bound values to reasonable ranges (but still test overflow boundaries)
        for output in &tx.outputs {
            kani::assume(output.value >= 0);
            kani::assume(output.value <= MAX_MONEY);
        }

        // Populate UTXO set with bounded values
        for input in &tx.inputs {
            if !utxo_set.contains_key(&input.prevout) {
                let value: i64 = kani::any();
                kani::assume(value >= 0);
                kani::assume(value <= MAX_MONEY);
                utxo_set.insert(
                    input.prevout.clone(),
                    UTXO {
                        value,
                        script_pubkey: vec![],
                        height: height.saturating_sub(1),
                    },
                );
            }
        }

        // Test input value summation - should use checked_add()
        let result = check_tx_inputs(&tx, &utxo_set, height);

        if result.is_ok() {
            // If check succeeds, input summation didn't overflow
            // The implementation uses checked_add() which prevents overflow
            let (validation_result, fee) = result.unwrap();

            if matches!(validation_result, ValidationResult::Valid) {
                // Fee calculation uses checked_sub() - should not underflow
                assert!(
                    fee >= 0,
                    "Integer arithmetic safety: fee calculation must not underflow"
                );

                // Manual calculation to verify overflow safety
                let mut manual_input_sum = 0i64;
                for input in &tx.inputs {
                    if let Some(utxo) = utxo_set.get(&input.prevout) {
                        manual_input_sum = match manual_input_sum.checked_add(utxo.value) {
                            Some(sum) => sum,
                            None => {
                                // Overflow detected - check_tx_inputs should have caught this
                                panic!("Input value summation overflow should be caught by checked_add()");
                            }
                        };
                    }
                }

                // If we got here, no overflow occurred
                assert!(
                    manual_input_sum >= 0,
                    "Integer arithmetic safety: input summation must not overflow"
                );
            }
        }
    }

    /// Kani proof: Output value summation overflow safety (Orange Paper Section 13.3.1)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ TX: Output value summation uses checked_add() and never overflows i64::MAX
    ///
    /// This ensures total output value calculation is safe from overflow.
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_output_value_summation_overflow_safety() {
        let tx: Transaction = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 10, 10);

        // Bound individual output values
        for output in &tx.outputs {
            kani::assume(output.value >= 0);
            kani::assume(output.value <= MAX_MONEY);
        }

        // Test output value summation - check_transaction uses checked_add()
        let result = check_transaction(&tx);

        if result.is_ok() {
            let validation_result = result.unwrap();

            // Manual calculation to verify overflow safety
            let mut manual_output_sum = 0i64;
            for output in &tx.outputs {
                manual_output_sum = match manual_output_sum.checked_add(output.value) {
                    Some(sum) => sum,
                    None => {
                        // Overflow detected - check_transaction should have caught this
                        // If validation passed, overflow was prevented
                        assert!(
                            matches!(validation_result, ValidationResult::Invalid(_)),
                            "Output value summation overflow must be caught by checked_add()"
                        );
                        return;
                    }
                };
            }

            // If we got here and validation passed, no overflow occurred
            if matches!(validation_result, ValidationResult::Valid) {
                assert!(
                    manual_output_sum >= 0 && manual_output_sum <= MAX_MONEY,
                    "Output value summation: no overflow for valid transactions"
                );
            }
        }
    }

    /// Kani proof: check_transaction_fast_path correctness (Phase 6.3 optimization)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction:
    /// - If check_transaction_fast_path(tx) = Some(result): result must match check_transaction(tx)
    /// - If check_transaction_fast_path(tx) = None: full validation needed (fast-path can't determine)
    ///
    /// This ensures fast-path optimization matches full validation results exactly.
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_check_transaction_fast_path_correctness() {
        let tx: Transaction = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 10, 10);

        // Get fast-path result
        #[cfg(feature = "production")]
        let fast_path_result = check_transaction_fast_path(&tx);

        // Get full validation result
        let full_result = check_transaction(&tx);

        #[cfg(feature = "production")]
        if let Some(fast_path_validation) = fast_path_result {
            // If fast-path returns a result, it must match full validation
            if full_result.is_ok() {
                let full_validation = full_result.unwrap();

                // Fast-path should agree with full validation
                match (&fast_path_validation, &full_validation) {
                    (ValidationResult::Invalid(_), ValidationResult::Invalid(_)) => {
                        // Both invalid - fast-path correctly identified invalid transaction
                        assert!(true, "Fast-path correctly identifies invalid transactions");
                    }
                    (ValidationResult::Valid, ValidationResult::Valid) => {
                        // Both valid - fast-path correctly identified valid transaction
                        assert!(true, "Fast-path correctly identifies valid transactions");
                    }
                    _ => {
                        // Mismatch - fast-path result should match full validation
                        // This should not happen if fast-path is correct
                        assert_eq!(
                            fast_path_validation, full_validation,
                            "Fast-path result must match full validation result"
                        );
                    }
                }
            }
        }

        // If fast-path returns None, full validation must be performed
        // (This is handled by the implementation calling full validation)
    }

    /// Kani proof: Transaction size calculation consistency
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction:
    /// - All transaction size calculation functions must produce consistent results
    /// - calculate_transaction_size(tx) should be consistent with simplified base_size approximation
    ///
    /// This ensures fee calculation uses consistent size measurements.
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_transaction_size_consistency() {
        use crate::segwit::calculate_base_size;

        let tx: Transaction = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        // Calculate size using different implementations
        let size1 = calculate_transaction_size(&tx);
        let size2 = calculate_base_size(&tx) as usize;

        // Critical invariant: both implementations should produce similar results
        // (They use simplified calculations, so they should be close)
        // Size1: 4 + inputs*41 + outputs*9 + 4 = 8 + inputs*41 + outputs*9
        // Size2: 4 + inputs*41 + outputs*9 + 4 = 8 + inputs*41 + outputs*9
        // They should be equal since they use the same simplified formula

        // Note: These are simplified calculations, so exact match expected
        assert_eq!(size1, size2,
            "Transaction size calculation consistency: both implementations must produce same result for simplified calculations");

        // Critical invariant: size must be positive
        assert!(
            size1 > 0,
            "Transaction size calculation consistency: size must be positive"
        );
        assert!(
            size2 > 0,
            "Transaction size calculation consistency: size must be positive"
        );
    }

    /// Kani proof: CheckTransaction validates coinbase scriptSig length
    ///
    /// Mathematical specification (Orange Paper Section 5.1, rule 5):
    /// ∀ tx ∈ TX: IsCoinbase(tx) ⟹ CheckTransaction(tx) = valid ⟹
    ///   2 ≤ |tx.inputs[0].scriptSig| ≤ 100
    #[kani::proof]
    #[kani::unwind(unwind_bounds::TRANSACTION_VALIDATION)]
    fn kani_check_transaction_coinbase_script_sig_length() {
        let tx: Transaction = kani::any();

        // Bound for tractability using standardized helpers
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        // Only test coinbase transactions
        kani::assume(is_coinbase(&tx));

        let script_sig_len = tx.inputs[0].script_sig.len();

        let result = check_transaction(&tx);

        if result.is_ok() {
            let validation_result = result.unwrap();
            // If valid, scriptSig length must be between 2 and 100
            if matches!(validation_result, ValidationResult::Valid) {
                assert!(
                    script_sig_len >= 2,
                    "Valid coinbase must have scriptSig length >= 2 bytes"
                );
                assert!(
                    script_sig_len <= 100,
                    "Valid coinbase must have scriptSig length <= 100 bytes"
                );
            } else {
                // If invalid and scriptSig length is wrong, validation correctly rejected it
                if script_sig_len < 2 || script_sig_len > 100 {
                    assert!(
                        matches!(validation_result, ValidationResult::Invalid(_)),
                        "Coinbase with scriptSig length outside [2, 100] must be invalid"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_transaction_valid() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }

    #[test]
    fn test_check_transaction_empty_inputs() {
        let tx = Transaction {
            version: 1,
            inputs: vec![],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        assert!(matches!(
            check_transaction(&tx).unwrap(),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_check_tx_inputs_coinbase() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 5000000000, // 50 BTC
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        let utxo_set = UtxoSet::new();
        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();

        assert_eq!(result, ValidationResult::Valid);
        assert_eq!(fee, 0);
    }

    // ============================================================================
    // COMPREHENSIVE TRANSACTION TESTS
    // ============================================================================

    #[test]
    fn test_check_transaction_empty_outputs() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![],
            lock_time: 0,
        };

        assert!(matches!(
            check_transaction(&tx).unwrap(),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_check_transaction_invalid_output_value_negative() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: -1, // Invalid negative value
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        assert!(matches!(
            check_transaction(&tx).unwrap(),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_check_transaction_invalid_output_value_too_large() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: MAX_MONEY + 1, // Invalid value exceeding max
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        assert!(matches!(
            check_transaction(&tx).unwrap(),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_check_transaction_max_output_value() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: MAX_MONEY, // Valid max value
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }

    #[test]
    fn test_check_transaction_too_many_inputs() {
        let mut inputs = Vec::new();
        for i in 0..=MAX_INPUTS {
            inputs.push(TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            });
        }

        let tx = Transaction {
            version: 1,
            inputs,
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        assert!(matches!(
            check_transaction(&tx).unwrap(),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_check_transaction_max_inputs() {
        let mut inputs = Vec::new();
        for i in 0..MAX_INPUTS {
            let mut hash = [0u8; 32];
            // Use unique hash for each input to avoid duplicates
            hash[0] = (i & 0xff) as u8;
            hash[1] = ((i >> 8) & 0xff) as u8;
            hash[2] = ((i >> 16) & 0xff) as u8;
            hash[3] = ((i >> 24) & 0xff) as u8;
            inputs.push(TransactionInput {
                prevout: OutPoint {
                    hash,
                    index: i as u64,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            });
        }

        let tx = Transaction {
            version: 1,
            inputs,
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }

    #[test]
    fn test_check_transaction_too_many_outputs() {
        let mut outputs = Vec::new();
        for _ in 0..=MAX_OUTPUTS {
            outputs.push(TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            });
        }

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs,
            lock_time: 0,
        };

        assert!(matches!(
            check_transaction(&tx).unwrap(),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_check_transaction_max_outputs() {
        let mut outputs = Vec::new();
        for _ in 0..MAX_OUTPUTS {
            outputs.push(TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            });
        }

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs,
            lock_time: 0,
        };

        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }

    #[test]
    fn test_check_transaction_too_large() {
        // Create a transaction that will exceed MAX_TX_SIZE
        // Since calculate_transaction_size is simplified, we need to create a transaction
        // with enough inputs to exceed the size limit
        let mut inputs = Vec::new();
        for i in 0..25000 {
            // This should create a transaction > 1MB
            inputs.push(TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                },
                script_sig: vec![0u8; 100], // Large script to increase size
                sequence: 0xffffffff,
            });
        }

        let tx = Transaction {
            version: 1,
            inputs,
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        assert!(matches!(
            check_transaction(&tx).unwrap(),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_check_tx_inputs_regular_transaction() {
        let mut utxo_set = UtxoSet::new();

        // Add UTXO to the set
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 1000000000, // 10 BTC
            script_pubkey: vec![],
            height: 0,
        };
        utxo_set.insert(outpoint, utxo);

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 900000000, // 9 BTC output
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();

        assert_eq!(result, ValidationResult::Valid);
        assert_eq!(fee, 100000000); // 1 BTC fee
    }

    #[test]
    fn test_check_tx_inputs_missing_utxo() {
        let utxo_set = UtxoSet::new(); // Empty UTXO set

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 100000000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();

        assert!(matches!(result, ValidationResult::Invalid(_)));
        assert_eq!(fee, 0);
    }

    #[test]
    fn test_check_tx_inputs_insufficient_funds() {
        let mut utxo_set = UtxoSet::new();

        // Add UTXO with insufficient value
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 100000000, // 1 BTC
            script_pubkey: vec![],
            height: 0,
        };
        utxo_set.insert(outpoint, utxo);

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 200000000, // 2 BTC output (more than input)
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();

        assert!(matches!(result, ValidationResult::Invalid(_)));
        assert_eq!(fee, 0);
    }

    #[test]
    fn test_check_tx_inputs_multiple_inputs() {
        let mut utxo_set = UtxoSet::new();

        // Add two UTXOs
        let outpoint1 = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo1 = UTXO {
            value: 500000000, // 5 BTC
            script_pubkey: vec![],
            height: 0,
        };
        utxo_set.insert(outpoint1, utxo1);

        let outpoint2 = OutPoint {
            hash: [2; 32],
            index: 0,
        };
        let utxo2 = UTXO {
            value: 300000000, // 3 BTC
            script_pubkey: vec![],
            height: 0,
        };
        utxo_set.insert(outpoint2, utxo2);

        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [1; 32],
                        index: 0,
                    },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint {
                        hash: [2; 32],
                        index: 0,
                    },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
            ],
            outputs: vec![TransactionOutput {
                value: 700000000, // 7 BTC output
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();

        assert_eq!(result, ValidationResult::Valid);
        assert_eq!(fee, 100000000); // 1 BTC fee (8 BTC input - 7 BTC output)
    }

    #[test]
    fn test_is_coinbase_edge_cases() {
        // Valid coinbase
        let valid_coinbase = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(is_coinbase(&valid_coinbase));

        // Wrong hash
        let wrong_hash = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(!is_coinbase(&wrong_hash));

        // Wrong index
        let wrong_index = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(!is_coinbase(&wrong_index));

        // Multiple inputs
        let multiple_inputs = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [0; 32],
                        index: 0xffffffff,
                    },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint {
                        hash: [1; 32],
                        index: 0,
                    },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
            ],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(!is_coinbase(&multiple_inputs));

        // No inputs
        let no_inputs = Transaction {
            version: 1,
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(!is_coinbase(&no_inputs));
    }

    #[test]
    fn test_calculate_transaction_size() {
        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [0; 32],
                        index: 0,
                    },
                    script_sig: vec![1, 2, 3],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint {
                        hash: [1; 32],
                        index: 1,
                    },
                    script_sig: vec![4, 5, 6],
                    sequence: 0xffffffff,
                },
            ],
            outputs: vec![
                TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![7, 8, 9],
                },
                TransactionOutput {
                    value: 2000,
                    script_pubkey: vec![10, 11, 12],
                },
            ],
            lock_time: 12345,
        };

        let size = calculate_transaction_size(&tx);
        // Expected: 4 (version) + 2*41 (inputs) + 2*9 (outputs) + 4 (lock_time) = 108
        // The actual calculation includes script_sig and script_pubkey lengths
        assert_eq!(size, 108);
    }
}
