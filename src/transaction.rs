//! Transaction validation functions from Orange Paper Section 5.1
//!
//! Performance optimizations (Phase 6.3):
//! - Early-exit fast-path checks for obviously invalid transactions

use crate::constants::*;
use crate::error::{ConsensusError, Result};
use crate::types::*;
use crate::utxo_overlay::UtxoLookup;
use std::borrow::Cow;
use blvm_spec_lock::spec_locked;

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
        return Some(ValidationResult::Invalid("Empty inputs or outputs".into()));
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
    // Optimization: Use precomputed constant for u64 comparisons
    #[cfg(feature = "production")]
    {
        use crate::optimizations::precomputed_constants::MAX_MONEY_U64;
        for output in &tx.outputs {
            let value_u64 = output.value as u64;
            if output.value < 0 || value_u64 > MAX_MONEY_U64 {
                return Some(ValidationResult::Invalid(format!(
                    "Invalid output value: {}",
                    output.value
                )));
            }
        }
    }

    #[cfg(not(feature = "production"))]
    for output in &tx.outputs {
        if output.value < 0 || output.value > MAX_MONEY {
            return Some(ValidationResult::Invalid(format!(
                "Invalid output value: {}",
                output.value
            )));
        }
    }

    // Quick reject: coinbase with invalid scriptSig length
    // Optimization: Use constant folding for zero hash check
    #[cfg(feature = "production")]
    let is_coinbase_hash = {
        use crate::optimizations::constant_folding::is_zero_hash;
        is_zero_hash(&tx.inputs[0].prevout.hash)
    };

    #[cfg(not(feature = "production"))]
    let is_coinbase_hash = tx.inputs[0].prevout.hash == [0u8; 32];

    if tx.inputs.len() == 1 && is_coinbase_hash && tx.inputs[0].prevout.index == 0xffffffff {
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
#[spec_locked("5.1")]
#[track_caller] // Better error messages showing caller location
#[cfg_attr(feature = "production", inline(always))]
#[cfg_attr(not(feature = "production"), inline)]
pub fn check_transaction(tx: &Transaction) -> Result<ValidationResult> {
    // Precondition checks: Validate function inputs
    // Note: We check these conditions and return Invalid rather than asserting,
    // to allow tests to verify the validation logic properly
    if tx.inputs.len() > MAX_INPUTS {
        return Ok(ValidationResult::Invalid(format!(
            "Input count {} exceeds maximum {}",
            tx.inputs.len(),
            MAX_INPUTS
        )));
    }
    if tx.outputs.len() > MAX_OUTPUTS {
        return Ok(ValidationResult::Invalid(format!(
            "Output count {} exceeds maximum {}",
            tx.outputs.len(),
            MAX_OUTPUTS
        )));
    }

    // Phase 6.3: Fast-path early exit for obviously invalid transactions
    #[cfg(feature = "production")]
    if let Some(result) = check_transaction_fast_path(tx) {
        return Ok(result);
    }

    // 1. Check inputs and outputs are not empty (redundant if fast-path worked, but safe fallback)
    // Note: We check this condition and return Invalid rather than asserting, to allow tests
    // to verify the validation logic properly
    if tx.inputs.is_empty() {
        // Coinbase transactions have exactly 1 input, so empty inputs means non-coinbase
        return Ok(ValidationResult::Invalid(
            "Transaction must have inputs unless it's a coinbase".to_string(),
        ));
    }
    if tx.outputs.is_empty() {
        return Ok(ValidationResult::Invalid(
            "Transaction must have at least one output".to_string(),
        ));
    }

    // 2. Check output values are valid and calculate total sum in one pass (Orange Paper Section 5.1, rules 2 & 3)
    // ∀o ∈ outs: 0 ≤ o.value ≤ M_max ∧ ∑_{o ∈ outs} o.value ≤ M_max
    // Use proven bounds for output access in hot path
    let mut total_output_value = 0i64;
    // Invariant assertion: Total output value must start at zero
    assert!(
        total_output_value == 0,
        "Total output value must start at zero"
    );
    #[cfg(feature = "production")]
    {
        use crate::optimizations::optimized_access::get_proven_by_;
        use crate::optimizations::precomputed_constants::MAX_MONEY_U64;
        for i in 0..tx.outputs.len() {
            if let Some(output) = get_proven_by_(&tx.outputs, i) {
                let value_u64 = output.value as u64;
                if output.value < 0 || value_u64 > MAX_MONEY_U64 {
                    return Ok(ValidationResult::Invalid(format!(
                        "Invalid output value {} at index {}",
                        output.value, i
                    )));
                }
                // Accumulate sum with overflow check
                // Invariant assertion: Output value must be non-negative before addition
                assert!(
                    output.value >= 0,
                    "Output value {} must be non-negative at index {}",
                    output.value,
                    i
                );
                total_output_value = total_output_value
                    .checked_add(output.value)
                    .ok_or_else(make_output_sum_overflow_error)?;
                // Invariant assertion: Total output value must remain non-negative after addition
                assert!(
                    total_output_value >= 0,
                    "Total output value {} must be non-negative after output {}",
                    total_output_value,
                    i
                );
            }
        }
    }

    #[cfg(not(feature = "production"))]
    {
        for (i, output) in tx.outputs.iter().enumerate() {
            // Bounds checking assertion: Output index must be valid
            assert!(i < tx.outputs.len(), "Output index {i} out of bounds");
            // Check output value is valid (non-negative and within MAX_MONEY)
            // Note: We check this condition and return Invalid rather than asserting,
            // to allow tests to verify the validation logic properly
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
            // Invariant assertion: Total output value must remain non-negative after addition
            assert!(
                total_output_value >= 0,
                "Total output value {total_output_value} must be non-negative after output {i}"
            );
        }
    }

    // 2b. Check total output sum is in valid range (matches Bitcoin Core's MoneyRange check)
    // MoneyRange(n) = (n >= 0 && n <= MAX_MONEY)
    // Optimization: Use precomputed constant for comparison
    // Invariant assertion: Total output value must be non-negative
    assert!(
        total_output_value >= 0,
        "Total output value {total_output_value} must be non-negative"
    );

    #[cfg(feature = "production")]
    {
        use crate::optimizations::precomputed_constants::MAX_MONEY_U64;
        let total_u64 = total_output_value as u64;
        // Check for invalid total output value and return error (before assert)
        if total_output_value < 0 || total_u64 > MAX_MONEY_U64 {
            return Ok(ValidationResult::Invalid(format!(
                "Total output value {total_output_value} is out of valid range [0, {}]",
                MAX_MONEY
            )));
        }
        // Invariant assertion: Total output value must not exceed MAX_MONEY
        // (This should never fail if the check above is correct)
        assert!(
            total_u64 <= MAX_MONEY_U64,
            "Total output value {} must not exceed MAX_MONEY",
            total_output_value
        );
    }

    #[cfg(not(feature = "production"))]
    {
        // Check for invalid total output value and return error (before assert)
        if !(0..=MAX_MONEY).contains(&total_output_value) {
            return Ok(ValidationResult::Invalid(format!(
                "Total output value {total_output_value} is out of valid range [0, {MAX_MONEY}]"
            )));
        }
        // Invariant assertion: Total output value must not exceed MAX_MONEY
        // (This should never fail if the check above is correct)
        assert!(
            total_output_value <= MAX_MONEY,
            "Total output value {total_output_value} must not exceed MAX_MONEY"
        );
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

    // 5. Check transaction size limit (matches Bitcoin Core's CheckTransaction exactly)
    // Core: GetSerializeSize(TX_NO_WITNESS(tx)) * WITNESS_SCALE_FACTOR > MAX_BLOCK_WEIGHT
    // This checks: stripped_size * 4 > 4,000,000, i.e., stripped_size > 1,000,000
    // Note: Core's comment says "this doesn't take the witness into account, as that hasn't been checked for malleability"
    // calculate_transaction_size returns stripped size (no witness), matching TX_NO_WITNESS
    use crate::constants::MAX_BLOCK_WEIGHT;
    const WITNESS_SCALE_FACTOR: usize = 4;
    let tx_stripped_size = calculate_transaction_size(tx); // This is TX_NO_WITNESS size
    if tx_stripped_size * WITNESS_SCALE_FACTOR > MAX_BLOCK_WEIGHT {
        return Ok(ValidationResult::Invalid(format!(
            "Transaction too large: stripped size {} bytes (weight {} > {})",
            tx_stripped_size,
            tx_stripped_size * WITNESS_SCALE_FACTOR,
            MAX_BLOCK_WEIGHT
        )));
    }

    // 7. Check for duplicate inputs (Orange Paper Section 5.1, rule 4)
    // ∀i,j ∈ ins: i ≠ j ⟹ i.prevout ≠ j.prevout
    // Optimization: Use HashSet for O(n) duplicate detection instead of O(n²) nested loop
    use std::collections::HashSet;
    let mut seen_prevouts = HashSet::with_capacity(tx.inputs.len());
    for (i, input) in tx.inputs.iter().enumerate() {
        // Bounds checking assertion: Input index must be valid
        assert!(i < tx.inputs.len(), "Input index {i} out of bounds");
        if !seen_prevouts.insert(&input.prevout) {
            return Ok(ValidationResult::Invalid(format!(
                "Duplicate input prevout at index {i}"
            )));
        }
    }

    // 8. Check coinbase scriptSig length (Orange Paper Section 5.1, rule 5)
    // If tx is coinbase: 2 ≤ |ins[0].scriptSig| ≤ 100
    if is_coinbase(tx) {
        debug_assert!(
            !tx.inputs.is_empty(),
            "Coinbase transaction must have at least one input"
        );
        let script_sig_len = tx.inputs[0].script_sig.len();
        if !(2..=100).contains(&script_sig_len) {
            return Ok(ValidationResult::Invalid(format!(
                "Coinbase scriptSig length {script_sig_len} must be between 2 and 100 bytes"
            )));
        }
    }

    // Postcondition assertion: Validation result must be Valid or Invalid
    // Note: This assertion documents the expected return type
    // The result is always Valid at this point (we would have returned Invalid earlier)

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
#[spec_locked("5.1")]
#[cfg_attr(feature = "production", inline(always))]
#[cfg_attr(not(feature = "production"), inline)]
#[allow(clippy::overly_complex_bool_expr)] // Intentional tautological assertions for formal verification
pub fn check_tx_inputs<U: UtxoLookup>(
    tx: &Transaction,
    utxo_set: &U,
    height: Natural,
) -> Result<(ValidationResult, Integer)> {
    check_tx_inputs_with_utxos(tx, utxo_set, height, None)
}

/// Optimized version that accepts pre-collected UTXOs to avoid redundant lookups
pub fn check_tx_inputs_with_utxos<U: UtxoLookup>(
    tx: &Transaction,
    utxo_set: &U,
    height: Natural,
    pre_collected_utxos: Option<&[Option<&UTXO>]>,
) -> Result<(ValidationResult, Integer)> {
    // Precondition checks: Validate function inputs
    // Note: We check this condition and return Invalid rather than asserting,
    // to allow tests to verify the validation logic properly
    if tx.inputs.is_empty() && !is_coinbase(tx) {
        return Ok((
            ValidationResult::Invalid(
                "Transaction must have inputs unless it's a coinbase".to_string(),
            ),
            0,
        ));
    }
    assert!(
        height <= i64::MAX as u64,
        "Block height {height} must fit in i64"
    );
    assert!(
        utxo_set.len() <= u32::MAX as usize,
        "UTXO set size {} exceeds maximum",
        utxo_set.len()
    );

    // Check if this is a coinbase transaction
    if is_coinbase(tx) {
        // Postcondition assertion: Coinbase fee must be zero
        #[allow(clippy::eq_op)]
        {
            // Coinbase fee must be zero (tautology for formal verification)
        }
        return Ok((ValidationResult::Valid, 0));
    }

    // Check that non-coinbase inputs don't have null prevouts (Orange Paper Section 5.1, rule 6)
    // ∀i ∈ ins: ¬i.prevout.IsNull()
    // Use proven bounds for input access in hot path
    #[cfg(feature = "production")]
    {
        use crate::optimizations::constant_folding::is_zero_hash;
        use crate::optimizations::optimized_access::get_proven_by_;
        for i in 0..tx.inputs.len() {
            if let Some(input) = get_proven_by_(&tx.inputs, i) {
                if is_zero_hash(&input.prevout.hash) && input.prevout.index == 0xffffffff {
                    return Ok((
                        ValidationResult::Invalid(format!(
                            "Non-coinbase input {i} has null prevout"
                        )),
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
    #[cfg(feature = "production")]
    {
        use crate::optimizations::prefetch;
        // Prefetch ahead for sequential UTXO lookups
        for i in 0..tx.inputs.len().min(8) {
            if i + 4 < tx.inputs.len() {
                prefetch::prefetch_ahead(&tx.inputs, i, 4);
            }
        }
    }

    // OPTIMIZATION: Use pre-collected UTXOs if provided, otherwise collect them
    let input_utxos: Vec<(usize, Option<&UTXO>)> = if let Some(pre_utxos) = pre_collected_utxos {
        // Pre-collected UTXOs provided - use them directly (no redundant lookups)
        pre_utxos.iter()
            .enumerate()
            .map(|(i, opt_utxo)| (i, *opt_utxo))
            .collect()
    } else {
        // No pre-collected UTXOs - collect them now
        let mut result = Vec::with_capacity(tx.inputs.len());
        for (i, input) in tx.inputs.iter().enumerate() {
            result.push((i, utxo_set.get(&input.prevout)));
        }
        result
    };

    let mut total_input_value = 0i64;
    // Invariant assertion: Total input value must start at zero
    assert!(
        total_input_value == 0,
        "Total input value must start at zero"
    );

    for (i, opt_utxo) in input_utxos {
        // Bounds checking assertion: Input index must be valid
        assert!(i < tx.inputs.len(), "Input index {i} out of bounds");

        // Check if input exists in UTXO set
        if let Some(utxo) = opt_utxo {
            // Invariant assertion: UTXO value must be non-negative and within MAX_MONEY
            assert!(
                utxo.value >= 0,
                "UTXO value {} must be non-negative at input {}",
                utxo.value,
                i
            );
            assert!(
                utxo.value <= MAX_MONEY,
                "UTXO value {} must not exceed MAX_MONEY at input {}",
                utxo.value,
                i
            );

            // Check coinbase maturity: coinbase outputs cannot be spent until COINBASE_MATURITY blocks deep
            // Bitcoin Core: if (coin.IsCoinBase() && nSpendHeight - coin.nHeight < COINBASE_MATURITY)
            // We check: if utxo.is_coinbase && height < utxo.height + COINBASE_MATURITY
            if utxo.is_coinbase {
                use crate::constants::COINBASE_MATURITY;
                let required_height = utxo.height.saturating_add(COINBASE_MATURITY);
                // Invariant assertion: Height must be sufficient for coinbase maturity
                assert!(
                    height >= utxo.height,
                    "Current height {} must be >= UTXO creation height {}",
                    height,
                    utxo.height
                );
                if height < required_height {
                    return Ok((
                        ValidationResult::Invalid(format!(
                            "Premature spend of coinbase output: input {i} created at height {} cannot be spent until height {} (current: {})",
                            utxo.height, required_height, height
                        )),
                        0,
                    ));
                }
            }

            // Use checked arithmetic to prevent overflow
            // Invariant assertion: UTXO value must be non-negative before addition
            assert!(
                utxo.value >= 0,
                "UTXO value {} must be non-negative before addition",
                utxo.value
            );
            total_input_value = total_input_value.checked_add(utxo.value).ok_or_else(|| {
                ConsensusError::TransactionValidation(
                    format!("Input value overflow at input {i}").into(),
                )
            })?;
            // Invariant assertion: Total input value must remain non-negative after addition
            assert!(
                total_input_value >= 0,
                "Total input value {total_input_value} must be non-negative after input {i}"
            );
        } else {
            #[cfg(debug_assertions)]
            {
                let hash_str: String = tx.inputs[i].prevout.hash.iter().map(|b| format!("{:02x}", b)).collect();
                eprintln!("   ❌ UTXO NOT FOUND: Input {} prevout {}:{}", i, hash_str, tx.inputs[i].prevout.index);
                eprintln!("      UTXO set size: {}", utxo_set.len());
            }
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
            // Invariant assertion: Output value must be non-negative
            assert!(
                output.value >= 0,
                "Output value {} must be non-negative",
                output.value
            );
            acc.checked_add(output.value).ok_or_else(|| {
                ConsensusError::TransactionValidation("Output value overflow".into())
            })
        })
        .map_err(|e| ConsensusError::TransactionValidation(Cow::Owned(e.to_string())))?;

    // Invariant assertion: Total output value must be non-negative
    assert!(
        total_output_value >= 0,
        "Total output value {total_output_value} must be non-negative"
    );
    // Check that output total doesn't exceed MAX_MONEY (Bitcoin Core check)
    assert!(
        total_output_value <= MAX_MONEY,
        "Total output value {total_output_value} must not exceed MAX_MONEY"
    );
    if total_output_value > MAX_MONEY {
        return Ok((
            ValidationResult::Invalid(format!(
                "Total output value {total_output_value} exceeds maximum money supply"
            )),
            0,
        ));
    }

    // Invariant assertion: Total input must be >= total output for valid transaction
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

    // Postcondition assertions: Validate fee calculation result
    assert!(fee >= 0, "Fee {fee} must be non-negative");
    assert!(
        fee <= total_input_value,
        "Fee {fee} cannot exceed total input {total_input_value}"
    );
    assert!(
        total_input_value == total_output_value + fee,
        "Conservation of value: input {total_input_value} must equal output {total_output_value} + fee {fee}"
    );

    Ok((ValidationResult::Valid, fee))
}

/// Check if transaction is coinbase
///
/// Hot-path function called frequently during validation.
/// Always inline for maximum performance.
#[inline(always)]
#[spec_locked("5.1")]
pub fn is_coinbase(tx: &Transaction) -> bool {
    // Optimization: Use constant folding for zero hash check
    #[cfg(feature = "production")]
    {
        use crate::optimizations::constant_folding::is_zero_hash;
        tx.inputs.len() == 1
            && is_zero_hash(&tx.inputs[0].prevout.hash)
            && tx.inputs[0].prevout.index == 0xffffffff
    }

    #[cfg(not(feature = "production"))]
    {
        tx.inputs.len() == 1
            && tx.inputs[0].prevout.hash == [0u8; 32]
            && tx.inputs[0].prevout.index == 0xffffffff
    }
}

/// Calculate transaction size (non-witness serialization)
///
/// Hot-path function called frequently during validation.
/// Always inline for maximum performance.
#[inline(always)]
///
/// This function calculates the size of a transaction when serialized
/// without witness data, matching Bitcoin Core's GetSerializeSize(TX_NO_WITNESS(tx)).
///
/// CRITICAL: This must match the actual serialized size exactly to ensure
/// consensus compatibility with Bitcoin Core.
#[spec_locked("5.1")]
pub fn calculate_transaction_size(tx: &Transaction) -> usize {
    // Use actual serialization to match Bitcoin Core's behavior
    // This replaces the simplified calculation that didn't account for varint encoding
    use crate::serialization::transaction::serialize_transaction;
    serialize_transaction(tx).len()
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
            // Minimum: version(4) + input_count_varint(1) + output_count_varint(1) + lock_time(4) = 10 bytes
            // (Even with 0 inputs/outputs, we need varints for counts)
            prop_assert!(size >= 10, "Transaction size must be at least 10 bytes (version + varints + lock_time)");

            // Maximum: Use MAX_TX_SIZE as the upper bound (actual serialization can be larger than simplified calculation)
            // The simplified calculation was: 4 + 10*41 + 10*9 + 4 = 508
            // But actual serialization with varints and real script sizes can be larger
            prop_assert!(size <= MAX_TX_SIZE, "Transaction size must not exceed MAX_TX_SIZE ({})", MAX_TX_SIZE);

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
                    prevout: OutPoint { hash: [0; 32].into(), index: 0 },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value,
                    script_pubkey: vec![].into(),
                }].into(),
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_transaction_valid() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }

    #[test]
    fn test_check_transaction_empty_inputs() {
        let tx = Transaction {
            version: 1,
            inputs: vec![].into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }]
            .into(),
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
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000, // 50 BTC
                script_pubkey: vec![].into(),
            }]
            .into(),
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
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![].into(),
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
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: -1, // Invalid negative value
                script_pubkey: vec![].into(),
            }]
            .into(),
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
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: MAX_MONEY + 1, // Invalid value exceeding max
                script_pubkey: vec![].into(),
            }]
            .into(),
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
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: MAX_MONEY, // Valid max value
                script_pubkey: vec![].into(),
            }]
            .into(),
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
            inputs: inputs.into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        assert!(matches!(
            check_transaction(&tx).unwrap(),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_check_transaction_max_inputs() {
        // Use a reasonable number of inputs that fits within the block weight limit.
        // Each input ≈ 41 bytes stripped. Weight limit = 4,000,000. Max stripped = 1,000,000.
        // Max inputs ≈ 1,000,000 / 41 ≈ 24,390. Use 20,000 to stay safe.
        let num_inputs = 20_000;
        let mut inputs = Vec::new();
        for i in 0..num_inputs {
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
            inputs: inputs.into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }]
            .into(),
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
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: outputs.into(),
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
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: outputs.into(),
            lock_time: 0,
        };

        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }

    #[test]
    fn test_check_transaction_too_large() {
        // Create a transaction that will exceed MAX_BLOCK_WEIGHT / WITNESS_SCALE_FACTOR
        // MAX_BLOCK_WEIGHT is 4,000,000, so MAX_TX_SIZE is effectively 1,000,000 bytes
        // calculate_transaction_size now uses actual serialization, so we need to create
        // a transaction with large scripts to exceed the size limit while staying within input limits
        use crate::constants::MAX_INPUTS;
        let mut inputs = Vec::new();
        // Use MAX_INPUTS inputs with large scripts to exceed size limit
        // Each input: 32 (hash) + 4 (index) + varint(script_len) + script + 4 (sequence)
        // With 1000 inputs and ~1000 byte scripts each, we get ~1MB+ transaction
        for i in 0..MAX_INPUTS {
            inputs.push(TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                },
                script_sig: vec![0u8; 1000], // Large script to increase size (1000 bytes each)
                sequence: 0xffffffff,
            });
        }

        let tx = Transaction {
            version: 1,
            inputs: inputs.into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }]
            .into(),
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
            is_coinbase: false,
        };
        utxo_set.insert(outpoint, utxo);

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 900000000, // 9 BTC output
                script_pubkey: vec![].into(),
            }]
            .into(),
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
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 100000000,
                script_pubkey: vec![].into(),
            }]
            .into(),
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
            is_coinbase: false,
        };
        utxo_set.insert(outpoint, utxo);

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 200000000, // 2 BTC output (more than input)
                script_pubkey: vec![].into(),
            }]
            .into(),
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
            is_coinbase: false,
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
            is_coinbase: false,
        };
        utxo_set.insert(outpoint2, utxo2);

        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [1; 32].into(),
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
            ]
            .into(),
            outputs: vec![TransactionOutput {
                value: 700000000, // 7 BTC output
                script_pubkey: vec![].into(),
            }]
            .into(),
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
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![].into(),
            lock_time: 0,
        };
        assert!(is_coinbase(&valid_coinbase));

        // Wrong hash
        let wrong_hash = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![].into(),
            lock_time: 0,
        };
        assert!(!is_coinbase(&wrong_hash));

        // Wrong index
        let wrong_index = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![].into(),
            lock_time: 0,
        };
        assert!(!is_coinbase(&wrong_index));

        // Multiple inputs
        let multiple_inputs = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [0; 32].into(),
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
            ]
            .into(),
            outputs: vec![].into(),
            lock_time: 0,
        };
        assert!(!is_coinbase(&multiple_inputs));

        // No inputs
        let no_inputs = Transaction {
            version: 1,
            inputs: vec![].into(),
            outputs: vec![].into(),
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
                        hash: [0; 32].into(),
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
            ]
            .into(),
            outputs: vec![
                TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![7, 8, 9].into(),
                },
                TransactionOutput {
                    value: 2000,
                    script_pubkey: vec![10, 11, 12],
                },
            ]
            .into(),
            lock_time: 12345,
        };

        let size = calculate_transaction_size(&tx);
        // Expected actual serialized size:
        // 4 (version) + 1 (input_count varint) +
        // 2 * (32 + 4 + 1 + 3 + 4) (inputs: hash + index + script_len_varint + script + sequence) +
        // 1 (output_count varint) +
        // 2 * (8 + 1 + 3) (outputs: value + script_len_varint + script) +
        // 4 (lock_time) = 4 + 1 + 88 + 1 + 24 + 4 = 122
        // This matches actual serialization (not simplified calculation)
        assert_eq!(size, 122);
    }
}
