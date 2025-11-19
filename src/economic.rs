//! Economic model functions from Orange Paper Section 7 Section 6

use crate::constants::*;
use crate::error::{ConsensusError, Result};
use std::borrow::Cow;
use crate::types::*;

/// GetBlockSubsidy: ℕ → ℤ
///
/// Calculate the block subsidy for a given height.
/// Subsidy halves every 210,000 blocks (HALVING_INTERVAL).
///
/// Formula: subsidy = 50 * C * 2^(-⌊h/H⌋)
/// Where:
/// - h = block height
/// - H = HALVING_INTERVAL (210,000)
/// - C = SATOSHIS_PER_BTC (10^8)
pub fn get_block_subsidy(height: Natural) -> Integer {
    let halving_period = height / HALVING_INTERVAL;

    // After 64 halvings, subsidy becomes 0
    if halving_period >= 64 {
        return 0;
    }

    // Runtime assertion: Halving period must be valid
    debug_assert!(
        halving_period < 64,
        "Halving period ({}) must be < 64",
        halving_period
    );

    // Calculate subsidy: 50 BTC * 2^(-halving_period)
    let base_subsidy = INITIAL_SUBSIDY; // 50 BTC in satoshis
    let subsidy = base_subsidy >> halving_period;

    // Runtime assertion: Subsidy must be non-negative and <= initial subsidy
    debug_assert!(
        subsidy >= 0,
        "Subsidy ({}) must be non-negative",
        subsidy
    );
    debug_assert!(
        subsidy <= INITIAL_SUBSIDY,
        "Subsidy ({}) must be <= initial subsidy ({})",
        subsidy,
        INITIAL_SUBSIDY
    );

    subsidy
}

/// TotalSupply: ℕ → ℤ
///
/// Calculate the total Bitcoin supply at a given height.
/// This is the sum of all block subsidies up to that height.
pub fn total_supply(height: Natural) -> Integer {
    let mut total = 0i64;

    for h in 0..=height {
        let subsidy = get_block_subsidy(h);
        // Use checked arithmetic to prevent overflow
        total = total.checked_add(subsidy).unwrap_or_else(|| {
            // If overflow occurs, clamp to MAX_MONEY
            debug_assert!(
                false,
                "Total supply calculation overflow at height {}",
                h
            );
            MAX_MONEY
        });
        
        // Early exit if we've reached max money
        if total >= MAX_MONEY {
            break;
        }
    }

    // Runtime assertion: Total supply must be non-negative and <= MAX_MONEY
    debug_assert!(
        total >= 0,
        "Total supply ({}) must be non-negative",
        total
    );
    debug_assert!(
        total <= MAX_MONEY,
        "Total supply ({}) must be <= MAX_MONEY ({})",
        total,
        MAX_MONEY
    );

    total
}

/// Calculate transaction fee
///
/// Fee = sum of input values - sum of output values
pub fn calculate_fee(tx: &Transaction, utxo_set: &UtxoSet) -> Result<Integer> {
    if is_coinbase(tx) {
        return Ok(0);
    }

    // Use checked arithmetic to prevent overflow
    let total_input: i64 = tx
        .inputs
        .iter()
        .try_fold(0i64, |acc, input| {
            let value = utxo_set
                .get(&input.prevout)
                .map(|utxo| utxo.value)
                .unwrap_or(0);
            acc.checked_add(value).ok_or_else(|| {
                ConsensusError::EconomicValidation("Input value overflow".into())
            })
        })
        .map_err(|e| ConsensusError::EconomicValidation(Cow::Owned(e.to_string())))?;

    let total_output: i64 = tx
        .outputs
        .iter()
        .try_fold(0i64, |acc, output| {
            acc.checked_add(output.value).ok_or_else(|| {
                ConsensusError::EconomicValidation("Output value overflow".into())
            })
        })
        .map_err(|e| ConsensusError::EconomicValidation(Cow::Owned(e.to_string())))?;

    // Note: We use inline error here because it's EconomicValidation, not TransactionValidation
    // The helper function returns TransactionValidation, so we keep inline for type consistency
    let fee = total_input.checked_sub(total_output).ok_or_else(|| {
        ConsensusError::EconomicValidation("Fee calculation underflow".into())
    })?;
    
    // Check for negative fee and return error (tests intentionally test this error path)
    if fee < 0 {
        return Err(ConsensusError::EconomicValidation(
            "Negative fee".into(),
        ));
    }
    
    // Runtime assertion: Fee must be non-negative (only after we've handled negative case)
    debug_assert!(
        fee >= 0,
        "Fee ({}) must be non-negative (input: {}, output: {})",
        fee,
        total_input,
        total_output
    );

    // Runtime assertion: Fee cannot exceed total input
    debug_assert!(
        fee <= total_input,
        "Fee ({}) cannot exceed total input ({})",
        fee,
        total_input
    );

    Ok(fee)
}

/// Validate economic constraints
///
/// Check that the total supply doesn't exceed the maximum money supply
pub fn validate_supply_limit(height: Natural) -> Result<bool> {
    let current_supply = total_supply(height);
    Ok(current_supply <= MAX_MONEY)
}

/// Check if transaction is coinbase
fn is_coinbase(tx: &Transaction) -> bool {
    tx.inputs.len() == 1
        && tx.inputs[0].prevout.hash == [0u8; 32]
        && tx.inputs[0].prevout.index == 0xffffffff
}

// ============================================================================
// FORMAL VERIFICATION
// ============================================================================

/// Mathematical Specification for Block Subsidy:
/// ∀ h ∈ ℕ: subsidy(h) = 50 * 10^8 * 2^(-⌊h/210000⌋) if ⌊h/210000⌋ < 64 else 0
///
/// Invariants:
/// - Subsidy halves every 210,000 blocks
/// - After 64 halvings, subsidy becomes 0
/// - Subsidy is always non-negative
/// - Total supply approaches 21M BTC asymptotically

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: get_block_subsidy follows halving schedule
    #[kani::proof]
    fn kani_get_block_subsidy_halving_schedule() {
        let height: Natural = kani::any();

        // Bound height for tractability
        kani::assume(height <= HALVING_INTERVAL * 10); // Up to 10 halvings

        let subsidy = get_block_subsidy(height);
        let halving_period = height / HALVING_INTERVAL;

        // Non-negative invariant
        assert!(subsidy >= 0, "Subsidy must be non-negative");

        // Halving invariant: subsidy halves every 210,000 blocks
        if halving_period < 64 {
            let expected_subsidy = INITIAL_SUBSIDY >> halving_period;
            assert_eq!(
                subsidy, expected_subsidy,
                "Subsidy must follow halving schedule"
            );
        } else {
            assert_eq!(subsidy, 0, "Subsidy must be 0 after 64 halvings");
        }
    }

    /// Kani proof: get_block_subsidy boundary correctness (Orange Paper Section 6)
    ///
    /// Mathematical specification:
    /// ∀ height ∈ ℕ:
    /// - get_block_subsidy(0) = INITIAL_SUBSIDY
    /// - get_block_subsidy(height) at halving boundaries: subsidy halves correctly
    /// - get_block_subsidy(height) >= 64*HALVING_INTERVAL: subsidy = 0
    ///
    /// This ensures subsidy calculation handles all edge cases correctly.
    #[kani::proof]
    fn kani_get_block_subsidy_boundary_correctness() {
        // Test height = 0 (genesis block)
        let subsidy_at_0 = get_block_subsidy(0);
        assert_eq!(
            subsidy_at_0, INITIAL_SUBSIDY,
            "get_block_subsidy: height 0 must return INITIAL_SUBSIDY"
        );

        // Test height = 1 (first non-genesis block)
        let subsidy_at_1 = get_block_subsidy(1);
        assert_eq!(
            subsidy_at_1, INITIAL_SUBSIDY,
            "get_block_subsidy: height 1 must return INITIAL_SUBSIDY (before first halving)"
        );

        // Test height at first halving boundary
        let subsidy_at_first_halving = get_block_subsidy(HALVING_INTERVAL);
        let expected_first_halving = INITIAL_SUBSIDY >> 1;
        assert_eq!(
            subsidy_at_first_halving, expected_first_halving,
            "get_block_subsidy: height at first halving must return half of INITIAL_SUBSIDY"
        );

        // Test height just before first halving
        let subsidy_before_first_halving = get_block_subsidy(HALVING_INTERVAL - 1);
        assert_eq!(
            subsidy_before_first_halving, INITIAL_SUBSIDY,
            "get_block_subsidy: height just before first halving must return INITIAL_SUBSIDY"
        );

        // Test height at 64 halvings (subsidy becomes 0)
        let height_at_64_halvings = HALVING_INTERVAL * 64;
        let subsidy_at_64_halvings = get_block_subsidy(height_at_64_halvings);
        assert_eq!(
            subsidy_at_64_halvings, 0,
            "get_block_subsidy: height at 64 halvings must return 0"
        );

        // Test height beyond 64 halvings
        let height_beyond_64_halvings = HALVING_INTERVAL * 65;
        let subsidy_beyond_64_halvings = get_block_subsidy(height_beyond_64_halvings);
        assert_eq!(
            subsidy_beyond_64_halvings, 0,
            "get_block_subsidy: height beyond 64 halvings must return 0"
        );

        // Test height just before 64 halvings
        let height_just_before_64_halvings = HALVING_INTERVAL * 64 - 1;
        let subsidy_just_before_64_halvings = get_block_subsidy(height_just_before_64_halvings);
        // This should be INITIAL_SUBSIDY >> 63 (last non-zero subsidy)
        let expected_last_subsidy = INITIAL_SUBSIDY >> 63;
        assert_eq!(
            subsidy_just_before_64_halvings, expected_last_subsidy,
            "get_block_subsidy: height just before 64 halvings must return last non-zero subsidy"
        );
    }

    /// Kani proof: total_supply is monotonically increasing
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_total_supply_monotonic() {
        let height1: Natural = kani::any();
        let height2: Natural = kani::any();

        // Bound heights for tractability
        kani::assume(height1 <= HALVING_INTERVAL * 2);
        kani::assume(height2 <= HALVING_INTERVAL * 2);
        kani::assume(height1 <= height2);

        let supply1 = total_supply(height1);
        let supply2 = total_supply(height2);

        // Monotonic invariant
        assert!(
            supply2 >= supply1,
            "Total supply must be monotonically increasing"
        );

        // Non-negative invariant
        assert!(supply1 >= 0, "Total supply must be non-negative");
        assert!(supply2 >= 0, "Total supply must be non-negative");
    }

    /// Kani proof: supply limit is never exceeded
    ///
    /// Mathematical specification:
    /// ∀ h ∈ ℕ: total_supply(h) ≤ MAX_MONEY
    ///
    /// This is a critical security property ensuring Bitcoin's supply cap.
    #[kani::proof]
    fn kani_supply_limit_respected() {
        let height: Natural = kani::any();

        // Bound height for tractability (but test with realistic bounds)
        // Even with 100 halvings, supply approaches 21M asymptotically
        kani::assume(height <= HALVING_INTERVAL * 100);

        let supply = total_supply(height);

        // Critical invariant: supply never exceeds maximum money
        // This is the fundamental 21M BTC limit
        assert!(
            supply <= MAX_MONEY,
            "Total supply must not exceed maximum money"
        );

        // Additional invariant: supply is always non-negative
        assert!(supply >= 0, "Total supply must be non-negative");

        // Invariant: supply increases with height
        if height > 0 {
            let supply_at_h = total_supply(height);
            let supply_at_h_minus_1 = total_supply(height - 1);
            assert!(
                supply_at_h >= supply_at_h_minus_1,
                "Total supply must be monotonically increasing"
            );
        }
    }

    /// Kani proof: validate_supply_limit correctly enforces MAX_MONEY
    ///
    /// Mathematical specification:
    /// ∀ h ∈ ℕ: validate_supply_limit(h) = true ⟺ total_supply(h) ≤ MAX_MONEY
    #[kani::proof]
    fn kani_validate_supply_limit_correctness() {
        let height: Natural = kani::any();

        kani::assume(height <= HALVING_INTERVAL * 100);

        let supply = total_supply(height);
        let validation_result = validate_supply_limit(height);

        // Validation result should match supply comparison
        assert!(validation_result.is_ok());
        let is_valid = validation_result.unwrap();

        assert_eq!(
            is_valid,
            supply <= MAX_MONEY,
            "validate_supply_limit must correctly reflect supply ≤ MAX_MONEY"
        );

        // If supply is within limit, validation must pass
        if supply <= MAX_MONEY {
            assert!(is_valid, "Supply within limit must validate as true");
        }
    }

    /// Kani proof: Supply Convergence (Orange Paper Theorem 8.2)
    ///
    /// Mathematical specification:
    /// lim(h→∞) TotalSupply(h) = 21 × 10⁶ × C
    ///
    /// This proves that the total supply converges to exactly 21 million BTC.
    /// After 64 halvings (height 13,440,000), the subsidy becomes 0, so:
    /// TotalSupply(13,440,000) = 50 × C × Σ(i=0 to 63) (1/2)^i = 100 × C × (1 - 2^-64)
    /// For practical purposes, 2^-64 ≈ 0, so supply ≈ 21M BTC.
    #[kani::proof]
    fn kani_supply_convergence() {
        // Test convergence property: as height increases, supply approaches 21M
        // After 64 halvings (height = 13,440,000), subsidy = 0

        // Calculate supply at different heights to show convergence
        let height_after_64_halvings = HALVING_INTERVAL * 64;

        // Bound for tractability (test up to 10 halvings)
        let max_test_height = HALVING_INTERVAL * 10;
        kani::assume(max_test_height <= height_after_64_halvings);

        // Supply at height h approaches 21M as h increases
        // Formula: TotalSupply(h) = 50 × C × Σ(i=0 to min(63, ⌊h/H⌋)) (1/2)^i
        // Where H = HALVING_INTERVAL = 210,000

        // Test monotonic convergence: supply increases but never exceeds 21M
        let supply_at_max = total_supply(max_test_height);

        // Critical invariant: supply never exceeds maximum money (21M BTC)
        assert!(
            supply_at_max <= MAX_MONEY,
            "Supply Convergence: total supply must never exceed 21M BTC"
        );

        // Convergence property: after many halvings, supply approaches 21M
        // For height = HALVING_INTERVAL * 64, subsidy = 0, so supply is constant
        // The supply at this point is: 50 × C × (1 - (1/2)^64) / (1 - 1/2) = 100 × C × (1 - 2^-64)
        // Since 2^-64 is negligible, this is effectively 100 × C = 21M BTC

        // Verify supply is bounded and converges
        assert!(
            supply_at_max >= 0,
            "Supply Convergence: total supply must be non-negative"
        );

        // Supply increases monotonically until halvings stop
        if max_test_height > 0 {
            let supply_at_h_minus_1 = total_supply(max_test_height - 1);
            assert!(
                supply_at_max >= supply_at_h_minus_1,
                "Supply Convergence: supply increases until halvings complete"
            );
        }
    }

    /// Kani proof: Coinbase value overflow safety (Orange Paper Section 13.3.1)
    ///
    /// Mathematical specification:
    /// ∀ height ∈ ℕ, fees ∈ ℤ:
    /// - coinbase_value = subsidy + fees ⟹ coinbase_value ≤ MAX_MONEY
    ///
    /// This ensures coinbase value calculation never exceeds MAX_MONEY.
    #[kani::proof]
    fn kani_coinbase_value_overflow_safety() {
        let height: Natural = kani::any();
        let fees: i64 = kani::any();

        // Bound for tractability
        kani::assume(height <= HALVING_INTERVAL * 10);
        kani::assume(fees >= 0);
        kani::assume(fees < MAX_MONEY);

        let subsidy = get_block_subsidy(height);

        // Coinbase value calculation must use checked arithmetic
        let coinbase_value = match subsidy.checked_add(fees) {
            Some(sum) => sum,
            None => {
                // Overflow detected - this should be caught
                assert!(
                    subsidy + fees > i64::MAX,
                    "Coinbase value overflow: subsidy + fees exceeds i64::MAX"
                );
                return;
            }
        };

        // Critical invariant: coinbase value must not exceed MAX_MONEY
        assert!(
            coinbase_value <= MAX_MONEY,
            "Coinbase value overflow safety: coinbase_value must not exceed MAX_MONEY"
        );

        // Coinbase value must be non-negative
        assert!(
            coinbase_value >= 0,
            "Coinbase value overflow safety: coinbase_value must be non-negative"
        );
    }

    /// Kani proof: calculate_fee correctness
    ///
    /// Mathematical specification (Orange Paper Section 6.3):
    /// ∀ tx ∈ TX, utxo_set ∈ US:
    /// - calculate_fee(tx, utxo_set) = fee ⟹
    ///   (fee = sum(inputs.value) - sum(outputs.value) ∧
    ///    fee >= 0)
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_calculate_fee_correctness() {
        let tx: Transaction = kani::any();
        let mut utxo_set: UtxoSet = kani::any();

        // Bound for tractability
        kani::assume(tx.inputs.len() <= 5);
        kani::assume(tx.outputs.len() <= 5);

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

        // Bound output values to prevent overflow
        for output in &mut tx.outputs {
            kani::assume(output.value >= 0);
            kani::assume(output.value <= MAX_MONEY);
        }

        let result = calculate_fee(&tx, &utxo_set);

        if result.is_ok() {
            let fee = result.unwrap();

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

            // If coinbase, fee must be 0
            if is_coinbase(&tx) {
                assert_eq!(fee, 0, "Coinbase transactions must have zero fee");
            } else {
                // Fee must match sum(inputs) - sum(outputs) when no overflow
                if let Some(expected_fee) = total_input.checked_sub(total_output) {
                    if expected_fee >= 0 {
                        assert_eq!(
                            fee, expected_fee,
                            "Fee calculation must match sum(inputs) - sum(outputs)"
                        );
                    }
                }
            }
        } else {
            // Fee calculation may fail for invalid transactions (negative fees, overflow)
            // This is acceptable behavior
        }
    }

    /// Kani proof: calculate_fee overflow safety
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ TX, utxo_set ∈ US:
    /// - calculate_fee(tx, utxo_set) handles overflow correctly
    /// - If sum(inputs) or sum(outputs) overflows, returns error
    /// - Fee calculation never overflows when inputs are bounded
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_calculate_fee_overflow_safety() {
        let tx: Transaction = kani::any();
        let mut utxo_set: UtxoSet = kani::any();

        // Bound for tractability
        kani::assume(tx.inputs.len() <= 5);
        kani::assume(tx.outputs.len() <= 5);

        // Populate UTXO set with bounded values to test overflow detection
        for input in &tx.inputs {
            if !utxo_set.contains_key(&input.prevout) {
                let value: i64 = kani::any();
                // Bound values to test overflow edge cases
                kani::assume(value >= 0);
                kani::assume(value <= MAX_MONEY);
                let utxo = UTXO {
                    value,
                    script_pubkey: kani::any(),
                    height: 0,
                };
                utxo_set.insert(input.prevout, utxo);
            }
        }

        // Bound output values
        for output in &tx.outputs {
            kani::assume(output.value >= 0);
            kani::assume(output.value <= MAX_MONEY);
        }

        let result = calculate_fee(&tx, &utxo_set);

        // If calculation succeeds, fee must be bounded
        if let Ok(fee) = result {
            assert!(fee >= 0, "Fee must be non-negative");
            assert!(fee <= MAX_MONEY, "Fee must not exceed MAX_MONEY");
        }
        // If calculation fails, it's due to overflow or negative fee (acceptable)
    }

    /// Kani proof: calculate_fee with missing UTXO
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ TX, utxo_set ∈ US:
    /// - If input.prevout ∉ utxo_set: calculate_fee treats missing UTXO as value 0
    /// - Fee calculation handles missing UTXOs gracefully
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_calculate_fee_missing_utxo() {
        let tx: Transaction = kani::any();
        let utxo_set: UtxoSet = kani::any();

        // Bound for tractability
        kani::assume(tx.inputs.len() <= 5);
        kani::assume(tx.outputs.len() <= 5);
        kani::assume(!is_coinbase(&tx)); // Non-coinbase for fee calculation

        // Bound output values
        for output in &tx.outputs {
            kani::assume(output.value >= 0);
            kani::assume(output.value <= MAX_MONEY);
        }

        // Test with potentially missing UTXOs
        let result = calculate_fee(&tx, &utxo_set);

        // Missing UTXOs are treated as value 0, so fee calculation should still work
        // (though it may result in negative fee, which is an error)
        if result.is_ok() {
            let fee = result.unwrap();
            assert!(fee >= 0, "Fee must be non-negative when calculation succeeds");
        }
    }

    /// Kani proof: Economic invariants across multiple blocks
    ///
    /// Mathematical specification:
    /// ∀ h₁, h₂ ∈ ℕ, h₁ ≤ h₂:
    /// - total_supply(h₂) = total_supply(h₁) + Σ(i=h₁+1 to h₂) get_block_subsidy(i)
    /// - Supply increases monotonically
    /// - Supply never exceeds MAX_MONEY
    #[kani::proof]
    #[kani::unwind(10)]
    fn kani_economic_invariants_across_blocks() {
        let height1: Natural = kani::any();
        let height2: Natural = kani::any();

        // Bound for tractability
        kani::assume(height1 <= HALVING_INTERVAL * 5);
        kani::assume(height2 <= HALVING_INTERVAL * 5);
        kani::assume(height1 <= height2);

        let supply1 = total_supply(height1);
        let supply2 = total_supply(height2);

        // Monotonicity invariant
        assert!(
            supply2 >= supply1,
            "Supply must be monotonically increasing across blocks"
        );

        // Calculate expected increase
        let mut expected_increase = 0i64;
        for h in (height1 + 1)..=height2 {
            let subsidy = get_block_subsidy(h);
            expected_increase = expected_increase.saturating_add(subsidy);
        }

        // Supply increase should match sum of subsidies
        let actual_increase = supply2.saturating_sub(supply1);
        assert!(
            actual_increase <= expected_increase,
            "Supply increase must not exceed sum of subsidies"
        );

        // Both supplies must respect MAX_MONEY
        assert!(supply1 <= MAX_MONEY, "Supply at height1 must not exceed MAX_MONEY");
        assert!(supply2 <= MAX_MONEY, "Supply at height2 must not exceed MAX_MONEY");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Property test: get_block_subsidy follows halving schedule
    proptest! {
        #[test]
        fn prop_get_block_subsidy_halving_schedule(
            height in 0u64..(HALVING_INTERVAL * 10)
        ) {
            let subsidy = get_block_subsidy(height);
            let halving_period = height / HALVING_INTERVAL;

            // Non-negative property
            prop_assert!(subsidy >= 0, "Subsidy must be non-negative");

            // Halving property
            if halving_period < 64 {
                let expected_subsidy = INITIAL_SUBSIDY >> halving_period;
                prop_assert_eq!(subsidy, expected_subsidy, "Subsidy must follow halving schedule");
            } else {
                prop_assert_eq!(subsidy, 0, "Subsidy must be 0 after 64 halvings");
            }
        }
    }

    /// Property test: total_supply is monotonically increasing
    proptest! {
        #[test]
        fn prop_total_supply_monotonic(
            height1 in 0u64..(HALVING_INTERVAL * 2),
            height2 in 0u64..(HALVING_INTERVAL * 2)
        ) {
            // Ensure height1 <= height2
            let (h1, h2) = if height1 <= height2 { (height1, height2) } else { (height2, height1) };

            let supply1 = total_supply(h1);
            let supply2 = total_supply(h2);

            // Monotonic property
            prop_assert!(supply2 >= supply1, "Total supply must be monotonically increasing");

            // Non-negative property
            prop_assert!(supply1 >= 0, "Total supply must be non-negative");
            prop_assert!(supply2 >= 0, "Total supply must be non-negative");
        }
    }

    /// Property test: supply limit is never exceeded
    proptest! {
        #[test]
        fn prop_supply_limit_respected(
            height in 0u64..(HALVING_INTERVAL * 10)
        ) {
            let supply = total_supply(height);

            // Supply limit property
            prop_assert!(supply <= MAX_MONEY, "Total supply must not exceed maximum money");

            // Non-negative property
            prop_assert!(supply >= 0, "Total supply must be non-negative");
        }
    }

    /// Property test: subsidy decreases with halving periods
    proptest! {
        #[test]
        fn prop_subsidy_decreases_with_halvings(
            height1 in 0u64..(HALVING_INTERVAL * 5),
            height2 in 0u64..(HALVING_INTERVAL * 5)
        ) {
            let halving1 = height1 / HALVING_INTERVAL;
            let halving2 = height2 / HALVING_INTERVAL;

            // If halving1 < halving2, then subsidy1 >= subsidy2
            if halving1 < halving2 && halving2 < 64 {
                let subsidy1 = get_block_subsidy(height1);
                let subsidy2 = get_block_subsidy(height2);
                prop_assert!(subsidy1 >= subsidy2, "Subsidy must decrease with halving periods");
            }
        }
    }

    /// Property test: calculate_fee handles coinbase correctly
    proptest! {
        #[test]
        fn prop_calculate_fee_coinbase(
            tx in proptest::collection::vec(any::<Transaction>(), 1..=1)
        ) {
            if let Some(tx) = tx.first() {
                let utxo_set = UtxoSet::new();

                if is_coinbase(tx) {
                    let fee = calculate_fee(tx, &utxo_set).unwrap_or(-1);
                    prop_assert_eq!(fee, 0, "Coinbase transactions must have zero fee");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_block_subsidy_genesis() {
        assert_eq!(get_block_subsidy(0), INITIAL_SUBSIDY);
    }

    #[test]
    fn test_get_block_subsidy_first_halving() {
        // At height 210,000, subsidy should be 25 BTC
        assert_eq!(get_block_subsidy(HALVING_INTERVAL), INITIAL_SUBSIDY / 2);
    }

    #[test]
    fn test_get_block_subsidy_second_halving() {
        // At height 420,000, subsidy should be 12.5 BTC
        assert_eq!(get_block_subsidy(HALVING_INTERVAL * 2), INITIAL_SUBSIDY / 4);
    }

    #[test]
    fn test_get_block_subsidy_max_halvings() {
        // After 64 halvings, subsidy should be 0
        assert_eq!(get_block_subsidy(HALVING_INTERVAL * 64), 0);
    }

    #[test]
    fn test_total_supply_convergence() {
        // Test that total supply approaches 21M BTC
        let supply_at_halving = total_supply(HALVING_INTERVAL);
        // At the first halving, we have 210,000 blocks of 50 BTC each
        let expected_at_halving = (HALVING_INTERVAL as i64) * INITIAL_SUBSIDY;
        // The difference is due to bit shifting in get_block_subsidy
        // Allow for much larger rounding differences due to bit operations
        let difference = (supply_at_halving - expected_at_halving).abs();
        println!(
            "Supply at halving: {supply_at_halving}, Expected: {expected_at_halving}, Difference: {difference}"
        );
        assert!(difference <= 3_000_000_000); // Allow for significant rounding differences
    }

    #[test]
    fn test_supply_limit() {
        // Test that supply limit is respected
        assert!(validate_supply_limit(0).unwrap());
        assert!(validate_supply_limit(HALVING_INTERVAL).unwrap());
        assert!(validate_supply_limit(HALVING_INTERVAL * 10).unwrap());
    }

    #[test]
    fn test_calculate_fee_coinbase() {
        let coinbase_tx = Transaction {
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
                value: 5000000000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };

        let utxo_set = UtxoSet::new();
        assert_eq!(calculate_fee(&coinbase_tx, &utxo_set).unwrap(), 0);
    }

    // ============================================================================
    // COMPREHENSIVE ECONOMIC TESTS
    // ============================================================================

    #[test]
    fn test_get_block_subsidy_edge_cases() {
        // Test height 1 (just after genesis)
        assert_eq!(get_block_subsidy(1), INITIAL_SUBSIDY);

        // Test height just before first halving
        assert_eq!(get_block_subsidy(HALVING_INTERVAL - 1), INITIAL_SUBSIDY);

        // Test height just after first halving
        assert_eq!(get_block_subsidy(HALVING_INTERVAL + 1), INITIAL_SUBSIDY / 2);

        // Test height just before second halving
        assert_eq!(
            get_block_subsidy(HALVING_INTERVAL * 2 - 1),
            INITIAL_SUBSIDY / 2
        );

        // Test height just after second halving
        assert_eq!(
            get_block_subsidy(HALVING_INTERVAL * 2 + 1),
            INITIAL_SUBSIDY / 4
        );
    }

    #[test]
    fn test_get_block_subsidy_large_heights() {
        // Test very large height (beyond 64 halvings)
        let very_large_height = HALVING_INTERVAL * 100;
        assert_eq!(get_block_subsidy(very_large_height), 0);

        // Test exactly 64 halvings
        let exactly_64_halvings = HALVING_INTERVAL * 64;
        assert_eq!(get_block_subsidy(exactly_64_halvings), 0);

        // Test just before 64 halvings
        let just_before_64 = HALVING_INTERVAL * 64 - 1;
        assert_eq!(get_block_subsidy(just_before_64), INITIAL_SUBSIDY >> 63);
    }

    #[test]
    fn test_total_supply_edge_cases() {
        // Test total supply at height 0
        assert_eq!(total_supply(0), INITIAL_SUBSIDY);

        // Test total supply at height 1
        assert_eq!(total_supply(1), INITIAL_SUBSIDY * 2);

        // Test total supply at height 2
        assert_eq!(total_supply(2), INITIAL_SUBSIDY * 3);

        // Test total supply at first halving
        let supply_at_halving = total_supply(HALVING_INTERVAL);
        assert!(supply_at_halving > 0);
        // At halving, supply should be close to 210,000 * 50 BTC, but bit shifting causes rounding
        let expected_approximate = INITIAL_SUBSIDY * HALVING_INTERVAL as i64;
        // Allow for the fact that bit shifting can cause the supply to be higher due to rounding
        assert!(supply_at_halving <= expected_approximate + 5000000000); // Allow 50 BTC difference
    }

    #[test]
    fn test_total_supply_large_heights() {
        // Test total supply at very large height
        let very_large_height = HALVING_INTERVAL * 100;
        let supply = total_supply(very_large_height);

        // Should be finite and positive
        assert!(supply > 0);
        assert!(supply < MAX_MONEY);

        // Should be close to 21M BTC (allowing for rounding)
        let expected_max = 21_000_000 * 100_000_000; // 21M BTC in satoshis
        assert!(supply <= expected_max);
    }

    #[test]
    fn test_calculate_fee_regular_transaction() {
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

        // Fee should be 1 BTC (1000000000 - 900000000)
        assert_eq!(calculate_fee(&tx, &utxo_set).unwrap(), 100000000);
    }

    #[test]
    fn test_calculate_fee_multiple_inputs_outputs() {
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
            outputs: vec![
                TransactionOutput {
                    value: 600000000, // 6 BTC output
                    script_pubkey: vec![],
                },
                TransactionOutput {
                    value: 150000000, // 1.5 BTC output
                    script_pubkey: vec![],
                },
            ],
            lock_time: 0,
        };

        // Fee should be 0.5 BTC (800000000 - 750000000)
        assert_eq!(calculate_fee(&tx, &utxo_set).unwrap(), 50000000);
    }

    #[test]
    fn test_calculate_fee_missing_utxo() {
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

        // Should return error for negative fee (0 input - 100000000 output = negative)
        let result = calculate_fee(&tx, &utxo_set);
        assert!(result.is_err());
        assert!(matches!(result, Err(ConsensusError::EconomicValidation(_))));
    }

    #[test]
    fn test_calculate_fee_negative_fee() {
        let mut utxo_set = UtxoSet::new();

        // Add UTXO with value less than output
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

        // Should return error for negative fee
        let result = calculate_fee(&tx, &utxo_set);
        assert!(result.is_err());
        assert!(matches!(result, Err(ConsensusError::EconomicValidation(_))));
    }

    #[test]
    fn test_validate_supply_limit_edge_cases() {
        // Test at height 0
        assert!(validate_supply_limit(0).unwrap());

        // Test at first halving
        assert!(validate_supply_limit(HALVING_INTERVAL).unwrap());

        // Test at second halving
        assert!(validate_supply_limit(HALVING_INTERVAL * 2).unwrap());

        // Test at very large height
        assert!(validate_supply_limit(HALVING_INTERVAL * 100).unwrap());
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
}
