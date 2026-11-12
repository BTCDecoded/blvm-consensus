//! Economic model functions from Orange Paper Section 7 Section 6

use crate::constants::*;
use crate::error::{ConsensusError, Result};
use crate::types::*;
#[cfg(feature = "spec-lock")]
use blvm_spec_lock::spec_locked;
use std::borrow::Cow;

/// GetBlockSubsidy: ℕ → ℤ
///
/// Calculate the block subsidy for a given height.
/// Subsidy halves every 210,000 blocks (HALVING_INTERVAL) by integer right-shift:
/// `INITIAL_SUBSIDY >> floor(h / H)`. The shift is 0 at halving 33
/// (`floor(5_000_000_000 / 2^33) = 0`, height ≥ 6,930,000). The `>= 64`
/// guard is for undefined 64-bit shifts. The real-valued form
/// `50 * C * 2^(-⌊h/H⌋)` is not the protocol.
#[cfg_attr(feature = "spec-lock", spec_locked("6.1", "GetBlockSubsidy"))]
#[cfg_attr(feature = "spec-lock", blvm_spec_lock::requires(height >= 0))]
#[cfg_attr(feature = "spec-lock", blvm_spec_lock::ensures(result >= 0))]
#[cfg_attr(feature = "spec-lock", blvm_spec_lock::ensures(result <= INITIAL_SUBSIDY))]
pub fn get_block_subsidy(height: Natural) -> Integer {
    let halving_period = height / HALVING_INTERVAL;

    // `>> 64` is undefined for 64-bit values. Practical zero is k = 33.
    if halving_period >= 64 {
        return 0;
    }

    // Runtime assertion: Halving period must be valid
    debug_assert!(
        halving_period < 64,
        "Halving period ({halving_period}) must be < 64"
    );

    let base_subsidy = INITIAL_SUBSIDY;
    let subsidy = base_subsidy >> halving_period;

    // Runtime assertion: Subsidy must be non-negative and <= initial subsidy
    debug_assert!(subsidy >= 0, "Subsidy ({subsidy}) must be non-negative");
    debug_assert!(
        subsidy <= INITIAL_SUBSIDY,
        "Subsidy ({subsidy}) must be <= initial subsidy ({INITIAL_SUBSIDY})"
    );

    subsidy
}

/// Regtest subsidy halves every 150 blocks. Other networks use [`HALVING_INTERVAL`].
pub const REGTEST_HALVING_INTERVAL: u64 = 150;

/// Halving interval for `network`. Regtest is 150; every other network is 210,000.
pub fn subsidy_halving_interval(network: Network) -> u64 {
    match network {
        Network::Regtest => REGTEST_HALVING_INTERVAL,
        Network::Mainnet | Network::Testnet | Network::Testnet4 | Network::Signet => {
            HALVING_INTERVAL
        }
    }
}

/// Block subsidy for `network`. Mainnet, testnet, testnet4, and signet use
/// [`get_block_subsidy`]. Regtest uses the same right-shift with interval 150.
pub fn get_block_subsidy_for_network(height: Natural, network: Network) -> Integer {
    let interval = subsidy_halving_interval(network);
    if interval == HALVING_INTERVAL {
        return get_block_subsidy(height);
    }
    block_subsidy_at_interval(height, interval, INITIAL_SUBSIDY)
}

/// Right-shift subsidy for an explicit interval and initial amount.
///
/// `interval == 0` yields 0. The 210,000-block schedule stays in
/// [`get_block_subsidy`].
pub fn block_subsidy_at_interval(height: Natural, interval: u64, initial: Integer) -> Integer {
    if interval == 0 {
        return 0;
    }
    let halving_period = height / interval;
    if halving_period >= 64 {
        return 0;
    }
    initial >> halving_period
}

/// CheckCoinbaseSubsidy: 𝒯𝒳 × ℤ × ℤ → {true, false}
///
/// Coinbase output sum must not exceed block subsidy plus aggregated fees.
#[cfg_attr(feature = "spec-lock", spec_locked("5.3", "CheckCoinbaseSubsidy"))]
pub fn check_coinbase_subsidy(
    coinbase: &Transaction,
    subsidy: Integer,
    total_fees: Integer,
) -> bool {
    // Negative fees/subsidy make the coinbase bound meaningless (connect already
    // rejects total_fees < 0 before calling; this is defense-in-depth for other callers).
    if total_fees < 0 || subsidy < 0 {
        return false;
    }
    let coinbase_output: i64 = match coinbase.outputs.iter().try_fold(0i64, |acc, output| {
        if output.value < 0 || output.value > MAX_MONEY {
            return None;
        }
        acc.checked_add(output.value)
    }) {
        Some(v) => v,
        None => return false,
    };
    if coinbase_output < 0 {
        return false;
    }
    let Some(max_coinbase) = total_fees.checked_add(subsidy) else {
        return false;
    };
    coinbase_output <= MAX_MONEY && coinbase_output <= max_coinbase
}

/// VerifyUtxoSupply: 𝒰𝒮 × ℕ → {true, false}
///
/// Theorem 8.1: sum of UTXO values equals total money supply at height h.
#[cfg_attr(feature = "spec-lock", spec_locked("8.1", "VerifyUtxoSupply"))]
pub fn verify_utxo_supply(utxo_set: &UtxoSet, height: Natural) -> bool {
    let expected = total_supply(height);
    if !(0..=MAX_MONEY).contains(&expected) {
        return false;
    }
    let mut actual: i64 = 0;
    for utxo in utxo_set.values() {
        if !(0..=MAX_MONEY).contains(&utxo.value) {
            return false;
        }
        actual = match actual.checked_add(utxo.value) {
            Some(v) => v,
            None => return false,
        };
    }
    actual == expected
}

/// TotalSupply: ℕ → ℤ
///
/// Calculate the total Bitcoin supply at a given height.
/// This is the sum of all block subsidies up to that height.
///
/// Subsidy is constant on each halving epoch \([k×H, (k+1)×H)\) for `k < 64`, so we sum
/// `count × (INITIAL_SUBSIDY >> k)` per epoch in **O(64)** instead of iterating every block.
/// (Naive `0..=height` is unusable for heights in the tens of millions used in tests.)
/// Non-negativity invariant: the total supply is always ≥ 0.
/// Each block subsidy is non-negative (integer-zero after 33 halvings; the loop
/// still walks 64 epochs to match the F_* undefined-shift guard), so the
/// accumulated sum is non-negative for any height.
/// Issued supply at and after height `33 * H` is exactly 2_099_999_997_690_000
/// satoshis, 2_310_000 below `MAX_MONEY`.
///
/// Non-negativity is proven by `_verify_f_total_supply_non_neg` (spec_witnesses.rs §6.2).
/// The Z3 translator cannot fully evaluate the 64-epoch for loop, so both bounds are
/// declared as `#[axiom]` (trusted from the spec_witness proof) in addition to the
/// `#[ensures]` postconditions that callee-axiom propagation can discharge for callers.
#[cfg_attr(feature = "spec-lock", spec_locked("6.2", "TotalSupply"))]
#[cfg_attr(feature = "spec-lock", blvm_spec_lock::axiom(result >= 0))]
#[cfg_attr(feature = "spec-lock", blvm_spec_lock::axiom(result <= 2100000000000000))]
#[cfg_attr(feature = "spec-lock", blvm_spec_lock::ensures(result >= 0))]
#[cfg_attr(feature = "spec-lock", blvm_spec_lock::ensures(result <= 2100000000000000))]
pub fn total_supply(height: Natural) -> Integer {
    let end = height;
    let h = HALVING_INTERVAL;
    let mut total = 0i64;

    for k in 0u64..64 {
        let period_start = k.saturating_mul(h);
        if period_start > end {
            break;
        }
        let subsidy = INITIAL_SUBSIDY >> k;
        let period_end = (k + 1).saturating_mul(h).saturating_sub(1);
        let overlap_hi = end.min(period_end);
        let count = overlap_hi
            .checked_sub(period_start)
            .map(|d| d + 1)
            .expect("overlap_hi >= period_start when period_start <= end");

        let contrib = match (count as i64).checked_mul(subsidy) {
            Some(c) => c,
            None => {
                debug_assert!(false, "Total supply contribution overflow at epoch {k}");
                return MAX_MONEY;
            }
        };
        total = match total.checked_add(contrib) {
            Some(t) => t,
            None => {
                debug_assert!(false, "Total supply sum overflow at epoch {k}");
                return MAX_MONEY;
            }
        };
        // Unreachable on the integer subsidy schedule: issued supply tops out
        // at 2_099_999_997_690_000 < MAX_MONEY (2_100_000_000_000_000).
        // Kept as a defensive overflow cap, not a reachable issuance path.
        if total >= MAX_MONEY {
            return MAX_MONEY;
        }
    }

    // Runtime assertion: Total supply must be non-negative and <= MAX_MONEY
    debug_assert!(total >= 0, "Total supply ({total}) must be non-negative");
    debug_assert!(
        total <= MAX_MONEY,
        "Total supply ({total}) must be <= MAX_MONEY ({MAX_MONEY})"
    );

    total
}

/// Calculate transaction fee
///
/// Fee = sum of input values - sum of output values
#[cfg_attr(feature = "spec-lock", spec_locked("6.5", "CalculateFee"))]
#[cfg_attr(feature = "spec-lock", blvm_spec_lock::ensures(result >= 0))]
pub fn calculate_fee(tx: &Transaction, utxo_set: &UtxoSet) -> Result<Integer> {
    if is_coinbase(tx) {
        return Ok(0);
    }

    // Use checked arithmetic to prevent overflow. Missing prevouts are outside
    // CalculateFee's domain (Orange Paper assumes us(i.prevout).value) — fail closed.
    let total_input: i64 = tx.inputs.iter().try_fold(0i64, |acc, input| {
        let utxo = utxo_set.get(&input.prevout).ok_or_else(|| {
            ConsensusError::UtxoNotFound(
                format!("UTXO not found for prevout index {}", input.prevout.index).into(),
            )
        })?;
        acc.checked_add(utxo.value)
            .ok_or_else(|| ConsensusError::EconomicValidation("Input value overflow".into()))
    })?;

    let total_output: i64 = tx
        .outputs
        .iter()
        .try_fold(0i64, |acc, output| {
            acc.checked_add(output.value)
                .ok_or_else(|| ConsensusError::EconomicValidation("Output value overflow".into()))
        })
        .map_err(|e| ConsensusError::EconomicValidation(Cow::Owned(e.to_string())))?;

    if total_output < 0 {
        return Err(ConsensusError::EconomicValidation(
            "Negative total output value".into(),
        ));
    }

    // Note: We use inline error here because it's EconomicValidation, not TransactionValidation
    // The helper function returns TransactionValidation, so we keep inline for type consistency
    let fee = total_input
        .checked_sub(total_output)
        .ok_or_else(|| ConsensusError::EconomicValidation("Fee calculation underflow".into()))?;

    // Check for negative fee and return error (tests intentionally test this error path)
    if fee < 0 {
        return Err(ConsensusError::EconomicValidation("Negative fee".into()));
    }

    // Runtime assertion: Fee must be non-negative (only after we've handled negative case)
    debug_assert!(
        fee >= 0,
        "Fee ({fee}) must be non-negative (input: {total_input}, output: {total_output})"
    );

    // Runtime assertion: Fee cannot exceed total input
    debug_assert!(
        fee <= total_input,
        "Fee ({fee}) cannot exceed total input ({total_input})"
    );

    Ok(fee)
}

/// Issued supply at `height` is at most `MAX_MONEY`.
///
/// The cap is proved by `_verify_f_issued_supply_below_cap`. This function is the
/// runtime comparison. spec-lock 0.1.45, which consensus CI runs, names
/// `total_supply` apart from the spec function `TotalSupply`, so a
/// `#[spec_locked]` row on this call is a partial rather than a proof.
pub fn validate_supply_limit(height: Natural) -> bool {
    total_supply(height) <= MAX_MONEY
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
            value in 0i64..=20_000,
            spend in 0i64..=20_000
        ) {
            let coinbase = Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0xffff_ffff },
                    script_sig: vec![0x51, 0x51],
                    sequence: 0xffff_ffff,
                }]
                .into(),
                outputs: vec![TransactionOutput {
                    value,
                    script_pubkey: vec![0x51],
                }]
                .into(),
                lock_time: 0,
            };
            prop_assert_eq!(calculate_fee(&coinbase, &UtxoSet::default()).unwrap(), 0);

            let prevout = OutPoint { hash: [1; 32], index: 0 };
            let mut utxo_set = UtxoSet::default();
            utxo_set.insert(
                prevout,
                std::sync::Arc::new(UTXO {
                    value,
                    script_pubkey: vec![].into(),
                    height: 0,
                    is_coinbase: false,
                }),
            );
            let spend_tx = Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout,
                    script_sig: vec![0x51],
                    sequence: 0xffff_ffff,
                }]
                .into(),
                outputs: vec![TransactionOutput {
                    value: spend,
                    script_pubkey: vec![0x51],
                }]
                .into(),
                lock_time: 0,
            };
            let fee = calculate_fee(&spend_tx, &utxo_set);
            if spend <= value {
                prop_assert_eq!(fee.unwrap(), value - spend);
            } else {
                prop_assert!(fee.is_err());
            }
        }
    }
}

#[cfg(test)]
#[path = "economic_tests.rs"]
mod tests;
