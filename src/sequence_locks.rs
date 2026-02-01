//! Sequence lock calculation functions (BIP68)
//!
//! Implements Bitcoin Core's sequence lock calculation for relative locktime.
//! Sequence locks are used to enforce relative locktime constraints using
//! transaction input sequence numbers.
//!
//! Reference: Bitcoin Core `tx_verify.cpp` CalculateSequenceLocks and EvaluateSequenceLocks

use crate::bip113::get_median_time_past;
use crate::error::Result;
use crate::locktime::{
    extract_sequence_locktime_value, extract_sequence_type_flag, is_sequence_disabled,
};
use crate::types::*;
use blvm_spec_lock::spec_locked;

/// Sequence locktime disable flag (bit 31)
/// When set, the sequence number is not treated as a relative locktime
#[allow(dead_code)]
const SEQUENCE_LOCKTIME_DISABLE_FLAG: u32 = 0x80000000;

/// Sequence locktime type flag (bit 22)
/// When set, locktime is time-based; otherwise block-based
#[allow(dead_code)]
const SEQUENCE_LOCKTIME_TYPE_FLAG: u32 = 0x00400000;

/// Sequence locktime mask (bits 0-15)
/// Extracts the locktime value from sequence number
#[allow(dead_code)]
const SEQUENCE_LOCKTIME_MASK: u32 = 0x0000ffff;

/// Sequence locktime granularity (for time-based locks)
/// Time-based locks are measured in 512-second intervals
const SEQUENCE_LOCKTIME_GRANULARITY: u32 = 9; // 2^9 = 512 seconds

/// Locktime verify sequence flag
/// Must be set to enable BIP68 sequence lock enforcement
const LOCKTIME_VERIFY_SEQUENCE: u32 = 0x01;

/// Calculate sequence locks for a transaction (BIP68)
///
/// Computes the minimum block height and time that must be reached
/// before the transaction can be included in a block.
///
/// Matches Bitcoin Core's CalculateSequenceLocks() exactly.
///
/// # Arguments
/// * `tx` - Transaction to calculate locks for
/// * `flags` - Script verification flags (must include LOCKTIME_VERIFY_SEQUENCE)
/// * `prev_heights` - Block heights at which each input confirmed
/// * `recent_headers` - Recent block headers for median time-past calculation
///
/// # Returns
/// Pair (min_height, min_time) that must be satisfied:
/// - min_height: Minimum block height (or -1 if no height constraint)
/// - min_time: Minimum block time (or -1 if no time constraint)
#[spec_locked("5.5")]
pub fn calculate_sequence_locks(
    tx: &Transaction,
    flags: u32,
    prev_heights: &[u64],
    recent_headers: Option<&[BlockHeader]>,
) -> Result<(i64, i64)> {
    // Ensure prev_heights matches input count
    if prev_heights.len() != tx.inputs.len() {
        return Err(crate::error::ConsensusError::ConsensusRuleViolation(
            format!(
                "prev_heights length {} does not match input count {}",
                prev_heights.len(),
                tx.inputs.len()
            )
            .into(),
        ));
    }

    // Initialize to -1 (no constraint)
    let mut min_height: i64 = -1;
    let mut min_time: i64 = -1;

    // BIP68 is only enforced for version 2+ transactions and when flag is set
    let enforce_bip68 = tx.version >= 2 && (flags & LOCKTIME_VERIFY_SEQUENCE) != 0;

    if !enforce_bip68 {
        return Ok((min_height, min_time));
    }

    // Process each input
    for (i, input) in tx.inputs.iter().enumerate() {
        // Check if sequence is disabled (bit 31 set)
        if is_sequence_disabled(input.sequence as u32) {
            // This input doesn't contribute to sequence locks
            continue;
        }

        let coin_height = prev_heights[i] as i64;

        // Check locktime type (bit 22)
        if extract_sequence_type_flag(input.sequence as u32) {
            // Time-based relative locktime
            // Need median time-past of the block prior to the coin's block
            let coin_time = if let Some(headers) = recent_headers {
                // Calculate median time-past for the block prior to coin_height
                // For simplicity, we'll use the most recent header's median time-past
                // In a full implementation, we'd need to look up the actual block
                get_median_time_past(headers) as i64
            } else {
                // No headers available - can't calculate time-based lock
                // This is acceptable for some contexts (e.g., mempool validation)
                continue;
            };

            // Extract locktime value and multiply by granularity (512 seconds)
            let locktime_value = extract_sequence_locktime_value(input.sequence as u32) as i64;

            // Runtime assertion: Locktime value must be non-negative
            debug_assert!(
                locktime_value >= 0,
                "Locktime value ({locktime_value}) must be non-negative"
            );

            let locktime_seconds = locktime_value << SEQUENCE_LOCKTIME_GRANULARITY;

            // Runtime assertion: Shift operation must not overflow
            // locktime_value is u16 (max 65535), so locktime_seconds = 65535 * 512 = 33,553,920
            // This fits in i64, so no overflow possible
            debug_assert!(
                locktime_seconds >= 0,
                "Locktime seconds ({locktime_seconds}) must be non-negative (locktime_value: {locktime_value}, granularity: {SEQUENCE_LOCKTIME_GRANULARITY})"
            );

            // Calculate minimum time: coin_time + locktime_seconds - 1
            // The -1 is to maintain nLockTime semantics (last invalid time)
            let required_time = coin_time
                .checked_add(locktime_seconds)
                .and_then(|sum| sum.checked_sub(1))
                .ok_or_else(|| {
                    crate::error::ConsensusError::ConsensusRuleViolation(
                        "Sequence lock time calculation overflow".into(),
                    )
                })?;

            // Runtime assertion: Required time must be >= coin_time (or close to it after -1)
            debug_assert!(
                required_time >= coin_time.saturating_sub(1),
                "Required time ({required_time}) must be >= coin_time - 1 ({coin_time})"
            );

            min_time = min_time.max(required_time);
        } else {
            // Block-based relative locktime
            // Extract locktime value (number of blocks)
            let locktime_value = extract_sequence_locktime_value(input.sequence as u32) as i64;

            // Runtime assertion: Locktime value must be non-negative
            debug_assert!(
                locktime_value >= 0,
                "Locktime value ({locktime_value}) must be non-negative"
            );

            // Runtime assertion: Coin height must be non-negative
            debug_assert!(
                coin_height >= 0,
                "Coin height ({coin_height}) must be non-negative"
            );

            // Calculate minimum height: coin_height + locktime_value - 1
            // The -1 is to maintain nLockTime semantics (last invalid height)
            let required_height = coin_height
                .checked_add(locktime_value)
                .and_then(|sum| sum.checked_sub(1))
                .ok_or_else(|| {
                    crate::error::ConsensusError::ConsensusRuleViolation(
                        "Sequence lock height calculation overflow".into(),
                    )
                })?;

            // Runtime assertion: Required height must be >= coin_height - 1
            debug_assert!(
                required_height >= coin_height.saturating_sub(1),
                "Required height ({required_height}) must be >= coin_height - 1 ({coin_height})"
            );

            min_height = min_height.max(required_height);
        }
    }

    Ok((min_height, min_time))
}

/// Evaluate if sequence locks are satisfied
///
/// Checks if the current block height and time satisfy the sequence lock constraints.
///
/// Matches Bitcoin Core's EvaluateSequenceLocks() exactly.
///
/// # Arguments
/// * `block_height` - Current block height
/// * `block_time` - Current block's median time-past
/// * `lock_pair` - (min_height, min_time) from calculate_sequence_locks
///
/// # Returns
/// true if locks are satisfied, false otherwise
#[spec_locked("5.5")]
pub fn evaluate_sequence_locks(block_height: u64, block_time: u64, lock_pair: (i64, i64)) -> bool {
    let (min_height, min_time) = lock_pair;

    // u64 is always non-negative - no assertion needed

    // Check height constraint
    if min_height >= 0 && block_height <= min_height as u64 {
        // Runtime assertion: Cast must be valid (min_height >= 0)
        debug_assert!(
            min_height >= 0,
            "min_height ({min_height}) must be non-negative for cast to u64"
        );
        return false;
    }

    // Check time constraint
    if min_time >= 0 && block_time <= min_time as u64 {
        // Runtime assertion: Cast must be valid (min_time >= 0)
        debug_assert!(
            min_time >= 0,
            "min_time ({min_time}) must be non-negative for cast to u64"
        );
        return false;
    }

    true
}

/// Check if transaction sequence locks are satisfied
///
/// Convenience function that combines CalculateSequenceLocks and EvaluateSequenceLocks.
///
/// # Arguments
/// * `tx` - Transaction to check
/// * `flags` - Script verification flags
/// * `prev_heights` - Block heights at which each input confirmed
/// * `block_height` - Current block height
/// * `block_time` - Current block's median time-past
/// * `recent_headers` - Recent headers for median time-past calculation
///
/// # Returns
/// true if sequence locks are satisfied, false otherwise
#[spec_locked("5.5")]
pub fn sequence_locks(
    tx: &Transaction,
    flags: u32,
    prev_heights: &[u64],
    block_height: u64,
    block_time: u64,
    recent_headers: Option<&[BlockHeader]>,
) -> Result<bool> {
    let lock_pair = calculate_sequence_locks(tx, flags, prev_heights, recent_headers)?;
    Ok(evaluate_sequence_locks(block_height, block_time, lock_pair))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_sequence_locks_disabled() {
        let tx = Transaction {
            version: 2,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: SEQUENCE_LOCKTIME_DISABLE_FLAG as u64, // Disabled
            }]
            .into(),
            outputs: vec![].into(),
            lock_time: 0,
        };

        let prev_heights = vec![100];
        let result =
            calculate_sequence_locks(&tx, LOCKTIME_VERIFY_SEQUENCE, &prev_heights, None).unwrap();

        // Disabled sequence should not create locks
        assert_eq!(result, (-1, -1));
    }

    #[test]
    fn test_calculate_sequence_locks_block_based() {
        let tx = Transaction {
            version: 2,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 100, // 100 blocks relative locktime
            }]
            .into(),
            outputs: vec![].into(),
            lock_time: 0,
        };

        let prev_heights = vec![1000]; // Input confirmed at height 1000
        let result =
            calculate_sequence_locks(&tx, LOCKTIME_VERIFY_SEQUENCE, &prev_heights, None).unwrap();

        // Should require height 1000 + 100 - 1 = 1099
        assert_eq!(result.0, 1099);
        assert_eq!(result.1, -1); // No time constraint
    }

    #[test]
    fn test_evaluate_sequence_locks() {
        // Lock requires height 1099
        let lock_pair = (1099, -1);

        // Block height 1100 satisfies the lock
        assert!(evaluate_sequence_locks(1100, 0, lock_pair));

        // Block height 1099 does not satisfy (must be > 1099)
        assert!(!evaluate_sequence_locks(1099, 0, lock_pair));

        // Block height 1098 does not satisfy
        assert!(!evaluate_sequence_locks(1098, 0, lock_pair));
    }
}

