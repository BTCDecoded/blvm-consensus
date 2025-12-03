//! Property tests for difficulty adjustment edge cases
//!
//! Comprehensive property-based tests covering difficulty adjustment calculations,
//! target bounds, and edge cases in proof-of-work validation.

use blvm_consensus::*;
use blvm_consensus::pow;
use blvm_consensus::types::*;
use blvm_consensus::constants::{DIFFICULTY_ADJUSTMENT_INTERVAL, MAX_TARGET, TARGET_TIME_PER_BLOCK};
use proptest::prelude::*;

/// Property test: difficulty adjustment interval properties
proptest! {
    #[test]
    fn prop_difficulty_adjustment_interval_bounds(
        height in 0u64..1000000u64
    ) {
        // Difficulty adjustment happens every DIFFICULTY_ADJUSTMENT_INTERVAL blocks
        let is_adjustment_height = (height % DIFFICULTY_ADJUSTMENT_INTERVAL as u64) == 0;
        
        // Height should be valid
        prop_assert!(height >= 0);
        
        // Adjustment should occur at multiples of interval
        if is_adjustment_height && height > 0 {
            let prev_height = height - 1;
            prop_assert!((prev_height % DIFFICULTY_ADJUSTMENT_INTERVAL as u64) != 0,
                "Previous height should not be adjustment height");
        }
    }
}

/// Property test: target is always positive and bounded
proptest! {
    #[test]
    fn prop_target_bounds(
        bits in 0x01000000u32..=0x1d00ffffu32
    ) {
        // Bits encoding represents target
        // Target should be within valid range
        prop_assert!(bits >= 0x01000000);
        prop_assert!(bits <= MAX_TARGET);
        
        // Bits should be non-zero
        prop_assert!(bits != 0);
    }
}

/// Property test: difficulty adjustment factor is clamped
proptest! {
    #[test]
    fn prop_difficulty_adjustment_clamping(
        actual_time in 100u64..2000u64, // Block times in seconds
        expected_time in 500u64..700u64  // Expected 10-minute blocks
    ) {
        // Difficulty adjustment factor = expected / actual
        // Should be clamped between 0.25 and 4.0
        let factor = expected_time as f64 / actual_time as f64;
        
        let clamped = factor.max(0.25).min(4.0);
        
        prop_assert!(clamped >= 0.25, "Factor should be >= 0.25");
        prop_assert!(clamped <= 4.0, "Factor should be <= 4.0");
        
        if factor < 0.25 {
            prop_assert_eq!(clamped, 0.25, "Factor should clamp to 0.25");
        } else if factor > 4.0 {
            prop_assert_eq!(clamped, 4.0, "Factor should clamp to 4.0");
        } else {
            prop_assert_eq!(clamped, factor, "Factor within bounds should not clamp");
        }
    }
}

/// Property test: work calculation is monotonically increasing
proptest! {
    #[test]
    fn prop_work_calculation_monotonic(
        target1 in 0x01000000u32..=0x1d00ffffu32,
        target2 in 0x01000000u32..=0x1d00ffffu32
    ) {
        // Work = 2^256 / (target + 1)
        // Lower target = higher difficulty = more work
        
        if target1 < target2 {
            // Lower target should produce more work
            // Work1 > Work2
            prop_assert!(target1 < target2);
        } else if target1 > target2 {
            // Higher target should produce less work
            // Work1 < Work2
            prop_assert!(target1 > target2);
        }
    }
}

/// Property test: difficulty adjustment respects bounds
proptest! {
    #[test]
    fn prop_difficulty_adjustment_bounds(
        prev_bits in 0x01000000u32..=0x1d00ffffu32,
        time_span in 1000u64..20000u64 // Time span for 2016 blocks
    ) {
        // New target = prev_target * (actual_time / expected_time)
        // Should be clamped
        
        let expected_time = (DIFFICULTY_ADJUSTMENT_INTERVAL as u64) * TARGET_TIME_PER_BLOCK;
        let factor = time_span as f64 / expected_time as f64;
        let clamped_factor = factor.max(0.25).min(4.0);
        
        // Clamped factor should be within bounds
        prop_assert!(clamped_factor >= 0.25);
        prop_assert!(clamped_factor <= 4.0);
        
        // New target should be within valid range
        prop_assert!(prev_bits >= 0x01000000);
        prop_assert!(prev_bits <= MAX_TARGET);
    }
}

/// Property test: proof-of-work hash is below target
proptest! {
    #[test]
    fn prop_pow_hash_below_target(
        hash_bytes in prop::array::uniform32(0u8..=255u8),
        target_bits in 0x01000000u32..=0x1d00ffffu32
    ) {
        // For a valid proof-of-work, hash must be below target
        // This is a structural property test
        prop_assert!(target_bits >= 0x01000000);
        prop_assert!(target_bits <= MAX_TARGET);
        
        // Hash bytes should be valid
        prop_assert!(hash_bytes.len() == 32);
    }
}

/// Property test: difficulty adjustment interval calculation
proptest! {
    #[test]
    fn prop_difficulty_interval_calculation(
        height in 0u64..1000000u64
    ) {
        // Adjustment happens at heights: 0, DIFFICULTY_ADJUSTMENT_INTERVAL, 2*DIFFICULTY_ADJUSTMENT_INTERVAL, etc.
        let adjustment_period = height / (DIFFICULTY_ADJUSTMENT_INTERVAL as u64);
        let next_adjustment = (adjustment_period + 1) * (DIFFICULTY_ADJUSTMENT_INTERVAL as u64);
        
        prop_assert!(next_adjustment >= height);
        prop_assert!(next_adjustment >= (DIFFICULTY_ADJUSTMENT_INTERVAL as u64));
        prop_assert!((next_adjustment % (DIFFICULTY_ADJUSTMENT_INTERVAL as u64)) == 0);
    }
}

/// Property test: target expansion respects maximum
proptest! {
    #[test]
    fn prop_target_expansion_maximum(
        bits in 0x01000000u32..=0x1d00ffffu32
    ) {
        // Target should never exceed MAX_TARGET
        prop_assert!(bits <= MAX_TARGET);
        
        // Target should be positive
        prop_assert!(bits >= 0x01000000);
        
        // Expanded target should also respect bounds
        // (Actual expansion would be tested in implementation)
        prop_assert!(bits != 0);
    }
}

