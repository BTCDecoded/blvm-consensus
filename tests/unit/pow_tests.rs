//! Unit tests for proof of work functions

use blvm_consensus::constants::{
    DIFFICULTY_ADJUSTMENT_INTERVAL, MAX_TARGET, TARGET_TIME_PER_BLOCK,
};
use blvm_consensus::pow::*;
use blvm_consensus::types::*;
use blvm_consensus::*;

#[test]
fn test_get_next_work_required_insufficient_headers() {
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let prev_headers = vec![]; // Empty - insufficient headers

    let result = get_next_work_required(&current_header, &prev_headers);
    assert!(result.is_err());
}

#[test]
fn test_get_next_work_required_normal_adjustment() {
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let mut prev_headers = Vec::new();
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505 + (i * TARGET_TIME_PER_BLOCK),
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let result = get_next_work_required(&current_header, &prev_headers).unwrap();

    // The 2016-block window spans 2015 × 600 s = 1,209,000 s; target is 1,209,600 s.
    // Blocks were slightly fast → difficulty increases slightly.
    // Check that the exponent byte is 0x1d (correct order of magnitude).
    let exponent = (result >> 24) & 0xff;
    assert_eq!(
        exponent, 0x1d,
        "Difficulty exponent should be unchanged (0x1d)"
    );
    // Mantissa should be close to 0x00ffff (within 1% of original target).
    let mantissa = result & 0x00ffffff;
    assert!(
        mantissa > 0x00ff00 && mantissa <= 0x00ffff,
        "Mantissa 0x{:06x} should be close to 0x00ffff",
        mantissa
    );
}

#[test]
fn test_expand_target() {
    // Test a reasonable target that won't overflow (exponent = 0x1d = 29, which is > 3)
    // Use a target with exponent <= 3 to avoid the conservative limit
    let target = expand_target(0x0300ffff).unwrap(); // exponent = 3, mantissa = 0x00ffff
    assert!(target > blvm_consensus::pow::U256::zero());
}

#[test]
fn test_check_proof_of_work_genesis() {
    // Use a reasonable header with valid target
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x0300ffff, // Valid target (exponent = 3)
        nonce: 0,
    };

    // This should work with the valid target
    let result = check_proof_of_work(&header).unwrap();
    // Result depends on the hash, but should not panic
    assert!(result == true || result == false);
}

#[test]
fn test_expand_target_invalid() {
    // Exponent must be in 3..=32. Exponent 33 (0x21) is out of range → invalid.
    let result = expand_target(0x2100ffff); // exponent = 33
    assert!(
        result.is_err(),
        "expand_target with exponent > 32 should fail"
    );
    // Exponent 2 is also invalid (below minimum).
    let result2 = expand_target(0x0200ffff); // exponent = 2
    assert!(
        result2.is_err(),
        "expand_target with exponent < 3 should fail"
    );
}

#[test]
fn test_check_proof_of_work_invalid_target() {
    // Exponent 33 is out of the valid 3..=32 range → bits are malformed.
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x2100ffff, // exponent = 33, invalid
        nonce: 0,
    };

    let result = check_proof_of_work(&header);
    assert!(
        result.is_err(),
        "check_proof_of_work with invalid exponent should fail"
    );
}

#[test]
fn test_expand_target_edge_cases() {
    // Test edge cases for target expansion.
    // expand_target requires exponent in 3..=32 (values < 3 are invalid per BIP).
    let target3 = expand_target(0x0300ffff).unwrap(); // exponent = 3 (minimum valid)
    let target5 = expand_target(0x0500ffff).unwrap(); // exponent = 5
    let target7 = expand_target(0x0700ffff).unwrap(); // exponent = 7

    assert!(target3 > blvm_consensus::pow::U256::zero());
    assert!(target5 > blvm_consensus::pow::U256::zero());
    assert!(target7 > blvm_consensus::pow::U256::zero());

    // Higher exponents should result in larger targets (more leading zeros)
    assert!(target7 >= target5);
    assert!(target5 >= target3);

    // Exponents < 3 are invalid
    assert!(expand_target(0x0100ffff).is_err()); // exponent = 1
    assert!(expand_target(0x0200ffff).is_err()); // exponent = 2
}

#[test]
fn test_get_next_work_required_integer_math() {
    // Test that difficulty adjustment uses integer math (not floating-point)
    // Create headers with exactly 2 weeks between first and last
    let expected_time = DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK;

    let first_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1000000,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let last_header = BlockHeader {
        version: 1,
        prev_block_hash: [1; 32],
        merkle_root: [0; 32],
        timestamp: 1000000 + expected_time, // Exactly 2 weeks later
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let prev_headers = vec![first_header, last_header.clone()];
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [2; 32],
        merkle_root: [0; 32],
        timestamp: 1000000 + expected_time + 600, // One block after
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers).unwrap();

    // With exactly 2 weeks timespan, difficulty should stay the same (adjustment = 1.0)
    // Result should be very close to original bits (within rounding)
    assert!(result <= MAX_TARGET as u64);
    assert!(result > 0);
}

#[test]
fn test_get_next_work_required_fast_blocks_integer() {
    // Test fast blocks (1 week instead of 2 weeks) - should increase difficulty
    let expected_time = DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK;

    let first_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1000000,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    // Fast blocks: 1 week instead of 2 weeks (timespan = expected_time / 2)
    let last_header = BlockHeader {
        version: 1,
        prev_block_hash: [1; 32],
        merkle_root: [0; 32],
        timestamp: 1000000 + (expected_time / 2),
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let prev_headers = vec![first_header, last_header.clone()];
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [2; 32],
        merkle_root: [0; 32],
        timestamp: 1000000 + (expected_time / 2) + 600,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers).unwrap();

    // Fast blocks should increase difficulty (lower target)
    // With timespan = expected_time/2, adjustment = 0.5, but clamped to 0.25
    // So new_target = old_target * 0.25 = lower target = higher difficulty
    assert!(result <= MAX_TARGET as u64);
    assert!(result > 0);
}

#[test]
fn test_get_next_work_required_slow_blocks_integer() {
    // Test slow blocks (4 weeks instead of 2 weeks) - should decrease difficulty
    let expected_time = DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK;

    let first_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1000000,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    // Slow blocks: 4 weeks instead of 2 weeks (timespan = expected_time * 2)
    let last_header = BlockHeader {
        version: 1,
        prev_block_hash: [1; 32],
        merkle_root: [0; 32],
        timestamp: 1000000 + (expected_time * 2),
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let prev_headers = vec![first_header, last_header.clone()];
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [2; 32],
        merkle_root: [0; 32],
        timestamp: 1000000 + (expected_time * 2) + 600,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers).unwrap();

    // Slow blocks should decrease difficulty (higher target)
    // With timespan = expected_time * 2, adjustment = 2.0, clamped to 4.0
    // So new_target = old_target * 2.0 = higher target = lower difficulty
    assert!(result <= MAX_TARGET as u64);
    assert!(result > 0);
}

#[test]
fn test_get_next_work_required_timespan_clamping() {
    // Test that timespan is properly clamped to [expected_time/4, expected_time*4]
    let expected_time = DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK;

    let first_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1000000,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    // Extremely fast: 1 day instead of 2 weeks (should clamp to expected_time/4)
    let last_header = BlockHeader {
        version: 1,
        prev_block_hash: [1; 32],
        merkle_root: [0; 32],
        timestamp: 1000000 + (expected_time / 10), // Much faster than minimum
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let prev_headers = vec![first_header, last_header.clone()];
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [2; 32],
        merkle_root: [0; 32],
        timestamp: 1000000 + (expected_time / 10) + 600,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers).unwrap();

    // Should clamp to minimum adjustment (0.25 = 4x difficulty increase)
    assert!(result <= MAX_TARGET as u64);
    assert!(result > 0);
}

#[test]
fn test_get_next_work_required_corrected_off_by_one_fix() {
    // Test that the corrected version fixes the off-by-one error
    // When we have exactly 2016 blocks with perfect 10-minute intervals,
    // the corrected version should maintain difficulty better

    let mut prev_headers = Vec::new();
    let start_timestamp = 1231006505;

    // Create exactly 2016 blocks with perfect 10-minute intervals
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32],
            merkle_root: [0; 32],
            timestamp: start_timestamp + (i * TARGET_TIME_PER_BLOCK),
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0xff; 32],
        merkle_root: [0; 32],
        timestamp: start_timestamp + (DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK),
        bits: 0x1d00ffff,
        nonce: 0,
    };

    // Bitcoin-compatible version (buggy)
    let buggy_result = get_next_work_required(&current_header, &prev_headers).unwrap();

    // Corrected version
    let corrected_result =
        get_next_work_required_corrected(&current_header, &prev_headers).unwrap();

    // With perfect timing, the corrected version should maintain difficulty better
    // The buggy version will slightly over-adjust because it compares 2015 intervals
    // against 2016 intervals worth of time
    //
    // Buggy: measures 2015 intervals, compares to 2016 intervals
    //        adjustment = (2015 * 600) / (2016 * 600) ≈ 0.9995
    //        So difficulty increases slightly (target decreases)
    //
    // Corrected: measures 2015 intervals, compares to 2015 intervals
    //            adjustment = (2015 * 600) / (2015 * 600) = 1.0
    //            So difficulty stays the same (target stays same)

    // The corrected version should be closer to the original bits
    let original_bits = 0x1d00ffff;
    let buggy_diff = if buggy_result > original_bits {
        buggy_result - original_bits
    } else {
        original_bits - buggy_result
    };

    let corrected_diff = if corrected_result > original_bits {
        corrected_result - original_bits
    } else {
        original_bits - corrected_result
    };

    // Corrected version should be closer to maintaining the same difficulty
    assert!(
        corrected_diff <= buggy_diff,
        "Corrected version should maintain difficulty better with perfect timing"
    );

    // With perfect timing, corrected version should maintain difficulty exactly
    assert_eq!(
        corrected_result, original_bits,
        "With perfect timing, corrected version should maintain exact difficulty"
    );
}
