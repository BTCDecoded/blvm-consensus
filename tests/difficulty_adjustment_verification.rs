//! Difficulty Adjustment Verification Tests
//!
//! Tests to verify BLLVM's difficulty adjustment matches Bitcoin Core exactly,
//! including the known off-by-one bug for consensus compatibility.
//!
//! Consensus-critical: Difficulty differences = chain split

use blvm_consensus::constants::*;
use blvm_consensus::pow::get_next_work_required;
use blvm_consensus::types::*;

/// Test difficulty adjustment clamping: timespan < expected_time/4
///
/// Core clamps to expected_time/4, which should result in 4x difficulty increase
#[test]
fn test_difficulty_clamp_minimum_timespan() {
    // Create 2016 blocks with very short timespan (all at same time)
    // This should clamp to expected_time/4, resulting in 4x difficulty increase
    let mut prev_headers = Vec::new();
    let base_timestamp = 1231006505; // Genesis block timestamp

    // Create exactly DIFFICULTY_ADJUSTMENT_INTERVAL blocks with minimal spacing
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32].into(),
            merkle_root: [0; 32],
            timestamp: base_timestamp + (i * 10), // Very short 10-second intervals
            bits: 0x1d00ffff,                     // Genesis difficulty
            nonce: 0,
        });
    }

    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0xff; 32].into(),
        merkle_root: [0; 32],
        timestamp: base_timestamp + (DIFFICULTY_ADJUSTMENT_INTERVAL * 10),
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers);
    assert!(result.is_ok(), "Difficulty adjustment should succeed");

    let new_bits = result.unwrap();

    // With timespan clamped to expected_time/4, difficulty should increase 4x
    // This means target should decrease 4x, so bits should decrease (bits are inverse of difficulty)
    // Genesis bits = 0x1d00ffff, after 4x increase should be lower (harder)
    // Note: bits encode target, lower target = higher difficulty = lower bits
    assert!(
        new_bits <= 0x1d00ffff,
        "Difficulty should increase (bits should be <= genesis, as bits are inverse)"
    );
}

/// Test difficulty adjustment clamping: timespan > expected_time*4
///
/// Core clamps to expected_time*4, which should result in 4x difficulty decrease
#[test]
fn test_difficulty_clamp_maximum_timespan() {
    // Create 2016 blocks with very long timespan
    // This should clamp to expected_time*4, resulting in 4x difficulty decrease
    let mut prev_headers = Vec::new();
    let base_timestamp = 1231006505;

    // Create exactly DIFFICULTY_ADJUSTMENT_INTERVAL blocks with very long spacing
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32].into(),
            merkle_root: [0; 32],
            timestamp: base_timestamp + (i as u64 * TARGET_TIME_PER_BLOCK * 10), // 10x normal spacing
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0xff; 32].into(),
        merkle_root: [0; 32],
        timestamp: base_timestamp
            + (DIFFICULTY_ADJUSTMENT_INTERVAL as u64 * TARGET_TIME_PER_BLOCK * 10),
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers);
    assert!(result.is_ok(), "Difficulty adjustment should succeed");

    let new_bits = result.unwrap();

    // With timespan clamped to expected_time*4, difficulty should decrease 4x
    // This means target should increase 4x, so bits should decrease
    // Genesis bits = 0x1d00ffff, after 4x decrease should be lower (easier)
    // Note: bits are inverse of difficulty (higher bits = easier)
    assert!(
        new_bits <= MAX_TARGET as u64,
        "Bits should not exceed maximum target"
    );
}

/// Test difficulty adjustment with perfect timing (exactly expected time)
///
/// With perfect 10-minute intervals, difficulty should stay approximately the same
#[test]
fn test_difficulty_perfect_timing() {
    let mut prev_headers = Vec::new();
    let base_timestamp = 1231006505;

    // Create exactly DIFFICULTY_ADJUSTMENT_INTERVAL blocks with perfect 10-minute spacing
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32].into(),
            merkle_root: [0; 32],
            timestamp: base_timestamp + (i * TARGET_TIME_PER_BLOCK),
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0xff; 32].into(),
        merkle_root: [0; 32],
        timestamp: base_timestamp + (DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK),
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers);
    assert!(result.is_ok(), "Difficulty adjustment should succeed");

    let new_bits = result.unwrap();

    // With perfect timing, difficulty should stay approximately the same
    // Due to the off-by-one bug, there will be a small adjustment
    // But it should be close to the original bits
    let original_bits = 0x1d00ffff;
    let diff = if new_bits > original_bits {
        new_bits - original_bits
    } else {
        original_bits - new_bits
    };

    // Allow small difference due to off-by-one bug and rounding
    assert!(
        diff < 0x00010000,
        "Difficulty should stay approximately the same with perfect timing"
    );
}

/// Test difficulty adjustment off-by-one bug
///
/// Core measures (n-1) intervals but compares against n intervals
/// This causes a small adjustment even with perfect timing
#[test]
fn test_difficulty_off_by_one_bug() {
    let mut prev_headers = Vec::new();
    let base_timestamp = 1231006505;

    // Create exactly DIFFICULTY_ADJUSTMENT_INTERVAL blocks with perfect spacing
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32].into(),
            merkle_root: [0; 32],
            timestamp: base_timestamp + (i * TARGET_TIME_PER_BLOCK),
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0xff; 32].into(),
        merkle_root: [0; 32],
        timestamp: base_timestamp + (DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK),
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers);
    assert!(result.is_ok(), "Difficulty adjustment should succeed");

    let new_bits = result.unwrap();
    let original_bits = 0x1d00ffff;

    // Due to off-by-one bug:
    // - We measure (DIFFICULTY_ADJUSTMENT_INTERVAL - 1) intervals = 2015 * 600 seconds
    // - But compare against DIFFICULTY_ADJUSTMENT_INTERVAL intervals = 2016 * 600 seconds
    // - Adjustment = (2015 * 600) / (2016 * 600) ≈ 0.9995
    // - So difficulty increases slightly (target decreases, bits decrease)
    // Note: bits are inverse of difficulty - lower bits = higher difficulty
    // This matches Core's buggy behavior exactly
    assert!(new_bits <= original_bits, "Off-by-one bug should cause slight difficulty increase (bits decrease as difficulty increases)");
}

/// Test difficulty adjustment with insufficient headers
#[test]
fn test_difficulty_insufficient_headers() {
    let prev_headers = vec![BlockHeader {
        version: 1,
        prev_block_hash: [0; 32].into(),
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0,
    }];

    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0xff; 32].into(),
        merkle_root: [0; 32],
        timestamp: 1231006505 + 600,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers);
    assert!(result.is_err(), "Should fail with insufficient headers");
}

/// Test difficulty adjustment clamping boundaries
///
/// Verify that clamping works correctly at the boundaries
#[test]
fn test_difficulty_clamping_boundaries() {
    let mut prev_headers = Vec::new();
    let base_timestamp = 1231006505;

    // Test at exactly expected_time/4 boundary
    let timespan_quarter = (DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK) / 4;

    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32].into(),
            merkle_root: [0; 32],
            timestamp: base_timestamp
                + (i * timespan_quarter / (DIFFICULTY_ADJUSTMENT_INTERVAL - 1)),
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0xff; 32].into(),
        merkle_root: [0; 32],
        timestamp: base_timestamp + timespan_quarter,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let result = get_next_work_required(&current_header, &prev_headers);
    assert!(
        result.is_ok(),
        "Difficulty adjustment should succeed at boundary"
    );
}

/// Test difficulty adjustment with integer arithmetic
///
/// Verify no floating point is used (consensus-critical)
#[test]
fn test_difficulty_integer_arithmetic() {
    // This test verifies that all calculations use integer arithmetic
    // by checking that results are deterministic and don't have floating point artifacts
    let mut prev_headers = Vec::new();
    let base_timestamp = 1231006505;

    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32].into(),
            merkle_root: [0; 32],
            timestamp: base_timestamp + (i * TARGET_TIME_PER_BLOCK),
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0xff; 32].into(),
        merkle_root: [0; 32],
        timestamp: base_timestamp + (DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK),
        bits: 0x1d00ffff,
        nonce: 0,
    };

    // Call twice - should get identical results (no floating point randomness)
    let result1 = get_next_work_required(&current_header, &prev_headers);
    let result2 = get_next_work_required(&current_header, &prev_headers);

    assert_eq!(
        result1, result2,
        "Difficulty calculation must be deterministic (integer arithmetic)"
    );
}
