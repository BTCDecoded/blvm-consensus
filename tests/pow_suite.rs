//! COV-C-04c: Proof-of-work API coverage (`pow.rs`).

use blvm_consensus::constants::{DIFFICULTY_ADJUSTMENT_INTERVAL, TARGET_TIME_PER_BLOCK};
use blvm_consensus::pow::{
    batch_check_proof_of_work, check_proof_of_work, difficulty_from_bits, expand_target,
    get_next_work_required, get_next_work_required_corrected,
};
use blvm_consensus::types::BlockHeader;

fn easy_header(nonce: u64) -> BlockHeader {
    BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [1; 32],
        timestamp: 1_231_006_505,
        bits: 0x0300ffff,
        nonce,
    }
}

#[test]
fn test_check_proof_of_work_succeeds_for_easy_bits_header() {
    let result = check_proof_of_work(&easy_header(0));
    assert!(result.is_ok(), "easy bits must not error in PoW check");
}

#[test]
fn test_check_proof_of_work_rejects_overlarge_target() {
    let mut header = easy_header(0);
    header.bits = 0xff00ffff;
    match check_proof_of_work(&header) {
        Ok(valid) => assert!(!valid),
        Err(_) => {}
    }
}

#[test]
fn test_difficulty_from_bits_genesis_near_one() {
    let difficulty = difficulty_from_bits(0x1d00ffff).unwrap();
    assert!(
        (difficulty - 1.0).abs() < 0.01,
        "genesis difficulty ≈ 1.0, got {difficulty}"
    );
}

#[test]
fn test_expand_target_rejects_low_exponent() {
    assert!(expand_target(0x020000ff).is_err());
}

#[test]
fn test_get_next_work_required_insufficient_headers_errors() {
    let current = easy_header(0);
    assert!(get_next_work_required(&current, &[]).is_err());
}

#[test]
fn test_get_next_work_required_normal_window() {
    let current = easy_header(0);
    let prev_headers: Vec<BlockHeader> = (0..DIFFICULTY_ADJUSTMENT_INTERVAL)
        .map(|i| BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32],
            merkle_root: [0; 32],
            timestamp: 1_231_006_505 + i * TARGET_TIME_PER_BLOCK,
            bits: 0x1d00ffff,
            nonce: 0,
        })
        .collect();
    let next = get_next_work_required(&current, &prev_headers).unwrap();
    assert_ne!(next, 0);
}

#[test]
fn test_get_next_work_required_corrected_can_differ_from_legacy() {
    let current = easy_header(0);
    let prev_headers: Vec<BlockHeader> = (0..DIFFICULTY_ADJUSTMENT_INTERVAL)
        .map(|i| BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32],
            merkle_root: [0; 32],
            timestamp: 1_231_006_505 + i * TARGET_TIME_PER_BLOCK,
            bits: 0x1d00ffff,
            nonce: 0,
        })
        .collect();
    let legacy = get_next_work_required(&current, &prev_headers).unwrap();
    let corrected = get_next_work_required_corrected(&current, &prev_headers).unwrap();
    assert_ne!(
        legacy, corrected,
        "corrected adjustment should differ from Bitcoin-compatible path on same window"
    );
}

#[test]
fn test_get_next_work_required_preserves_bits_when_target_mul_overflows() {
    let current = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [1; 32],
        timestamp: 10_000_000,
        bits: 0x207fffff,
        nonce: 0,
    };
    let start = 1_000_000u64;
    let prev_headers: Vec<BlockHeader> = (0..DIFFICULTY_ADJUSTMENT_INTERVAL)
        .map(|i| BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32],
            merkle_root: [0; 32],
            timestamp: start + i * TARGET_TIME_PER_BLOCK * 50,
            bits: 0x207fffff,
            nonce: 0,
        })
        .collect();
    let next = get_next_work_required(&current, &prev_headers).unwrap();
    assert_eq!(
        next, 0x207fffff,
        "regtest min-difficulty overflow should preserve prior bits"
    );
}

#[cfg(feature = "production")]
#[test]
fn test_batch_check_proof_of_work_empty_input() {
    assert!(batch_check_proof_of_work(&[]).unwrap().is_empty());
}

#[cfg(feature = "production")]
#[test]
fn test_batch_check_proof_of_work_marks_invalid_bits_as_false() {
    let mut bad = easy_header(0);
    bad.bits = 0xff00ffff;
    let batch = batch_check_proof_of_work(&[bad]).unwrap();
    assert_eq!(batch.len(), 1);
    assert!(!batch[0].0);
    assert!(batch[0].1.is_none());
}

#[test]
fn test_difficulty_from_bits_rejects_invalid_exponent() {
    assert!(difficulty_from_bits(0x020000ff).is_err());
}

#[cfg(feature = "production")]
#[test]
fn test_batch_check_proof_of_work_matches_serial() {
    let headers = vec![easy_header(0), easy_header(1)];
    let batch = batch_check_proof_of_work(&headers).unwrap();
    assert_eq!(batch.len(), 2);
    for (header, (valid, hash_opt)) in headers.iter().zip(batch.iter()) {
        assert_eq!(*valid, check_proof_of_work(header).unwrap());
        if *valid {
            assert!(hash_opt.is_some());
        }
    }
}
