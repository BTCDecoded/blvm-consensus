//! COV-C-06c: BIP68 sequence lock coverage (block- and time-based paths).

use blvm_consensus::sequence_locks::{
    calculate_sequence_locks, evaluate_sequence_locks, sequence_locks,
};
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

const LOCKTIME_VERIFY_SEQUENCE: u32 = 0x01;
const SEQUENCE_LOCKTIME_DISABLE_FLAG: u32 = 0x8000_0000;
const SEQUENCE_LOCKTIME_TYPE_FLAG: u32 = 0x0040_0000;

fn sample_tx(sequence: u64) -> Transaction {
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x11; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

#[test]
fn test_sequence_locks_convenience_satisfied() {
    let tx = sample_tx(50);
    let prev_heights = vec![100u64];
    let ok = sequence_locks(&tx, LOCKTIME_VERIFY_SEQUENCE, &prev_heights, 200, 0, None).unwrap();
    assert!(ok, "height 200 must satisfy block-based lock requiring 149");
}

#[test]
fn test_sequence_locks_convenience_not_satisfied() {
    let tx = sample_tx(50);
    let prev_heights = vec![100u64];
    let ok = sequence_locks(&tx, LOCKTIME_VERIFY_SEQUENCE, &prev_heights, 140, 0, None).unwrap();
    assert!(!ok, "height 140 must not satisfy lock requiring 149");
}

#[test]
fn test_time_based_sequence_lock_uses_prior_block_mtp() {
    let tx = sample_tx(SEQUENCE_LOCKTIME_TYPE_FLAG as u64 | 2);
    let prev_heights = vec![10u64];
    let coin_mtp = 1_000_000i64;
    let (min_height, min_time) = calculate_sequence_locks(
        &tx,
        LOCKTIME_VERIFY_SEQUENCE,
        &prev_heights,
        Some(&[coin_mtp]),
    )
    .unwrap();
    assert_eq!(min_height, -1);
    // 2 × 512 seconds, last-invalid: coin_mtp + 1024 - 1
    assert_eq!(min_time, 1_001_023);
    assert!(!evaluate_sequence_locks(
        100,
        min_time as u64,
        (min_height, min_time)
    ));
    assert!(evaluate_sequence_locks(
        100,
        min_time as u64 + 1,
        (min_height, min_time)
    ));
}

#[test]
fn test_time_based_sequence_lock_without_mtp_errors() {
    let tx = sample_tx(SEQUENCE_LOCKTIME_TYPE_FLAG as u64 | 2);
    let err = calculate_sequence_locks(&tx, LOCKTIME_VERIFY_SEQUENCE, &[10], None).unwrap_err();
    assert!(matches!(
        err,
        blvm_consensus::ConsensusError::ConsensusRuleViolation(_)
    ));
}

#[test]
fn test_calculate_sequence_locks_version_one_skips_bip68() {
    let mut tx = sample_tx(10);
    tx.version = 1;
    let prev_heights = vec![100u64];
    let result =
        calculate_sequence_locks(&tx, LOCKTIME_VERIFY_SEQUENCE, &prev_heights, None).unwrap();
    assert_eq!(result, (-1, -1));
}

#[test]
fn test_calculate_sequence_locks_prev_heights_mismatch_errors() {
    let tx = sample_tx(10);
    let err = calculate_sequence_locks(&tx, LOCKTIME_VERIFY_SEQUENCE, &[], None).unwrap_err();
    assert!(matches!(
        err,
        blvm_consensus::ConsensusError::ConsensusRuleViolation(_)
    ));
}

#[test]
fn test_disabled_sequence_does_not_contribute() {
    let tx = sample_tx(SEQUENCE_LOCKTIME_DISABLE_FLAG as u64 | 999);
    let prev_heights = vec![100u64];
    let result =
        calculate_sequence_locks(&tx, LOCKTIME_VERIFY_SEQUENCE, &prev_heights, None).unwrap();
    assert_eq!(result, (-1, -1));
}

#[test]
fn test_calculate_sequence_locks_block_based_returns_min_height() {
    let tx = sample_tx(50);
    let prev_heights = vec![100u64];
    let (min_height, min_time) =
        calculate_sequence_locks(&tx, LOCKTIME_VERIFY_SEQUENCE, &prev_heights, None).unwrap();
    assert_eq!(min_height, 149);
    assert_eq!(min_time, -1);
}

#[test]
fn test_evaluate_sequence_locks_block_height_boundary() {
    assert!(!evaluate_sequence_locks(149, 0, (149, -1)));
    assert!(evaluate_sequence_locks(150, 0, (149, -1)));
}
