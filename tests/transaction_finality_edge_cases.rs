//! Transaction finality edge cases
//!
//! Tests for IsFinalTx logic matching Bitcoin Core exactly.
//!
//! Consensus-critical: Finality differences = different transaction acceptance

use blvm_consensus::constants::{LOCKTIME_THRESHOLD, SEQUENCE_FINAL};
use blvm_consensus::mempool::is_final_tx;
use blvm_consensus::types::{Transaction, TransactionInput, TransactionOutput};

/// Test that locktime = 0 is always final
#[test]
fn test_finality_locktime_zero() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };

    // Locktime 0 should always be final
    assert!(
        is_final_tx(&tx, 0, 0),
        "Locktime 0 should be final at height 0"
    );
    assert!(
        is_final_tx(&tx, 100, 1000),
        "Locktime 0 should be final at any height"
    );
}

/// Test that height-based locktime uses < comparison
///
/// Bitcoin Core uses < comparison: if locktime < threshold, then locktime < height
/// This means locktime = height is NOT satisfied (must be locktime < height)
#[test]
fn test_finality_height_based_locktime() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 100, // Height-based locktime (< LOCKTIME_THRESHOLD)
    };

    // At height 100, locktime 100 is NOT satisfied (must be locktime < height, i.e., 100 < 100 is false)
    assert!(
        !is_final_tx(&tx, 100, 0),
        "Height 100 should NOT satisfy locktime 100 (must be locktime < height)"
    );

    // At height 101, locktime 100 IS satisfied (100 < 101 is true)
    assert!(
        is_final_tx(&tx, 101, 0),
        "Height 101 should satisfy locktime 100"
    );

    // At height 99, locktime 100 is NOT satisfied (100 < 99 is false)
    assert!(
        !is_final_tx(&tx, 99, 0),
        "Height 99 should NOT satisfy locktime 100"
    );
}

/// Test that timestamp-based locktime uses < comparison
///
/// Bitcoin Core uses < comparison: if locktime >= threshold, then locktime < block_time
#[test]
fn test_finality_timestamp_based_locktime() {
    let locktime_value = LOCKTIME_THRESHOLD as u64 + 1000; // Timestamp-based locktime

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: locktime_value,
    };

    // At time = locktime, locktime is NOT satisfied (must be locktime < block_time)
    assert!(
        !is_final_tx(&tx, 0, locktime_value),
        "Time equal to locktime should NOT satisfy (must be locktime < block_time)"
    );

    // At time = locktime + 1, locktime IS satisfied (locktime < block_time)
    assert!(
        is_final_tx(&tx, 0, locktime_value + 1),
        "Time greater than locktime should satisfy"
    );

    // At time = locktime - 1, locktime is NOT satisfied
    assert!(
        !is_final_tx(&tx, 0, locktime_value - 1),
        "Time less than locktime should NOT satisfy"
    );
}

/// Test SEQUENCE_FINAL override logic
///
/// If all inputs have SEQUENCE_FINAL (0xffffffff), the transaction is final
/// regardless of locktime. This allows transactions to bypass locktime.
#[test]
fn test_finality_sequence_final_override() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: SEQUENCE_FINAL as u64, // SEQUENCE_FINAL
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 10000, // Locktime that is NOT satisfied
    };

    // Even though locktime is not satisfied, transaction should be final
    // because all inputs have SEQUENCE_FINAL
    assert!(
        is_final_tx(&tx, 0, 0),
        "SEQUENCE_FINAL should override unsatisfied locktime"
    );
    assert!(
        is_final_tx(&tx, 1, 1),
        "SEQUENCE_FINAL should override unsatisfied locktime at any height"
    );
}

/// Test that mixed sequences prevent SEQUENCE_FINAL override
///
/// If any input does NOT have SEQUENCE_FINAL, the override doesn't apply.
#[test]
fn test_finality_mixed_sequences_no_override() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: blvm_consensus::types::OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: SEQUENCE_FINAL as u64, // First input has SEQUENCE_FINAL
            },
            TransactionInput {
                prevout: blvm_consensus::types::OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0, // Second input does NOT have SEQUENCE_FINAL
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 10000, // Locktime that is NOT satisfied
    };

    // Since not all inputs have SEQUENCE_FINAL, locktime must be satisfied
    assert!(
        !is_final_tx(&tx, 0, 0),
        "Mixed sequences should NOT override unsatisfied locktime"
    );
    assert!(
        !is_final_tx(&tx, 1, 1),
        "Mixed sequences should NOT override unsatisfied locktime"
    );
}

/// Test boundary condition: locktime = LOCKTIME_THRESHOLD
#[test]
fn test_finality_locktime_threshold_boundary() {
    // Locktime exactly at threshold should be treated as timestamp-based
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: LOCKTIME_THRESHOLD as u64, // Exactly at threshold
    };

    // Should be treated as timestamp-based (>= threshold)
    // At time = LOCKTIME_THRESHOLD, locktime LOCKTIME_THRESHOLD is NOT satisfied (must be locktime < block_time)
    assert!(
        !is_final_tx(&tx, 0, LOCKTIME_THRESHOLD as u64),
        "Time equal to threshold should NOT satisfy locktime at threshold"
    );

    // At time = LOCKTIME_THRESHOLD + 1, locktime LOCKTIME_THRESHOLD IS satisfied (LOCKTIME_THRESHOLD < LOCKTIME_THRESHOLD + 1)
    assert!(
        is_final_tx(&tx, 0, LOCKTIME_THRESHOLD as u64 + 1),
        "Time above threshold should satisfy locktime at threshold"
    );
}

/// Test that multiple inputs all need SEQUENCE_FINAL for override
#[test]
fn test_finality_all_inputs_sequence_final() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: blvm_consensus::types::OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: SEQUENCE_FINAL as u64,
            },
            TransactionInput {
                prevout: blvm_consensus::types::OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: SEQUENCE_FINAL as u64,
            },
            TransactionInput {
                prevout: blvm_consensus::types::OutPoint {
                    hash: [2; 32].into(),
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: SEQUENCE_FINAL as u64,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 50000, // Locktime that is NOT satisfied
    };

    // All inputs have SEQUENCE_FINAL, so transaction should be final
    assert!(
        is_final_tx(&tx, 0, 0),
        "All inputs with SEQUENCE_FINAL should override locktime"
    );
}
