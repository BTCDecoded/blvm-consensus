//! Duplicate Input Detection Verification Tests
//!
//! Tests to verify BLLVM's duplicate input detection matches Bitcoin Core exactly.
//! Core uses std::set, BLLVM uses HashSet - both should detect duplicates correctly.
//!
//! Consensus-critical: Duplicate inputs = inflation bug (CVE-2018-17144)

use bllvm_consensus::transaction::check_transaction;
use bllvm_consensus::types::ValidationResult;
use bllvm_consensus::types::*;

/// Test duplicate inputs with same hash and index
///
/// Core rejects this with "bad-txns-inputs-duplicate"
#[test]
fn test_duplicate_inputs_same_prevout() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(), // Same hash
                    index: 0,             // Same index
                },
                script_sig: vec![0x52],
                sequence: 0xfffffffe,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Invalid(_)),
        "Transaction with duplicate inputs should be invalid"
    );
}

/// Test duplicate inputs with same hash but different index
///
/// These should be valid (different prevouts)
#[test]
fn test_duplicate_inputs_same_hash_different_index() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(), // Same hash
                    index: 1,             // Different index
                },
                script_sig: vec![0x52],
                sequence: 0xfffffffe,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Transaction with same hash but different index should be valid"
    );
}

/// Test duplicate inputs with different hash but same index
///
/// These should be valid (different prevouts)
#[test]
fn test_duplicate_inputs_different_hash_same_index() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2; 32].into(), // Different hash
                    index: 0,             // Same index
                },
                script_sig: vec![0x52],
                sequence: 0xfffffffe,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Transaction with different hash but same index should be valid"
    );
}

/// Test multiple duplicate inputs
///
/// Core should reject on first duplicate found
#[test]
fn test_multiple_duplicate_inputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0, // Duplicate of first
                },
                script_sig: vec![0x52],
                sequence: 0xfffffffe,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x53],
                sequence: 0xfffffffd,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2; 32].into(),
                    index: 0, // Duplicate of third
                },
                script_sig: vec![0x54],
                sequence: 0xfffffffc,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Invalid(_)),
        "Transaction with multiple duplicate inputs should be invalid"
    );
}

/// Test all inputs are duplicates
///
/// Edge case: all inputs reference the same prevout
#[test]
fn test_all_inputs_duplicate() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x52],
                sequence: 0xfffffffe,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x53],
                sequence: 0xfffffffd,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Invalid(_)),
        "Transaction with all duplicate inputs should be invalid"
    );
}

/// Test no duplicate inputs (valid transaction)
#[test]
fn test_no_duplicate_inputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2; 32].into(),
                    index: 1,
                },
                script_sig: vec![0x52],
                sequence: 0xfffffffe,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [3; 32].into(),
                    index: 2,
                },
                script_sig: vec![0x53],
                sequence: 0xfffffffd,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Transaction with no duplicate inputs should be valid"
    );
}

/// Test duplicate detection with many inputs
///
/// Verify HashSet performance and correctness with large input sets
#[test]
fn test_duplicate_detection_many_inputs() {
    let mut inputs = Vec::new();

    // Create 100 unique inputs
    for i in 0..100 {
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [i as u8; 32].into(),
                index: i,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        });
    }

    // Add a duplicate of input 50
    inputs.push(TransactionInput {
        prevout: OutPoint {
            hash: [50; 32].into(),
            index: 50,
        },
        script_sig: vec![0x52],
        sequence: 0xfffffffe,
    });

    let tx = Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Invalid(_)),
        "Transaction with duplicate input should be invalid even with many inputs"
    );
}
