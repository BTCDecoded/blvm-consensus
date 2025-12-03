//! Merkle root mutation detection tests (CVE-2012-2459)
//!
//! Tests for Bitcoin's merkle root calculation that detects mutations
//! (duplicate hashes at the same level) to prevent CVE-2012-2459 vulnerability.
//!
//! Consensus-critical: Merkle root differences = chain split

use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::types::{Transaction, TransactionInput, TransactionOutput};

/// Test that duplicate transaction hashes are detected as mutations
///
/// CVE-2012-2459: If duplicate transaction IDs exist at the same level in the merkle tree,
/// the merkle root calculation should detect this and reject the block.
#[test]
fn test_merkle_mutation_detection_duplicate_txids() {
    // Create two transactions with the same hash (simulated by using identical transactions)
    // In reality, this would require two transactions that hash to the same value
    // For testing, we'll create identical transactions which will have the same hash
    let tx1 = Transaction {
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

    // Create identical transaction (will have same hash)
    let tx2 = tx1.clone();

    // Merkle root calculation should detect mutation and reject
    let result = calculate_merkle_root(&[tx1, tx2]);
    assert!(
        result.is_err(),
        "Duplicate transaction hashes should be detected as mutation"
    );

    // Verify error message mentions CVE-2012-2459
    if let Err(e) = result {
        let error_msg = format!("{}", e);
        assert!(
            error_msg.contains("mutation") || error_msg.contains("CVE-2012-2459"),
            "Error message should mention mutation or CVE-2012-2459"
        );
    }
}

/// Test that normal transactions (no duplicates) produce valid merkle root
#[test]
fn test_merkle_root_normal_transactions() {
    let tx1 = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [1; 32].into(),
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

    let tx2 = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [2; 32].into(),
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 2000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };

    // Normal transactions should produce valid merkle root
    let result = calculate_merkle_root(&[tx1, tx2]);
    assert!(
        result.is_ok(),
        "Normal transactions should produce valid merkle root"
    );

    let merkle_root = result.unwrap();
    assert_eq!(merkle_root.len(), 32, "Merkle root must be 32 bytes");
}

/// Test that single transaction produces valid merkle root
#[test]
fn test_merkle_root_single_transaction() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![].into(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };

    // Single transaction should produce valid merkle root (duplicated and hashed)
    let result = calculate_merkle_root(&[tx]);
    assert!(
        result.is_ok(),
        "Single transaction should produce valid merkle root"
    );

    let merkle_root = result.unwrap();
    assert_eq!(merkle_root.len(), 32, "Merkle root must be 32 bytes");
}

/// Test that empty transaction list is rejected
#[test]
fn test_merkle_root_empty_list() {
    let result = calculate_merkle_root(&[]);
    assert!(result.is_err(), "Empty transaction list should be rejected");
}

/// Test that odd number of transactions works correctly
///
/// Bitcoin's merkle tree duplicates the last hash when there's an odd number
/// of transactions. This should not trigger mutation detection unless the
/// duplicated hash matches another hash at the same level.
#[test]
fn test_merkle_root_odd_number() {
    let tx1 = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [1; 32].into(),
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

    let tx2 = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [2; 32].into(),
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 2000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let tx3 = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [3; 32].into(),
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 3000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };

    // Three transactions (odd number) should work correctly
    let result = calculate_merkle_root(&[tx1, tx2, tx3]);
    assert!(
        result.is_ok(),
        "Odd number of transactions should produce valid merkle root"
    );

    let merkle_root = result.unwrap();
    assert_eq!(merkle_root.len(), 32, "Merkle root must be 32 bytes");
}
