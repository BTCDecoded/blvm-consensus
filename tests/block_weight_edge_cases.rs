//! Block weight calculation edge cases
//!
//! Tests for block weight calculation boundaries and SegWit discount.
//! Block weight = 4 × base_size + total_size
//! Maximum block weight: 4,000,000 weight units
//!
//! Consensus-critical: Incorrect weight calculation causes block rejection/acceptance divergence.

use blvm_consensus::opcodes::OP_1;
use blvm_consensus::segwit::Witness;
use blvm_consensus::segwit::calculate_transaction_weight;
use blvm_consensus::types::{OutPoint, Transaction, TransactionInput, TransactionOutput};

/// Maximum block weight: 4,000,000 weight units
pub const MAX_BLOCK_WEIGHT: u64 = 4_000_000;

/// Test block weight at exact 4MB limit
#[test]
fn test_block_weight_exact_limit() {
    // Create a transaction that would result in block weight at exactly 4MB
    // This is a simplified test - actual implementation would need to calculate
    // precise transaction sizes

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    let weight = calculate_transaction_weight(&tx, Some(&Witness::new())).expect("weight");
    assert_eq!(
        weight, 240,
        "one empty input and one empty output is 60 stripped bytes, weight 4 times that"
    );
}

/// Test block weight exceeding 4MB limit
#[test]
fn test_block_weight_exceeding_limit() {
    // Create a block that would exceed 4MB weight
    // This would need to be constructed with actual size calculations

    assert_eq!(
        blvm_consensus::witness::weight_to_vsize(MAX_BLOCK_WEIGHT + 1),
        1_000_001
    );
}

/// Test weight calculation with SegWit discount
///
/// SegWit discount: base_size counted 4x, witness_size counted 1x
/// Weight = 4 × base_size + total_size
#[test]
fn test_segwit_weight_discount() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    // Transaction without witness (non-SegWit)
    let weight_no_witness = calculate_transaction_weight(&tx, None);
    assert!(weight_no_witness.is_ok());

    // Transaction with witness (SegWit)
    let witness = Some(Witness::new());
    let weight_with_witness = calculate_transaction_weight(&tx, witness.as_ref());
    assert!(weight_with_witness.is_ok());

    let weight_no = weight_no_witness.expect("bare weight");
    let weight_with = weight_with_witness.expect("empty witness weight");
    assert_eq!(
        weight_no % 4,
        0,
        "no witness means weight is 4 times the base size"
    );
    assert_eq!(weight_with, weight_no, "an empty witness adds no bytes");
}

/// Test weight calculation with mixed witness/non-witness transactions
#[test]
fn test_mixed_witness_weight() {
    // Create a block with both SegWit and non-SegWit transactions
    let segwit_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    let non_segwit_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![OP_1], // OP_1
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    // Calculate weights separately
    let segwit_weight = calculate_transaction_weight(&segwit_tx, Some(&Witness::new()));
    let non_segwit_weight = calculate_transaction_weight(&non_segwit_tx, None);

    let segwit_weight = segwit_weight.expect("segwit weight");
    let non_segwit_weight = non_segwit_weight.expect("non-segwit weight");
    assert!(segwit_weight > 0);
    assert!(non_segwit_weight > 0);
    assert_ne!(
        segwit_weight, non_segwit_weight,
        "the extra script byte must change the weight"
    );
    assert!(segwit_weight < MAX_BLOCK_WEIGHT);
    assert!(non_segwit_weight < MAX_BLOCK_WEIGHT);
}

/// A short witness is added to the weight once.
#[test]
fn test_witness_bytes_counted_once() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    let bare = calculate_transaction_weight(&tx, None).expect("bare weight");
    let witness = vec![vec![0x11u8; 4]];
    let witnessed = calculate_transaction_weight(&tx, Some(&witness)).expect("witness weight");
    assert_eq!(witnessed - bare, 4, "witness element bytes are added once");
}

/// Test weight calculation with large witness data
#[test]
fn test_weight_large_witness() {
    // Create transaction with large witness data
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    // Create large witness (up to 520 bytes per element)
    let large_witness_element = vec![0x42; 520];
    let witness = Some(vec![large_witness_element]);

    let weight = calculate_transaction_weight(&tx, witness.as_ref());

    let bare = calculate_transaction_weight(&tx, None).expect("bare weight");
    let weight_value = weight.expect("large witness weight");
    assert_eq!(
        weight_value - bare,
        520,
        "a 520-byte witness element is added once"
    );
}

/// Test block weight boundary conditions
#[test]
fn test_block_weight_boundaries() {
    // Test exact boundary values
    let exact_limit = MAX_BLOCK_WEIGHT;
    let one_over_limit = MAX_BLOCK_WEIGHT + 1;
    let one_under_limit = MAX_BLOCK_WEIGHT - 1;

    assert_eq!(
        blvm_consensus::witness::weight_to_vsize(exact_limit),
        1_000_000
    );
    assert_eq!(
        blvm_consensus::witness::weight_to_vsize(one_over_limit),
        1_000_001
    );
    assert_eq!(
        blvm_consensus::witness::weight_to_vsize(one_under_limit),
        1_000_000
    );
}
