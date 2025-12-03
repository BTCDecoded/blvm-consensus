//! Coinbase maturity edge cases
//!
//! Tests for consensus-critical coinbase maturity checking that must match
//! Bitcoin Core's behavior exactly.
//!
//! Consensus-critical: Coinbase maturity differences can cause different validation results.

use blvm_consensus::constants::COINBASE_MATURITY;
use blvm_consensus::transaction::check_tx_inputs;
use blvm_consensus::types::{
    OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet, UTXO,
};

/// Test that coinbase outputs cannot be spent before maturity
#[test]
fn test_coinbase_immature_rejected() {
    let mut utxo_set = UtxoSet::new();

    // Create a coinbase UTXO at height 0
    let coinbase_outpoint = OutPoint {
        hash: [1; 32].into(),
        index: 0,
    };
    let coinbase_utxo = UTXO {
        value: 50_000_000_000, // 50 BTC
        script_pubkey: vec![0x51],
        height: 0,
        is_coinbase: true, // This is a coinbase output
    };
    utxo_set.insert(coinbase_outpoint.clone(), coinbase_utxo);

    // Try to spend it at height 99 (one block before maturity)
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: coinbase_outpoint,
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let (result, _fee) = check_tx_inputs(&tx, &utxo_set, COINBASE_MATURITY - 1).unwrap();

    // Should be invalid: coinbase not yet mature
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Invalid(_)
    ));
}

/// Test that coinbase outputs can be spent exactly at maturity
#[test]
fn test_coinbase_mature_accepted() {
    let mut utxo_set = UtxoSet::new();

    // Create a coinbase UTXO at height 0
    let coinbase_outpoint = OutPoint {
        hash: [1; 32].into(),
        index: 0,
    };
    let coinbase_utxo = UTXO {
        value: 50_000_000_000,
        script_pubkey: vec![0x51],
        height: 0,
        is_coinbase: true,
    };
    utxo_set.insert(coinbase_outpoint.clone(), coinbase_utxo);

    // Spend it at height 100 (exactly at maturity)
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: coinbase_outpoint,
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let (result, _fee) = check_tx_inputs(&tx, &utxo_set, COINBASE_MATURITY).unwrap();

    // Should be valid: coinbase is mature
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Valid
    ));
}

/// Test that coinbase outputs can be spent after maturity
#[test]
fn test_coinbase_after_maturity_accepted() {
    let mut utxo_set = UtxoSet::new();

    // Create a coinbase UTXO at height 0
    let coinbase_outpoint = OutPoint {
        hash: [1; 32].into(),
        index: 0,
    };
    let coinbase_utxo = UTXO {
        value: 50_000_000_000,
        script_pubkey: vec![0x51],
        height: 0,
        is_coinbase: true,
    };
    utxo_set.insert(coinbase_outpoint.clone(), coinbase_utxo);

    // Spend it at height 200 (well after maturity)
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: coinbase_outpoint,
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let (result, _fee) = check_tx_inputs(&tx, &utxo_set, 200).unwrap();

    // Should be valid: coinbase is mature
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Valid
    ));
}

/// Test that non-coinbase UTXOs are not subject to maturity requirement
#[test]
fn test_non_coinbase_no_maturity_requirement() {
    let mut utxo_set = UtxoSet::new();

    // Create a non-coinbase UTXO at height 0
    let outpoint = OutPoint {
        hash: [1; 32].into(),
        index: 0,
    };
    let utxo = UTXO {
        value: 50_000_000_000,
        script_pubkey: vec![0x51],
        height: 0,
        is_coinbase: false, // Not a coinbase output
    };
    utxo_set.insert(outpoint.clone(), utxo);

    // Try to spend it immediately at height 0
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: outpoint,
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let (result, _fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();

    // Should be valid: non-coinbase UTXOs have no maturity requirement
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Valid
    ));
}

/// Test coinbase maturity with different creation heights
#[test]
fn test_coinbase_maturity_different_heights() {
    let mut utxo_set = UtxoSet::new();

    // Create a coinbase UTXO at height 50
    let coinbase_outpoint = OutPoint {
        hash: [1; 32].into(),
        index: 0,
    };
    let coinbase_utxo = UTXO {
        value: 50_000_000_000,
        script_pubkey: vec![0x51],
        height: 50, // Created at height 50
        is_coinbase: true,
    };
    utxo_set.insert(coinbase_outpoint.clone(), coinbase_utxo);

    // Try to spend it at height 149 (one block before maturity: 50 + 100 - 1)
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: coinbase_outpoint.clone(),
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let (result, _fee) = check_tx_inputs(&tx, &utxo_set, 149).unwrap();
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Invalid(_)
    ));

    // Spend it at height 150 (exactly at maturity: 50 + 100)
    let (result, _fee) = check_tx_inputs(&tx, &utxo_set, 150).unwrap();
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Valid
    ));
}
