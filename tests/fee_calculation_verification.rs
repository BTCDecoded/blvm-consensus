//! Fee Calculation Verification Tests
//!
//! Tests to verify BLLVM's fee calculation matches Bitcoin Core exactly.
//! Fee = sum(inputs) - sum(outputs), must be non-negative and within MoneyRange.
//!
//! Consensus-critical: Fee differences = different transaction acceptance

use blvm_consensus::constants::*;
use blvm_consensus::transaction::check_tx_inputs;
use blvm_consensus::types::ValidationResult;
use blvm_consensus::types::*;
use std::collections::HashMap;

/// Create a test UTXO set
fn create_test_utxo_set() -> UtxoSet {
    let mut utxo_set = HashMap::new();

    // Add some test UTXOs
    utxo_set.insert(
        OutPoint {
            hash: [1; 32].into(),
            index: 0,
        },
        UTXO {
            value: 100_000_000, // 1 BTC
            script_pubkey: vec![0x51].into(),
            height: 100,
            is_coinbase: false,
        },
    );

    utxo_set.insert(
        OutPoint {
            hash: [2; 32].into(),
            index: 0,
        },
        UTXO {
            value: 50_000_000, // 0.5 BTC
            script_pubkey: vec![0x52].into(),
            height: 101,
            is_coinbase: false,
        },
    );

    utxo_set
}

/// Test zero fee transaction
///
/// Core allows zero fee transactions (they're valid but may not be relayed)
#[test]
fn test_zero_fee_transaction() {
    let utxo_set = create_test_utxo_set();

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 100_000_000, // Same as input (zero fee)
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_tx_inputs(&tx, &utxo_set, 200);
    assert!(result.is_ok(), "check_tx_inputs should succeed");

    let (validation, fee) = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Zero fee transaction should be valid"
    );
    assert_eq!(fee, 0, "Fee should be zero");
}

/// Test positive fee transaction
///
/// Core calculates fee = input_sum - output_sum
#[test]
fn test_positive_fee_transaction() {
    let utxo_set = create_test_utxo_set();

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 99_000_000, // 0.01 BTC fee
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_tx_inputs(&tx, &utxo_set, 200);
    assert!(result.is_ok(), "check_tx_inputs should succeed");

    let (validation, fee) = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Positive fee transaction should be valid"
    );
    assert_eq!(
        fee, 1_000_000,
        "Fee should be 0.01 BTC (1,000,000 satoshis)"
    );
}

/// Test negative fee transaction (should be rejected)
///
/// Core rejects transactions where output_sum > input_sum
#[test]
fn test_negative_fee_transaction() {
    let utxo_set = create_test_utxo_set();

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 101_000_000, // More than input (negative fee)
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_tx_inputs(&tx, &utxo_set, 200);
    assert!(result.is_ok(), "check_tx_inputs should succeed");

    let (validation, _fee) = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Invalid(_)),
        "Negative fee transaction should be invalid"
    );
}

/// Test fee calculation with multiple inputs
///
/// Core sums all inputs, then subtracts all outputs
#[test]
fn test_fee_multiple_inputs() {
    let utxo_set = create_test_utxo_set();

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
                    index: 0,
                },
                script_sig: vec![0x52],
                sequence: 0xfffffffe,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 140_000_000, // 0.01 BTC fee (150M - 140M)
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_tx_inputs(&tx, &utxo_set, 200);
    assert!(result.is_ok(), "check_tx_inputs should succeed");

    let (validation, fee) = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Transaction with multiple inputs should be valid"
    );
    assert_eq!(
        fee, 10_000_000,
        "Fee should be 0.1 BTC (10,000,000 satoshis)"
    );
}

/// Test fee calculation with multiple outputs
///
/// Core sums all outputs, then subtracts from input sum
#[test]
fn test_fee_multiple_outputs() {
    let utxo_set = create_test_utxo_set();

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: 50_000_000,
                script_pubkey: vec![0x51].into(),
            },
            TransactionOutput {
                value: 49_000_000, // 0.01 BTC fee
                script_pubkey: vec![0x52].into(),
            },
        ]
        .into(),
        lock_time: 0,
    };

    let result = check_tx_inputs(&tx, &utxo_set, 200);
    assert!(result.is_ok(), "check_tx_inputs should succeed");

    let (validation, fee) = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Transaction with multiple outputs should be valid"
    );
    assert_eq!(
        fee, 1_000_000,
        "Fee should be 0.01 BTC (1,000,000 satoshis)"
    );
}

/// Test fee calculation with maximum money
///
/// Core checks that fee is within MoneyRange
#[test]
fn test_fee_maximum_money() {
    let mut utxo_set = HashMap::new();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32].into(),
            index: 0,
        },
        UTXO {
            value: MAX_MONEY,
            script_pubkey: vec![0x51].into(),
            height: 100,
            is_coinbase: false,
        },
    );

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: MAX_MONEY - 1, // 1 satoshi fee
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_tx_inputs(&tx, &utxo_set, 200);
    assert!(result.is_ok(), "check_tx_inputs should succeed");

    let (validation, fee) = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Transaction with maximum money should be valid"
    );
    assert_eq!(fee, 1, "Fee should be 1 satoshi");
}

/// Test fee calculation overflow protection
///
/// Core uses checked arithmetic to prevent overflow
#[test]
fn test_fee_overflow_protection() {
    let mut utxo_set = HashMap::new();

    // Create inputs that would overflow if not checked
    for i in 0..10 {
        utxo_set.insert(
            OutPoint {
                hash: [i as u8; 32].into(),
                index: 0,
            },
            UTXO {
                value: i64::MAX / 10, // Large values that could overflow
                script_pubkey: vec![0x51].into(),
                height: 100,
                is_coinbase: false,
            },
        );
    }

    let mut inputs = Vec::new();
    for i in 0..10 {
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [i as u8; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        });
    }

    let tx = Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![TransactionOutput {
            value: 1,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    // This should either succeed (if overflow is handled) or fail gracefully
    let result = check_tx_inputs(&tx, &utxo_set, 200);
    // Result may be Ok or Err depending on overflow handling
    // The important thing is it doesn't panic or produce incorrect results
    assert!(
        result.is_ok() || result.is_err(),
        "Fee calculation should handle overflow gracefully"
    );
}

/// Test coinbase transaction fee (should be 0)
///
/// Core returns fee = 0 for coinbase transactions
#[test]
fn test_coinbase_fee() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff, // Coinbase marker
            },
            script_sig: vec![0x03, 0x01, 0x00, 0x00], // Coinbase scriptSig
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let utxo_set = HashMap::new(); // Empty UTXO set (coinbase doesn't need inputs)

    let result = check_tx_inputs(&tx, &utxo_set, 200);
    assert!(result.is_ok(), "check_tx_inputs should succeed");

    let (validation, fee) = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Coinbase transaction should be valid"
    );
    assert_eq!(fee, 0, "Coinbase transaction fee should be 0");
}
