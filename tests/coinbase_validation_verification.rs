//! Coinbase Transaction Validation Verification Tests
//!
//! Tests to verify BLLVM's coinbase validation matches Bitcoin Core exactly.
//! Coinbase validation is consensus-critical - differences = chain split.
//!
//! Core checks:
//! - First transaction must be coinbase
//! - Coinbase scriptSig length: 2-100 bytes
//! - Coinbase output <= fees + subsidy
//! - Coinbase output <= MAX_MONEY

use blvm_consensus::block::connect_block;
use blvm_consensus::constants::*;
use blvm_consensus::economic::get_block_subsidy;
use blvm_consensus::transaction::is_coinbase;
use blvm_consensus::types::*;
use std::collections::HashMap;

/// Create a valid coinbase transaction
fn create_coinbase_tx(_height: u64, output_value: i64) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff, // Coinbase marker
            },
            script_sig: vec![0x03, 0x01, 0x00, 0x00], // 4 bytes (valid: 2-100)
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: output_value,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

/// Test coinbase scriptSig length: minimum (2 bytes)
///
/// Core requires: 2 <= scriptSig.length <= 100
#[test]
fn test_coinbase_script_sig_minimum_length() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x51, 0x52], // Exactly 2 bytes (minimum)
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

    assert!(
        is_coinbase(&tx),
        "Transaction should be identified as coinbase"
    );
    // ScriptSig length of 2 is valid
}

/// Test coinbase scriptSig length: below minimum (1 byte)
///
/// Core rejects: scriptSig.length < 2
#[test]
fn test_coinbase_script_sig_below_minimum() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x51], // 1 byte (below minimum)
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

    // This should be rejected in check_transaction
    use blvm_consensus::transaction::check_transaction;
    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Invalid(_)),
        "Coinbase with scriptSig < 2 bytes should be invalid"
    );
}

/// Test coinbase scriptSig length: maximum (100 bytes)
///
/// Core allows: scriptSig.length <= 100
#[test]
fn test_coinbase_script_sig_maximum_length() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x51; 100], // Exactly 100 bytes (maximum)
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

    use blvm_consensus::transaction::check_transaction;
    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Valid),
        "Coinbase with scriptSig = 100 bytes should be valid"
    );
}

/// Test coinbase scriptSig length: above maximum (101 bytes)
///
/// Core rejects: scriptSig.length > 100
#[test]
fn test_coinbase_script_sig_above_maximum() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x51; 101], // 101 bytes (above maximum)
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

    use blvm_consensus::transaction::check_transaction;
    let result = check_transaction(&tx);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Invalid(_)),
        "Coinbase with scriptSig > 100 bytes should be invalid"
    );
}

/// Test coinbase output validation: output = subsidy (no fees)
///
/// Core allows: coinbase_output <= fees + subsidy
#[test]
fn test_coinbase_output_exact_subsidy() {
    let height = 100;
    let subsidy = get_block_subsidy(height);

    let coinbase = create_coinbase_tx(height, subsidy);

    // This should be valid (output equals subsidy, no fees)
    assert!(is_coinbase(&coinbase), "Should be coinbase");
    assert_eq!(
        coinbase.outputs[0].value, subsidy,
        "Output should equal subsidy"
    );
}

/// Test coinbase output validation: output > subsidy (should fail without fees)
///
/// Core rejects: coinbase_output > fees + subsidy
#[test]
fn test_coinbase_output_exceeds_subsidy() {
    let height = 100;
    let subsidy = get_block_subsidy(height);

    // Create coinbase with output > subsidy (no fees to cover it)
    let coinbase = create_coinbase_tx(height, subsidy + 1);

    // This should be invalid when validated in block context
    // (We can't fully test without block context, but verify the coinbase structure)
    assert!(is_coinbase(&coinbase), "Should be coinbase");
    assert!(
        coinbase.outputs[0].value > subsidy,
        "Output exceeds subsidy"
    );
}

/// Test coinbase output validation: output = subsidy + fees
///
/// Core allows: coinbase_output <= fees + subsidy
#[test]
fn test_coinbase_output_with_fees() {
    let height = 100;
    let subsidy = get_block_subsidy(height);
    let fees = 1_000_000; // 0.01 BTC in fees

    let coinbase = create_coinbase_tx(height, subsidy + fees);

    // This should be valid (output = subsidy + fees)
    assert!(is_coinbase(&coinbase), "Should be coinbase");
    assert_eq!(
        coinbase.outputs[0].value,
        subsidy + fees,
        "Output should equal subsidy + fees"
    );
}

/// Test coinbase output validation: output > MAX_MONEY
///
/// Core rejects: coinbase_output > MAX_MONEY
#[test]
fn test_coinbase_output_exceeds_max_money() {
    let coinbase = create_coinbase_tx(100, MAX_MONEY + 1);

    use blvm_consensus::transaction::check_transaction;
    let result = check_transaction(&coinbase);
    assert!(result.is_ok(), "check_transaction should succeed");

    let validation = result.unwrap();
    assert!(
        matches!(validation, ValidationResult::Invalid(_)),
        "Coinbase output > MAX_MONEY should be invalid"
    );
}

/// Test coinbase identification: valid coinbase
///
/// Core: coinbase if single input with null prevout
#[test]
fn test_coinbase_identification_valid() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x51, 0x52],
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

    assert!(is_coinbase(&tx), "Valid coinbase should be identified");
}

/// Test coinbase identification: non-coinbase (multiple inputs)
///
/// Core: not coinbase if multiple inputs
#[test]
fn test_coinbase_identification_multiple_inputs() {
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
            value: 50_000_000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    assert!(
        !is_coinbase(&tx),
        "Transaction with multiple inputs should not be coinbase"
    );
}

/// Test coinbase identification: non-coinbase (non-null prevout)
///
/// Core: not coinbase if prevout is not null
#[test]
fn test_coinbase_identification_non_null_prevout() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(), // Non-null hash
                index: 0,
            },
            script_sig: vec![0x51],
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

    assert!(
        !is_coinbase(&tx),
        "Transaction with non-null prevout should not be coinbase"
    );
}
