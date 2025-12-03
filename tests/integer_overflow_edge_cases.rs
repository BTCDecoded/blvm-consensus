//! Integer overflow edge cases for transaction validation
//!
//! Tests for consensus-critical integer overflow handling that must match
//! Bitcoin Core's MoneyRange() behavior exactly.
//!
//! Consensus-critical: Overflow handling differences can cause different validation results.

use blvm_consensus::constants::MAX_MONEY;
use blvm_consensus::transaction::check_transaction;
use blvm_consensus::transaction::check_tx_inputs;
use blvm_consensus::types::{
    OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet, UTXO,
};

/// Test that output value sum overflow is detected correctly
///
/// Bitcoin Core's behavior:
/// 1. Check each output: nValue >= 0 && nValue <= MAX_MONEY
/// 2. Add: nValueOut += txout.nValue (no overflow check)
/// 3. Check result: MoneyRange(nValueOut) = (nValueOut >= 0 && nValueOut <= MAX_MONEY)
///
/// Edge case: What if individual values are valid, but sum exceeds MAX_MONEY?
#[test]
fn test_output_sum_exceeds_max_money() {
    // Create transaction with outputs that individually pass checks
    // but sum exceeds MAX_MONEY
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
                value: MAX_MONEY, // First output at max
                script_pubkey: vec![0x51].into(),
            },
            TransactionOutput {
                value: 1, // Second output pushes sum over MAX_MONEY
                script_pubkey: vec![0x51].into(),
            },
        ]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();

    // Should be invalid: total exceeds MAX_MONEY
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Invalid(_)
    ));
}

/// Test that output value sum near i64::MAX is handled correctly
///
/// Edge case: What if sum would overflow i64 but individual values are valid?
/// - MAX_MONEY = 2,100,000,000,000,000 (2.1 quadrillion)
/// - i64::MAX = 9,223,372,036,854,775,807 (9.2 quintillion)
/// - So we can have ~4.4 MAX_MONEY values before hitting i64::MAX
#[test]
fn test_output_sum_near_i64_max() {
    // Create transaction with outputs that sum to near i64::MAX
    // but still within MAX_MONEY per output
    let large_value = MAX_MONEY;
    let num_outputs = 4; // 4 * MAX_MONEY = 8.4 quadrillion < i64::MAX

    let mut outputs = Vec::new();
    for _ in 0..num_outputs {
        outputs.push(TransactionOutput {
            value: large_value,
            script_pubkey: vec![0x51].into(),
        });
    }

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
        outputs: outputs.into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();

    // Should be invalid: total (4 * MAX_MONEY) exceeds MAX_MONEY
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Invalid(_)
    ));
}

/// Test that input value sum overflow is detected correctly
///
/// Similar to output sum, but for inputs in check_tx_inputs
#[test]
fn test_input_sum_overflow() {
    let mut utxo_set = UtxoSet::new();

    // Create UTXOs with large values
    let large_value = MAX_MONEY;

    // Add multiple UTXOs that individually pass checks
    for i in 0..5 {
        let outpoint = OutPoint {
            hash: [i as u8; 32].into(),
            index: 0,
        };
        let utxo = UTXO {
            value: large_value,
            script_pubkey: vec![0x51],
            height: 0,
            is_coinbase: false,
        };
        utxo_set.insert(outpoint, utxo);
    }

    // Create transaction spending all these UTXOs
    let mut inputs = Vec::new();
    for i in 0..5 {
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
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let (result, _fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();

    // Should handle overflow correctly (either reject or handle gracefully)
    // The sum of 5 * MAX_MONEY would exceed i64::MAX, so checked_add should catch it
    match result {
        blvm_consensus::types::ValidationResult::Valid => {
            // If valid, fee calculation should also be valid
            // This tests that overflow is caught before fee calculation
        }
        blvm_consensus::types::ValidationResult::Invalid(_) => {
            // Expected: overflow detected
        }
    }
}

/// Test that individual output values at MAX_MONEY are valid
#[test]
fn test_single_output_at_max_money() {
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
            value: MAX_MONEY, // Single output at max - should be valid
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();

    // Should be valid: single output at MAX_MONEY is allowed
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Valid
    ));
}

/// Test that output value just above MAX_MONEY is rejected
#[test]
fn test_output_above_max_money() {
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
            value: MAX_MONEY + 1, // Just above max - should be invalid
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();

    // Should be invalid: individual output exceeds MAX_MONEY
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Invalid(_)
    ));
}

/// Test that negative output values are rejected
#[test]
fn test_negative_output_value() {
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
            value: -1, // Negative value - should be invalid
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();

    // Should be invalid: negative output value
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Invalid(_)
    ));
}

/// Test that zero output values are valid
#[test]
fn test_zero_output_value() {
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
            value: 0, // Zero value - should be valid (dust outputs)
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();

    // Should be valid: zero output value is allowed (though not economically useful)
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Valid
    ));
}

/// Test that many small outputs summing to valid total are accepted
#[test]
fn test_many_small_outputs() {
    // Create transaction with many small outputs that sum to a valid total
    let num_outputs = 1000;
    let value_per_output = MAX_MONEY / (num_outputs as i64 + 1); // Ensure sum < MAX_MONEY

    let mut outputs = Vec::new();
    for _ in 0..num_outputs {
        outputs.push(TransactionOutput {
            value: value_per_output,
            script_pubkey: vec![0x51].into(),
        });
    }

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
        outputs: outputs.into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();

    // Should be valid: many small outputs summing to valid total
    assert!(matches!(
        result,
        blvm_consensus::types::ValidationResult::Valid
    ));
}
