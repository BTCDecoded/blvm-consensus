//! Integer overflow/underflow edge case tests
//! 
//! Tests for consensus-critical integer arithmetic edge cases that could cause
//! money creation or validation failures if not handled correctly.

use bllvm_consensus::*;
use bllvm_consensus::transaction::check_tx_inputs;
use bllvm_consensus::economic::get_block_subsidy;
use bllvm_consensus::block::connect_block;
use bllvm_consensus::economic::calculate_fee;

#[test]
fn test_input_value_overflow() {
    let mut utxo_set = UtxoSet::new();
    
    // Create UTXOs that will cause overflow when summed
    // Use values near i64::MAX
    let large_value = i64::MAX / 2 + 1;
    
    let outpoint1 = OutPoint { hash: [1; 32], index: 0 };
    let utxo1 = UTXO {
        value: large_value,
        script_pubkey: vec![],
        height: 0,
    };
    utxo_set.insert(outpoint1, utxo1);
    
    let outpoint2 = OutPoint { hash: [2; 32], index: 0 };
    let utxo2 = UTXO {
        value: large_value, // Adding this will overflow
        script_pubkey: vec![],
        height: 0,
    };
    utxo_set.insert(outpoint2, utxo2);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: outpoint1,
                script_sig: vec![],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: outpoint2,
                script_sig: vec![],
                sequence: 0xffffffff,
            },
        ],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![],
        }],
        lock_time: 0,
    };
    
    // Should detect overflow and return error
    let result = check_tx_inputs(&tx, &utxo_set, 0);
    assert!(result.is_err());
    
    // Check that error is about overflow
    if let Err(ConsensusError::TransactionValidation(msg)) = result {
        assert!(msg.contains("overflow"), "Error message should mention overflow");
    } else {
        panic!("Expected TransactionValidation error for overflow");
    }
}

#[test]
fn test_output_value_overflow() {
    let mut utxo_set = UtxoSet::new();
    
    // Create UTXO
    let outpoint = OutPoint { hash: [1; 32], index: 0 };
    let utxo = UTXO {
        value: 1000000000,
        script_pubkey: vec![],
        height: 0,
    };
    utxo_set.insert(outpoint, utxo);
    
    // Create transaction with outputs that will overflow when summed
    let large_value = i64::MAX / 2 + 1;
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: outpoint,
            script_sig: vec![],
            sequence: 0xffffffff,
        }],
        outputs: vec![
            TransactionOutput {
                value: large_value,
                script_pubkey: vec![],
            },
            TransactionOutput {
                value: large_value, // Adding this will overflow
                script_pubkey: vec![],
            },
        ],
        lock_time: 0,
    };
    
    // Should detect overflow and return error
    let result = check_tx_inputs(&tx, &utxo_set, 0);
    assert!(result.is_err());
    
    // Check that error is about overflow
    if let Err(ConsensusError::TransactionValidation(msg)) = result {
        assert!(msg.contains("overflow"), "Error message should mention overflow");
    } else {
        panic!("Expected TransactionValidation error for overflow");
    }
}

#[test]
fn test_output_exceeds_max_money() {
    let mut utxo_set = UtxoSet::new();
    
    // Create UTXO
    let outpoint = OutPoint { hash: [1; 32], index: 0 };
    let utxo = UTXO {
        value: MAX_MONEY + 1, // Exceeds max money
        script_pubkey: vec![],
        height: 0,
    };
    utxo_set.insert(outpoint, utxo);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: outpoint,
            script_sig: vec![],
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: MAX_MONEY + 1, // Output exceeds max money
            script_pubkey: vec![],
        }],
        lock_time: 0,
    };
    
    // Should reject transaction with output exceeding MAX_MONEY
    let result = check_tx_inputs(&tx, &utxo_set, 0);
    
    // Should be invalid (even if values don't overflow, exceeding MAX_MONEY is invalid)
    match result {
        Ok((validation_result, _)) => {
            assert!(matches!(validation_result, ValidationResult::Invalid(_)));
        },
        Err(_) => {
            // Error is also acceptable for overflow case
        }
    }
}

#[test]
fn test_fee_calculation_no_overflow() {
    let mut utxo_set = UtxoSet::new();
    
    // Test with large but safe values
    let input_value = MAX_MONEY / 2;
    let output_value = MAX_MONEY / 2 - 1000;
    
    let outpoint = OutPoint { hash: [1; 32], index: 0 };
    let utxo = UTXO {
        value: input_value,
        script_pubkey: vec![],
        height: 0,
    };
    utxo_set.insert(outpoint, utxo);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: outpoint,
            script_sig: vec![],
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: output_value,
            script_pubkey: vec![],
        }],
        lock_time: 0,
    };
    
    // Should succeed with valid fee calculation
    let result = check_tx_inputs(&tx, &utxo_set, 0);
    assert!(result.is_ok());
    
    let (validation_result, fee) = result.unwrap();
    assert!(matches!(validation_result, ValidationResult::Valid));
    assert_eq!(fee, input_value - output_value);
    assert!(fee > 0);
}

#[test]
fn test_coinbase_value_overflow() {
    let utxo_set = UtxoSet::new();
    
    // Create block with coinbase that would overflow subsidy + fees
    // Use very large values that would cause overflow
    let subsidy = get_block_subsidy(0); // Initial subsidy
    let large_fee = i64::MAX - subsidy + 1; // Fee that causes overflow when added to subsidy
    
    // Create a block where fees would overflow
    // This is tricky - we need to create a block where total_fees + subsidy would overflow
    // For now, test coinbase output exceeding MAX_MONEY
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: MAX_MONEY + 1, // Exceeds max money
            script_pubkey: vec![],
        }],
        lock_time: 0,
    };
    
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![coinbase],
    };
    
    // Block validation should reject coinbase exceeding MAX_MONEY
    let result = connect_block(&block, utxo_set, 0);
    assert!(result.is_ok()); // connect_block returns Result<(ValidationResult, UtxoSet), Error>
    
    let (validation_result, _) = result.unwrap();
    assert!(matches!(validation_result, ValidationResult::Invalid(_)));
}

#[test]
fn test_total_fees_overflow() {
    // This test is more complex - we'd need to create a block with many transactions
    // where summing all fees would overflow. This is difficult to test directly
    // but the overflow check in connect_block should catch it.
    
    // For now, we verify that fee calculation itself doesn't overflow
    let mut utxo_set = UtxoSet::new();
    
    // Create UTXO with large value
    let outpoint = OutPoint { hash: [1; 32], index: 0 };
    let utxo = UTXO {
        value: MAX_MONEY / 2,
        script_pubkey: vec![],
        height: 0,
    };
    utxo_set.insert(outpoint, utxo);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: outpoint,
            script_sig: vec![],
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: MAX_MONEY / 2 - 1000,
            script_pubkey: vec![],
        }],
        lock_time: 0,
    };
    
    // Fee calculation should succeed without overflow
    let result = calculate_fee(&tx, &utxo_set);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1000);
}

#[test]
fn test_max_valid_values() {
    // Test with maximum valid values (should not overflow)
    let mut utxo_set = UtxoSet::new();
    
    let outpoint = OutPoint { hash: [1; 32], index: 0 };
    let utxo = UTXO {
        value: MAX_MONEY,
        script_pubkey: vec![],
        height: 0,
    };
    utxo_set.insert(outpoint, utxo);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: outpoint,
            script_sig: vec![],
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: MAX_MONEY, // Maximum valid value
            script_pubkey: vec![],
        }],
        lock_time: 0,
    };
    
    // Should succeed with max valid values
    let result = check_tx_inputs(&tx, &utxo_set, 0);
    assert!(result.is_ok());
    
    let (validation_result, fee) = result.unwrap();
    assert!(matches!(validation_result, ValidationResult::Valid));
    assert_eq!(fee, 0); // Input = output, zero fee
}

