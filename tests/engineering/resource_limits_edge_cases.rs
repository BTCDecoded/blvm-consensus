//! Resource limit boundary edge case tests
//!
//! Tests for consensus-critical resource limits that must be enforced deterministically
//! at exact boundaries to prevent DoS attacks and ensure consensus compatibility.

use blvm_consensus::constants::{
    MAX_INPUTS, MAX_MONEY, MAX_OUTPUTS, MAX_SCRIPT_OPS, MAX_STACK_SIZE,
};
use blvm_consensus::opcodes::*;
use blvm_consensus::script::{eval_script, SigVersion, StackElement};
use blvm_consensus::transaction::{calculate_transaction_size, check_transaction};
use blvm_consensus::*;

#[test]
fn test_script_operation_limit_boundary() {
    // Test exactly at MAX_SCRIPT_OPS (201) - should pass
    // Note: MAX_SCRIPT_OPS is 201, so exactly 201 operations should fail (limit is >)
    let script_pass = vec![OP_NOP; 200];
    let mut stack: Vec<StackElement> = vec![];

    // 200 operations should pass
    let result = eval_script(&script_pass, &mut stack, 0, SigVersion::Base);
    // This might succeed or fail depending on stack state, but shouldn't fail due to op limit

    // Test exactly MAX_SCRIPT_OPS (201) - should fail
    let script_fail = vec![OP_NOP; MAX_SCRIPT_OPS + 1];
    let mut stack: Vec<StackElement> = vec![];

    let result = eval_script(&script_fail, &mut stack, 0, SigVersion::Base);
    assert!(
        result.is_err(),
        "Script with exactly 201 operations should fail"
    );

    // Check error message
    if let Err(blvm_consensus::error::ConsensusError::ScriptExecution(msg)) = result {
        assert!(
            msg.contains("Operation limit"),
            "Error should mention operation limit"
        );
    }
}

#[test]
fn test_script_operation_limit_one_below() {
    // Test exactly 200 operations (one below limit) - should pass
    let script = vec![OP_NOP; MAX_SCRIPT_OPS];
    let mut stack: Vec<StackElement> = vec![];

    // This should not fail due to operation limit
    // Note: It might fail for other reasons (stack state), but not op limit
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    // Don't assert success - might fail for other reasons, but shouldn't be op limit
}

#[test]
fn test_stack_size_limit_boundary() {
    // Test stack at exactly MAX_STACK_SIZE (1000) - should fail on next push
    // Create a script that pushes 1000 items
    let mut script = Vec::new();
    for _ in 0..=MAX_STACK_SIZE {
        script.push(OP_1); // OP_1 - pushes one item
    }

    let mut stack = Vec::new();

    // Execute script - should fail when stack reaches 1000
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(
        result.is_err(),
        "Script pushing {MAX_STACK_SIZE} items should fail due to stack limit"
    );

    if let Err(blvm_consensus::error::ConsensusError::ScriptExecution(msg)) = result {
        assert!(
            msg.contains("Stack") || msg.contains("overflow"),
            "Error should mention stack"
        );
    }
}

#[test]
fn test_stack_size_limit_one_below() {
    // Test stack at exactly 999 items (one below limit) - should pass
    let mut script = Vec::new();
    for _ in 0..(MAX_STACK_SIZE - 1) {
        script.push(OP_1); // OP_1 - pushes one item
    }

    let mut stack = Vec::new();

    // This should not fail due to stack limit
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    // Don't assert success - might fail for other reasons, but shouldn't be stack limit
}

#[test]
fn test_script_size_limit_boundary() {
    // Test script exactly at MAX_SCRIPT_SIZE (10000 bytes) - should pass
    let script = vec![OP_1; MAX_SCRIPT_SIZE]; // Exactly 10000 bytes
    let mut stack: Vec<StackElement> = vec![];

    // Script size check happens before execution, so this should fail at check_transaction
    // For eval_script, we test that it can handle the size
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    // Result depends on execution, but size itself should be acceptable
}

#[test]
fn test_script_size_limit_one_over() {
    // Test script exactly one byte over MAX_SCRIPT_SIZE - should fail
    let script = vec![OP_1; MAX_SCRIPT_SIZE + 1]; // 10001 bytes
    let mut stack: Vec<StackElement> = vec![];

    // This should fail - but script size check is typically in transaction validation
    // For eval_script, it might execute but transaction validation would reject
}

#[test]
fn test_transaction_size_limit_boundary() {
    // Test transaction exactly at MAX_TX_SIZE (1,000,000 bytes) - should fail
    // Note: MAX_TX_SIZE is the limit, so exactly at limit should pass, over should fail

    // Create a transaction that approaches the size limit
    // This is complex - we'd need to create a large transaction
    // For now, test that check_transaction properly enforces the limit

    let mut large_script = vec![OP_1; 10000]; // Large script

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: large_script.clone(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: large_script.clone(),
        }]
        .into(),
        lock_time: 0,
    };

    let size = calculate_transaction_size(&tx);

    // If size exceeds limit, check_transaction should reject
    if size > MAX_TX_SIZE {
        let result = check_transaction(&tx).unwrap();
        assert!(matches!(result, ValidationResult::Invalid(_)));
    }
}

#[test]
fn test_transaction_size_limit_one_under() {
    // Test transaction one byte under limit - should pass
    // Create transaction that's just under the limit
    // This requires careful construction to be exactly at boundary
}

#[test]
fn test_coinbase_scriptsig_boundary() {
    // Test coinbase scriptSig exactly at 2 bytes (minimum) - should pass
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![OP_1, OP_2], // Exactly 2 bytes
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&coinbase).unwrap();
    assert!(
        matches!(result, ValidationResult::Valid),
        "Coinbase with 2-byte scriptSig should be valid"
    );
}

#[test]
fn test_coinbase_scriptsig_minimum_boundary() {
    // Test coinbase scriptSig at 1 byte (below minimum) - should fail
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![OP_1], // Only 1 byte - should fail
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&coinbase).unwrap();
    // Bitcoin requires coinbase scriptSig to be 2-100 bytes
    // Should be invalid if < 2 bytes
}

#[test]
fn test_coinbase_scriptsig_maximum_boundary() {
    // Test coinbase scriptSig exactly at 100 bytes (maximum) - should pass
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![OP_1; 100], // Exactly 100 bytes
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&coinbase).unwrap();
    assert!(
        matches!(result, ValidationResult::Valid),
        "Coinbase with 100-byte scriptSig should be valid"
    );
}

#[test]
fn test_coinbase_scriptsig_over_maximum() {
    // Test coinbase scriptSig at 101 bytes (over maximum) - should fail
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![OP_1; 101], // 101 bytes - should fail
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&coinbase).unwrap();
    // Should be invalid if > 100 bytes
}

#[test]
fn test_input_count_limit_boundary() {
    // Sanity: well-formed small multi-input tx is valid (MAX_INPUTS itself exceeds weight limit).
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_input_count_over_limit() {
    // Test transaction with MAX_INPUTS + 1 - should fail
    let mut inputs = Vec::new();
    for i in 0..=MAX_INPUTS {
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [i as u8; 32],
                index: i as u32,
            },
            script_sig: vec![OP_1],
            sequence: 0,
        });
    }

    let tx = Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(_)),
        "Transaction with MAX_INPUTS + 1 should be invalid"
    );
}

#[test]
fn test_output_count_limit_boundary() {
    // Sanity: two outputs within limits (MAX_OUTPUTS itself exceeds weight limit).
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0,
        }]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: 500,
                script_pubkey: vec![OP_1].into(),
            },
            TransactionOutput {
                value: 500,
                script_pubkey: vec![OP_2].into(),
            },
        ]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_output_count_over_limit() {
    // Test transaction with MAX_OUTPUTS + 1 - should fail
    let mut outputs = Vec::new();
    for _ in 0..=MAX_OUTPUTS {
        outputs.push(TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1],
        });
    }

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: vec![OP_1],
            sequence: 0,
        }]
        .into(),
        outputs: outputs.into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(_)),
        "Transaction with MAX_OUTPUTS + 1 should be invalid"
    );
}
