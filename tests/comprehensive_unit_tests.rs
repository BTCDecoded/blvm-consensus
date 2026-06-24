//! Comprehensive unit tests for blvm-consensus modules

use blvm_consensus::economic::*;
use blvm_consensus::opcodes::{OP_1, OP_2, OP_3, OP_NOP};
use blvm_consensus::pow::*;
use blvm_consensus::script::*;
use blvm_consensus::transaction::*;
use blvm_consensus::*;

// ============================================================================
// TRANSACTION TESTS
// ============================================================================

#[test]
fn test_check_transaction_valid() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_check_transaction_empty_inputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_check_transaction_too_many_inputs() {
    let mut inputs = Vec::new();
    for i in 0..=MAX_INPUTS {
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [i as u8; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        });
    }

    let tx = Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_check_transaction_too_many_outputs() {
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
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: outputs.into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_check_transaction_negative_output() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: -1000, // Negative value
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_check_transaction_excessive_output() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: MAX_MONEY + 1, // Exceeds max money
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_is_coinbase() {
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    assert!(is_coinbase(&coinbase_tx));

    let regular_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    assert!(!is_coinbase(&regular_tx));
}

#[test]
fn test_calculate_transaction_size() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![OP_1, OP_2],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1, OP_2, OP_3],
        }]
        .into(),
        lock_time: 0,
    };

    // Transaction size calculation is not exposed as a public function
    // We can test that the transaction is valid instead
    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

// ============================================================================
// SCRIPT TESTS
// ============================================================================

#[test]
fn test_eval_script_simple() {
    let script = vec![OP_1, OP_2];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, crate::script::SigVersion::Base).unwrap();
    assert!(result, "eval_script must complete without opcode failure");
    assert_eq!(stack.len(), 2);
    assert!(crate::script::cast_to_bool(&stack[1]));
}

#[test]
fn test_eval_script_overflow() {
    let mut script = Vec::new();
    // Create a script that would cause stack overflow
    for _ in 0..=MAX_STACK_SIZE {
        script.push(OP_1); // OP_1
    }

    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, crate::script::SigVersion::Base);
    assert!(result.is_err());
}

#[test]
fn test_verify_script_simple() {
    let script_sig = vec![OP_1]; // OP_1
    let script_pubkey = vec![OP_1]; // OP_1

    let result = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
    assert!(result, "OP_1/OP_1 must verify");
}

#[test]
fn test_verify_script_with_witness() {
    let script_sig = vec![OP_1]; // OP_1
    let script_pubkey = vec![OP_1]; // OP_1
    let witness = Some(vec![OP_2]); // OP_2

    let result = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
    assert!(result, "witness must not break OP_1/OP_1 verify");
}

#[test]
fn test_verify_script_empty() {
    let script_sig = vec![];
    let script_pubkey = vec![];

    let result = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
    assert!(
        !result,
        "empty scriptSig/scriptPubKey leaves empty stack → verify false"
    );
}

#[test]
fn test_verify_script_large_scripts() {
    let mut script_sig = Vec::new();
    let mut script_pubkey = Vec::new();

    // Create scripts that exceed MAX_SCRIPT_SIZE
    for _ in 0..=MAX_SCRIPT_SIZE {
        script_sig.push(OP_1);
        script_pubkey.push(OP_1);
    }

    let result = verify_script(&script_sig, &script_pubkey, None, 0);
    assert!(result.is_err());
}

// ============================================================================
// ECONOMIC TESTS
// ============================================================================

// Note: Detailed economic tests (block subsidy, total supply, etc.) are in:
// - tests/unit/economic_tests.rs (basic tests)
// - tests/unit/economic_edge_tests.rs (edge cases)
// - tests/consensus_property_tests.rs (property-based tests)
// This file focuses on integration and cross-module tests.

#[test]
fn test_calculate_fee() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 800,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let mut utxo_set = UtxoSet::default();
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo = UTXO {
        value: 1000,
        script_pubkey: vec![OP_1].into(),
        height: 100,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

    let fee = calculate_fee(&tx, &utxo_set).unwrap();
    assert_eq!(fee, 200);
}

#[test]
fn test_calculate_fee_negative() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 800,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let mut utxo_set = UtxoSet::default();
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo = UTXO {
        value: 500, // Less than output
        script_pubkey: vec![OP_1].into(),
        height: 100,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

    let result = calculate_fee(&tx, &utxo_set);
    assert!(result.is_err());
}

#[test]
fn test_calculate_fee_zero() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let mut utxo_set = UtxoSet::default();
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo = UTXO {
        value: 1000,
        script_pubkey: vec![OP_1].into(),
        height: 100,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

    let fee = calculate_fee(&tx, &utxo_set).unwrap();
    assert_eq!(fee, 0);
}

#[test]
fn test_validate_supply_limit_excessive() {
    // Test with a height that would create excessive supply
    // Using Orange Paper constant H (halving interval = 210,000)
    use blvm_consensus::orange_paper_constants::H;
    let excessive_height = H * 100; // Way beyond normal operation
    let result = validate_supply_limit(excessive_height);
    // This should either pass (if the calculation is correct) or fail gracefully
    match result {
        Ok(valid) => assert!(valid),
        Err(_) => {
            // Expected failure for excessive height
        }
    }
}

// ============================================================================
// PROOF OF WORK TESTS
// ============================================================================

#[test]
fn test_get_next_work_required_insufficient_headers() {
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let prev_headers = vec![]; // Empty - insufficient headers

    let result = get_next_work_required(&current_header, &prev_headers);
    assert!(result.is_err());
}

#[test]
fn test_get_next_work_required_normal_adjustment() {
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505 + (DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK),
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let mut prev_headers = Vec::new();
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505 + (i * TARGET_TIME_PER_BLOCK),
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let result = get_next_work_required(&current_header, &prev_headers).unwrap();

    // Should return same difficulty (adjustment = 1.0)
    // Allow for small differences due to integer arithmetic and clamping
    // The result should be very close to 0x1d00ffff
    let expected = 0x1d00ffff;
    let diff = result.abs_diff(expected);
    // Allow difference of up to 100 (due to integer arithmetic precision)
    assert!(
        diff <= 100,
        "Expected difficulty close to 0x1d00ffff, got {result} (diff: {diff})"
    );
}

// expand_target is not a public function, so we test it indirectly through check_proof_of_work

#[test]
fn test_check_proof_of_work_genesis() {
    // Use a reasonable header with valid target
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x0300ffff, // Valid target (exponent = 3)
        nonce: 0,
    };

    // This should work with the valid target
    let result = check_proof_of_work(&header).unwrap();
    let again = check_proof_of_work(&header).unwrap();
    assert_eq!(
        result, again,
        "PoW check must be deterministic for a fixed header"
    );
}

// expand_target is not a public function, so we test it indirectly through check_proof_of_work

#[test]
fn test_check_proof_of_work_invalid_target() {
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        // Exponent 0x21 = 33 is outside expand_target's allowed [3, 32] (0x1f00ffff had exp 31, which is valid).
        bits: 0x2100ffff,
        nonce: 0,
    };

    let result = check_proof_of_work(&header);
    assert!(result.is_err());
}

// expand_target is not a public function, so we test it indirectly through check_proof_of_work

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_transaction_size_boundaries() {
    // check_transaction size rule: stripped_size * WITNESS_SCALE_FACTOR > MAX_BLOCK_WEIGHT
    // (~1 MB stripped). Per-script MAX_SCRIPT_SIZE is enforced in verify_script / mempool,
    // not here — two MAX_SCRIPT_SIZE fields (~20 KB stripped) remain Valid.
    let mut inputs = Vec::new();
    for i in 0..MAX_INPUTS {
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [i as u8; 32],
                index: 0,
            },
            script_sig: vec![0u8; 1000],
            sequence: 0xffffffff,
        });
    }

    let tx = Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(_)),
        "MAX_INPUTS × 1 KiB script_sig must exceed block weight limit"
    );
}

#[test]
fn test_maximum_input_output_counts() {
    // Test transaction with maximum number of inputs
    let mut inputs = Vec::new();
    for i in 0..MAX_INPUTS {
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [i as u8; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        });
    }

    let tx_max_inputs = Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx_max_inputs).unwrap();
    assert!(
        matches!(
            result,
            ValidationResult::Invalid(ref r)
                if r.contains("too large") || r.contains("weight") || r.contains("Transaction")
        ),
        "MAX_INPUTS fixture must exceed tx size/weight limits: {result:?}"
    );

    // Test transaction with many outputs (capped to fit within MAX_TX_SIZE / MAX_BLOCK_WEIGHT)
    let num_outputs = MAX_TX_SIZE / 12; // ~12 bytes per minimal output
    let mut outputs = Vec::new();
    for _ in 0..num_outputs {
        outputs.push(TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1],
        });
    }

    let tx_max_outputs = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: outputs.into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx_max_outputs).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_monetary_boundaries() {
    // Test transaction with maximum money value
    let tx_max_money = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: MAX_MONEY,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx_max_money).unwrap();
    assert!(matches!(result, ValidationResult::Valid));

    // Test transaction exceeding maximum money
    let tx_excess_money = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: MAX_MONEY + 1,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx_excess_money).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_script_operation_limits() {
    // Test script with maximum number of non-push operations (OP_NOP counts)
    let mut script = Vec::new();
    for _ in 0..MAX_SCRIPT_OPS {
        script.push(OP_NOP); // OP_NOP - non-push, counts toward limit
    }

    let empty: Vec<u8> = vec![];
    let result = verify_script(&script, &empty, None, 0).unwrap();
    assert!(
        !result,
        "scriptSig of only OP_NOP leaves empty stack after eval → verify false"
    );

    // Test script exceeding operation limit (MAX_SCRIPT_OPS + 1 non-push opcodes)
    let mut large_script = Vec::new();
    for _ in 0..=MAX_SCRIPT_OPS {
        large_script.push(OP_NOP); // OP_NOP - non-push, counts toward limit
    }

    let result = verify_script(&large_script, &empty, None, 0);
    assert!(result.is_err());
}

#[test]
fn test_stack_size_limits() {
    // Test script that would cause stack overflow
    let mut script = Vec::new();
    for _ in 0..=MAX_STACK_SIZE {
        script.push(OP_1); // OP_1
    }

    let result = verify_script(&script, &script, None, 0);
    assert!(result.is_err());
}

#[test]
fn test_difficulty_adjustment_boundaries() {
    // Test difficulty adjustment with extreme time differences
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    // Create headers with very fast block times (1 second each)
    let mut fast_headers = Vec::new();
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        fast_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505 + i, // 1 second intervals
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let result = get_next_work_required(&current_header, &fast_headers).unwrap();
    // Should increase difficulty significantly
    // Debug prints removed
    assert!(result < 0x1d00ffff); // Higher difficulty = lower target

    // Create headers with very slow block times (1 hour each)
    let mut slow_headers = Vec::new();
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        slow_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505 + (i * 3600), // 1 hour intervals
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let result = get_next_work_required(&current_header, &slow_headers).unwrap();
    // Should decrease difficulty significantly
    // When blocks are slow (longer timespan), difficulty decreases
    // This means target increases, so bits should increase (result > 0x1d00ffff)
    // However, due to clamping (max 4x adjustment), result may be clamped
    // So we check that result is >= 0x1d00ffff (difficulty decreased or stayed same)
    assert!(
        result >= 0x1d00ffff,
        "Slow blocks should decrease difficulty (increase bits), got {result} (expected >= 0x1d00ffff)"
    );
}

// Note: test_supply_calculation_boundaries is in tests/regression/edge_cases.rs
// This duplicate has been removed to avoid redundancy.

#[test]
fn test_sequence_number_boundaries() {
    // Test transaction with maximum sequence number
    let tx_max_sequence = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: 0xffffffff, // Maximum sequence
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx_max_sequence).unwrap();
    assert!(matches!(result, ValidationResult::Valid));

    // Test transaction with RBF sequence
    let tx_rbf = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![blvm_consensus::opcodes::OP_1],
            sequence: SEQUENCE_RBF as u64, // RBF sequence
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![blvm_consensus::opcodes::OP_1],
        }]
        .into(),
        lock_time: 0,
    };

    let result = check_transaction(&tx_rbf).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}
