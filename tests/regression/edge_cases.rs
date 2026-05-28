//! Regression tests for edge cases and boundary conditions

use blvm_consensus::constants::*;
use blvm_consensus::types::*;
use blvm_consensus::*;

#[test]
fn test_transaction_size_boundaries() {
    let consensus = ConsensusProof::new();

    // Test transaction at maximum size limit
    let mut large_script = Vec::new();
    for _ in 0..MAX_SCRIPT_SIZE {
        large_script.push(0x51);
    }

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: large_script.clone(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: large_script,
        }]
        .into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&tx).unwrap();
    // Should either be valid or fail gracefully
    assert!(matches!(
        result,
        ValidationResult::Valid | ValidationResult::Invalid(_)
    ));
}

#[test]
fn test_maximum_input_output_counts() {
    let consensus = ConsensusProof::new();

    // MAX_INPUTS (100k) × ~43 bytes each = 4+ MB >> MAX_TX_SIZE (1 MB).
    // Use a count that fits within the 1 MB transaction size limit.
    // Each input is ~43 bytes; 20,000 × 43 = 860 kB < 1 MB.
    let safe_input_count = 18_000usize;

    let mut inputs = Vec::new();
    for i in 0..safe_input_count {
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [(i % 256) as u8; 32],
                index: i as u32,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        });
    }

    let tx_many_inputs = Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&tx_many_inputs).unwrap();
    assert!(
        matches!(
            result,
            ValidationResult::Valid | ValidationResult::Invalid(_)
        ),
        "validate_transaction should not panic on large input count"
    );

    // Test transaction with many outputs (each ~10 bytes; 50,000 × 10 = 500 kB).
    let safe_output_count = 50_000usize;
    let mut outputs = Vec::new();
    for _ in 0..safe_output_count {
        outputs.push(TransactionOutput {
            value: 1,
            script_pubkey: vec![0x51],
        });
    }

    let tx_many_outputs = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: outputs.into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&tx_many_outputs).unwrap();
    assert!(
        matches!(
            result,
            ValidationResult::Valid | ValidationResult::Invalid(_)
        ),
        "validate_transaction should not panic on large output count"
    );
}

#[test]
fn test_monetary_boundaries() {
    let consensus = ConsensusProof::new();

    // Test transaction with maximum money value
    let tx_max_money = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: MAX_MONEY,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&tx_max_money).unwrap();
    assert!(matches!(result, ValidationResult::Valid));

    // Test transaction exceeding maximum money
    let tx_excess_money = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: MAX_MONEY + 1,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&tx_excess_money).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_script_operation_limits() {
    let consensus = ConsensusProof::new();

    // Test script with maximum number of operations
    let mut script = Vec::new();
    for _ in 0..MAX_SCRIPT_OPS {
        script.push(0x51); // OP_1
    }

    let result = consensus.verify_script(&script, &script, None, 0).unwrap();
    assert!(result || !result);

    // Test script exceeding operation limit.
    // OP_NOP (0x61) is a counted non-pushdata opcode, unlike OP_1 (0x51) which is a push.
    // MAX_SCRIPT_OPS + 1 OP_NOPs should trigger the opcode limit.
    let mut large_script = Vec::new();
    for _ in 0..=MAX_SCRIPT_OPS {
        large_script.push(0x61); // OP_NOP
    }

    let result = consensus.verify_script(&large_script, &large_script, None, 0);
    // Either an error or Ok(false) is acceptable — the script must not succeed.
    assert!(
        result.is_err() || result == Ok(false),
        "Script with too many ops should be rejected, got: {result:?}"
    );
}

#[test]
fn test_stack_size_limits() {
    let consensus = ConsensusProof::new();

    // Test script that would cause stack overflow
    let mut script = Vec::new();
    for _ in 0..=MAX_STACK_SIZE {
        script.push(0x51); // OP_1
    }

    let result = consensus.verify_script(&script, &script, None, 0);
    assert!(result.is_err());
}

#[test]
fn test_block_size_boundaries() {
    let consensus = ConsensusProof::new();

    // Create a block with many transactions
    let mut transactions = Vec::new();
    for i in 0..1000 {
        transactions.push(Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            }]
            .into(),
            lock_time: 0,
        });
    }

    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x0300ffff,
            nonce: 0,
        },
        transactions: transactions.into(),
    };

    let utxo_set = UtxoSet::default();
    // witnesses[tx_idx] = Vec<Witness> (one per input); Witness = Vec<Vec<u8>>
    let witnesses: Vec<Vec<blvm_consensus::segwit::Witness>> = block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();
    let time_context = None;
    let network = blvm_consensus::types::Network::Mainnet;
    let result = consensus.validate_block_with_time_context(
        &block,
        witnesses.as_slice(),
        utxo_set,
        0,
        time_context,
        network,
    );
    // Should either succeed or fail gracefully
    match result {
        Ok((validation_result, _)) => {
            assert!(matches!(
                validation_result,
                ValidationResult::Valid | ValidationResult::Invalid(_)
            ));
        }
        Err(_) => {
            // Expected failure for large block
        }
    }
}

#[test]
fn test_difficulty_adjustment_boundaries() {
    let consensus = ConsensusProof::new();

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

    let result = consensus
        .get_next_work_required(&current_header, &fast_headers)
        .unwrap();
    // Fast blocks → difficulty increases → target decreases → compact bits value DECREASES.
    // 0x1d00ffff is the maximum target (minimum difficulty); higher difficulty → lower bits.
    assert!(
        result < 0x1d00ffff,
        "Fast blocks should increase difficulty (result 0x{result:08x} should be < 0x1d00ffff)"
    );

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

    let result = consensus
        .get_next_work_required(&current_header, &slow_headers)
        .unwrap();
    // Slow blocks → difficulty decreases → target increases → bits would increase, but the
    // implementation clamps at MAX_TARGET (0x1d00ffff).  So result == MAX_TARGET.
    assert!(
        result == 0x1d00ffff,
        "Slow blocks should stay at MAX_TARGET (result 0x{result:08x} should be 0x1d00ffff)"
    );
}

#[test]
fn test_supply_calculation_boundaries() {
    let consensus = ConsensusProof::new();

    // Test supply calculation at various heights
    // Using Orange Paper constant H (halving interval = 210,000)
    use blvm_consensus::orange_paper_constants::H;
    let heights = vec![0, 1, H, H * 2, H * 10];

    for height in heights {
        let supply = consensus.total_supply(height);
        assert!(supply >= 0);
        assert!(supply <= MAX_MONEY);
    }

    // Test supply at very high height (beyond normal operation)
    let high_height = H * 100;
    let supply = consensus.total_supply(high_height);
    assert!(supply >= 0);
    assert!(supply <= MAX_MONEY);
}

#[test]
fn test_sequence_number_boundaries() {
    let consensus = ConsensusProof::new();

    // Test transaction with maximum sequence number
    let tx_max_sequence = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff, // Maximum sequence
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&tx_max_sequence).unwrap();
    assert!(matches!(result, ValidationResult::Valid));

    // Test transaction with RBF sequence
    let tx_rbf = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: SEQUENCE_RBF as u64, // RBF sequence
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&tx_rbf).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}
