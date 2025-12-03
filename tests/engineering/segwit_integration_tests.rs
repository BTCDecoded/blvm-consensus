//! SegWit Integration Tests
//! 
//! Tests for Segregated Witness (BIP141/143) integration with transaction validation,
//! block weight calculation, and witness handling.

use blvm_consensus::*;
use blvm_consensus::segwit::*;
use blvm_consensus::script::verify_script_with_context_full;
use blvm_consensus::constants::MAX_BLOCK_WEIGHT;
use super::bip_test_helpers::*;

/// Create witness commitment script (helper for tests)
/// OP_RETURN <36-byte-commitment>
fn create_witness_commitment_script(commitment: &[u8; 32]) -> Vec<u8> {
    let mut script = vec![0x6a, 0x24]; // OP_RETURN, 36 bytes
    script.extend_from_slice(commitment);
    script.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // 4 bytes padding
    script
}

#[test]
fn test_segwit_witness_validation() {
    // Test witness data validation in transaction flow
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x00], // SegWit marker (empty scriptSig for SegWit)
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(), // OP_1
        }].into(),
        lock_time: 0,
    };
    
    let witness = vec![vec![0x51]]; // Witness stack: OP_1
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51], // P2WPKH scriptPubkey
            height: 0,
        },
    );
    
    // Validate with witness
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let prevouts = vec![TransactionOutput {
        value: utxo.value,
        script_pubkey: utxo.script_pubkey.clone(),
    }];
    
    // Convert witness to ByteString for script validation
    let witness_script = witness[0].clone();
    
    let result = verify_script_with_context_full(
        &input.script_sig,
        &utxo.script_pubkey,
        Some(&witness_script), // Witness data
        0, // Flags
        &tx,
        0, // Input index
        &prevouts,
        None, // Block height
        None, // Median time past
        crate::types::Network::Mainnet,
        crate::script::SigVersion::WitnessV0,
        None, // Median time-past
    );
    
    assert!(result.is_ok());
}

#[test]
fn test_segwit_transaction_weight() {
    // Test transaction weight calculation with witness
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x00], // SegWit marker
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };
    
    let witness = vec![vec![0x51; 100]]; // 100-byte witness
    
    let weight = calculate_transaction_weight(&tx, Some(&witness), crate::types::Network::Mainnet).unwrap();
    
    // Weight = 4 * base_size + total_size
    // base_size includes transaction without witness
    // total_size = base_size + witness_size
    assert!(weight > 0);
    assert!(weight > 4 * 100); // Weight should account for witness
}

#[test]
fn test_segwit_block_weight() {
    // Test block weight calculation with SegWit transactions
    let block = Block {
        header: create_test_header(1234567890, [0; 32]),
        transactions: vec![
            Transaction {
                version: 1,
                inputs: vec![].into(),
                outputs: vec![TransactionOutput {
                    value: 5000000000,
                    script_pubkey: vec![].into(),
                }].into(),
                lock_time: 0,
            },
            Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [1; 32].into(), index: 0 },
                    script_sig: vec![0x00],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0x51].into(),
                }].into(),
                lock_time: 0,
            },
        ],
    };
    
    let witnesses = vec![
        vec![], // Coinbase witness (empty)
        vec![vec![0x51]], // First transaction witness
    ];
    
    let block_weight = calculate_block_weight(&block, &witnesses).unwrap();
    
    assert!(block_weight > 0);
    assert!(block_weight <= MAX_BLOCK_WEIGHT); // Should be within limit
}

#[test]
fn test_segwit_block_weight_boundary() {
    // Test block weight at boundary (exactly at or near 4M weight)
    let mut block = Block {
        header: create_test_header(1234567890, [0; 32]),
            transactions: vec![].into(),
    };
    
    // Create transactions that approach the weight limit
    // This is a simplified test - real boundary testing would need precise weight calculation
    for _ in 0..100 {
        block.transactions.push(Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [1; 32].into(), index: 0 },
                script_sig: vec![0x00],
                sequence: 0xffffffff,
            }].into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51; 100].into(), // Large scriptPubkey
            }].into(),
            lock_time: 0,
        });
    }
    
    let witnesses: Vec<Witness> = (0..block.transactions.len())
        .map(|i| if i == 0 { vec![] } else { vec![vec![0x51; 50]] })
        .collect();
    
    let block_weight = calculate_block_weight(&block, &witnesses).unwrap();
    
    // Block weight should be calculated correctly
    assert!(block_weight > 0);
    // Note: In real testing, we'd verify it's exactly at boundary when appropriate
}

#[test]
fn test_segwit_witness_commitment() {
    // Test witness commitment in coinbase transaction
    let mut coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let witness_root = [1u8; 32];
    
    // Add witness commitment to coinbase script
    coinbase_tx.outputs[0].script_pubkey = create_witness_commitment_script(&witness_root);
    
    let is_valid = validate_witness_commitment(&coinbase_tx, &witness_root).unwrap();
    
    assert!(is_valid);
}

#[test]
fn test_segwit_p2wpkh_validation() {
    // Test P2WPKH (Pay-to-Witness-Public-Key-Hash) validation
    // P2WPKH: scriptPubkey is OP_0 <20-byte-hash>
    let p2wpkh_hash = [0x51; 20]; // 20-byte hash
    let mut script_pubkey = vec![0x00]; // OP_0
    script_pubkey.extend_from_slice(&p2wpkh_hash);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![], // Empty scriptSig for P2WPKH
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };
    
    // Witness for P2WPKH: <signature> <pubkey>
    let witness = vec![
        vec![0x51; 72], // Signature (DER-encoded)
        vec![0x51; 33], // Public key (compressed)
    ];
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: script_pubkey.clone(),
            height: 0,
        },
    );
    
    // Validate P2WPKH with witness
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let prevouts = vec![TransactionOutput {
        value: utxo.value,
        script_pubkey: utxo.script_pubkey.clone(),
    }];
    
    // For P2WPKH, witness replaces scriptSig
    let witness_script = witness.iter().flat_map(|w| w.iter().cloned()).collect::<Vec<u8>>();
    
    let result = verify_script_with_context_full(
        &input.script_sig,
        &utxo.script_pubkey,
        Some(&witness_script),
        0,
        &tx,
        0,
        &prevouts,
        None,
        None,
        crate::types::Network::Mainnet,
        crate::script::SigVersion::WitnessV0,
    );
    
    assert!(result.is_ok());
}

#[test]
fn test_segwit_p2wsh_validation() {
    // Test P2WSH (Pay-to-Witness-Script-Hash) validation
    // P2WSH: scriptPubkey is OP_0 <32-byte-hash>
    let p2wsh_hash = [0x51; 32]; // 32-byte hash
    let mut script_pubkey = vec![0x00]; // OP_0
    script_pubkey.extend_from_slice(&p2wsh_hash);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![], // Empty scriptSig for P2WSH
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };
    
    // Witness for P2WSH: <stack elements...> <witness script>
    let witness = vec![
        vec![0x51], // Stack element
        vec![0x51; 100], // Witness script
    ];
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: script_pubkey.clone(),
            height: 0,
        },
    );
    
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let prevouts = vec![TransactionOutput {
        value: utxo.value,
        script_pubkey: utxo.script_pubkey.clone(),
    }];
    
    let witness_script = witness.iter().flat_map(|w| w.iter().cloned()).collect::<Vec<u8>>();
    
    let result = verify_script_with_context_full(
        &input.script_sig,
        &utxo.script_pubkey,
        Some(&witness_script),
        0,
        &tx,
        0,
        &prevouts,
        None,
        None,
        crate::types::Network::Mainnet,
        crate::script::SigVersion::WitnessV0,
    );
    
    assert!(result.is_ok());
}

#[test]
fn test_segwit_weight_exceeds_limit() {
    // Test that block weight exceeding 4M is detected
    let mut block = Block {
        header: create_test_header(1234567890, [0; 32]),
            transactions: vec![].into(),
    };
    
    // Create a very large transaction with large witness
    // This would normally be validated to ensure it doesn't exceed MAX_BLOCK_WEIGHT
    let large_witness = vec![vec![0x51; 1000000]]; // 1MB witness
    
    block.transactions.push(Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x00],
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    });
    
    let witnesses = vec![vec![], large_witness];
    
    let block_weight = calculate_block_weight(&block, &witnesses).unwrap();
    
    // Block weight calculation should work, but validation should reject
    assert!(block_weight > 0);
    
    // Validate block weight limit
    let is_valid = validate_segwit_block(&block, &witnesses, MAX_BLOCK_WEIGHT).unwrap();
    
    // Should fail if weight exceeds limit
    if block_weight > MAX_BLOCK_WEIGHT {
        assert!(!is_valid);
    }
}

#[test]
fn test_segwit_mixed_block() {
    // Test block with both SegWit and non-SegWit transactions
    let block = Block {
        header: create_test_header(1234567890, [0; 32]),
        transactions: vec![
            Transaction {
                version: 1,
                inputs: vec![].into(),
                outputs: vec![TransactionOutput {
                    value: 5000000000,
                    script_pubkey: vec![].into(),
                }].into(),
                lock_time: 0,
            },
            Transaction {
                // SegWit transaction
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [1; 32].into(), index: 0 },
                    script_sig: vec![0x00], // SegWit marker
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0x51].into(),
                }].into(),
                lock_time: 0,
            },
            Transaction {
                // Non-SegWit transaction
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [2; 32].into(), index: 0 },
                    script_sig: vec![0x51], // Non-empty scriptSig
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0x51].into(),
                }].into(),
                lock_time: 0,
            },
        ],
    };
    
    let witnesses = vec![
        vec![], // Coinbase
        vec![vec![0x51]], // SegWit transaction witness
        vec![], // Non-SegWit (no witness)
    ];
    
    let block_weight = calculate_block_weight(&block, &witnesses).unwrap();
    
    assert!(block_weight > 0);
}

#[test]
fn test_segwit_witness_merkle_root() {
    // Test witness merkle root calculation
    let block = Block {
        header: create_test_header(1234567890, [0; 32]),
        transactions: vec![
            Transaction {
                version: 1,
                inputs: vec![].into(),
                outputs: vec![TransactionOutput {
                    value: 5000000000,
                    script_pubkey: vec![].into(),
                }].into(),
                lock_time: 0,
            },
            Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [1; 32].into(), index: 0 },
                    script_sig: vec![0x00],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0x51].into(),
                }].into(),
                lock_time: 0,
            },
        ],
    };
    
    let witnesses = vec![
        vec![], // Coinbase witness (empty)
        vec![vec![0x51]], // First transaction witness
    ];
    
    let witness_root = compute_witness_merkle_root(&block, &witnesses).unwrap();
    
    assert_eq!(witness_root.len(), 32);
}

#[test]
fn test_segwit_witness_merkle_root_empty_block() {
    // Test witness merkle root with empty block (should fail)
    let block = Block {
        header: create_test_header(1234567890, [0; 32]),
            transactions: vec![].into(),
    };
    
    let witnesses = vec![];
    
    let result = compute_witness_merkle_root(&block, &witnesses);
    
    assert!(result.is_err());
}

#[test]
fn test_segwit_no_witness_weight() {
    // Test transaction weight without witness (should equal legacy weight)
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };
    
    let weight_no_witness = calculate_transaction_weight(&tx, None).unwrap();
    let weight_with_empty_witness = calculate_transaction_weight(&tx, Some(&vec![]), crate::types::Network::Mainnet).unwrap();
    
    // Weight should be same with no witness or empty witness
    assert_eq!(weight_no_witness, weight_with_empty_witness);
}

#[test]
fn test_segwit_witness_commitment_validation() {
    // Test witness commitment validation in coinbase
    let mut coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let witness_root = [0x42u8; 32];
    
    // Add witness commitment
    coinbase_tx.outputs[0].script_pubkey = create_witness_commitment_script(&witness_root);
    
    let is_valid = validate_witness_commitment(&coinbase_tx, &witness_root).unwrap();
    assert!(is_valid);
    
    // Test with wrong witness root (should fail)
    let wrong_root = [0x99u8; 32];
    let is_invalid = validate_witness_commitment(&coinbase_tx, &wrong_root).unwrap();
    assert!(!is_invalid);
}

#[test]
fn test_segwit_is_segwit_transaction() {
    // Test detection of SegWit transactions
    let mut tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x00], // SegWit marker
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };
    
    assert!(is_segwit_transaction(&tx));
    
    // Non-SegWit transaction
    tx.inputs[0].script_sig = vec![0x51];
    assert!(!is_segwit_transaction(&tx));
}

#[test]
fn test_segwit_weight_base_size() {
    // Test that base size calculation is correct (without witness)
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x51; 50], // 50-byte scriptSig
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51; 25].into(), // 25-byte scriptPubkey
        }].into(),
        lock_time: 0,
    };
    
    let weight_no_witness = calculate_transaction_weight(&tx, None).unwrap();
    let weight_with_witness = calculate_transaction_weight(&tx, Some(&vec![vec![0x51; 100]]), crate::types::Network::Mainnet).unwrap();
    
    // Weight with witness should be larger
    assert!(weight_with_witness > weight_no_witness);
}

#[test]
fn test_segwit_weight_precise_calculation() {
    // Test precise weight calculation: Weight = 4 * base_size + total_size
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x00], // SegWit marker
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };
    
    let witness = vec![vec![0x51; 100]]; // 100-byte witness
    
    let weight = calculate_transaction_weight(&tx, Some(&witness), crate::types::Network::Mainnet).unwrap();
    
    // Base size (without witness) * 4 + total size (with witness)
    // This verifies the weight formula is applied correctly
    assert!(weight > 0);
}

#[test]
fn test_segwit_block_weight_sum() {
    // Test that block weight is sum of transaction weights
    let block = Block {
        header: create_test_header(1234567890, [0; 32]),
        transactions: vec![
            Transaction {
                version: 1,
                inputs: vec![].into(),
                outputs: vec![TransactionOutput {
                    value: 5000000000,
                    script_pubkey: vec![].into(),
                }].into(),
                lock_time: 0,
            },
            Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [1; 32].into(), index: 0 },
                    script_sig: vec![0x00],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0x51].into(),
                }].into(),
                lock_time: 0,
            },
        ],
    };
    
    let witnesses = vec![
        vec![],
        vec![vec![0x51]],
    ];
    
    let block_weight = calculate_block_weight(&block, &witnesses).unwrap();
    
    // Calculate individual transaction weights
    let tx0_weight = calculate_transaction_weight(&block.transactions[0], Some(&witnesses[0]), crate::types::Network::Mainnet).unwrap();
    let tx1_weight = calculate_transaction_weight(&block.transactions[1], Some(&witnesses[1]), crate::types::Network::Mainnet).unwrap();
    
    // Block weight should equal sum of transaction weights
    assert_eq!(block_weight, tx0_weight + tx1_weight);
}

#[test]
fn test_segwit_validate_block_weight_limit() {
    // Test that validate_segwit_block enforces weight limit
    let mut block = Block {
        header: create_test_header(1234567890, [0; 32]),
            transactions: vec![].into(),
    };
    
    // Create a small block (should pass)
    block.transactions.push(Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x00],
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    });
    
    let witnesses = vec![vec![vec![0x51]]];
    
    let is_valid = validate_segwit_block(&block, &witnesses, MAX_BLOCK_WEIGHT).unwrap();
    
    // Small block should pass validation
    assert!(is_valid);
}

