//! BIP Interaction Tests
//! 
//! Tests for interactions between multiple BIPs in single transactions and blocks.
//! Covers SegWit + CLTV/CSV, Taproot + relative locktime, and mixed transaction types.

use blvm_consensus::*;
use blvm_consensus::segwit::*;
use blvm_consensus::script::verify_script_with_context_full;
use super::bip_test_helpers::*;

#[test]
fn test_segwit_with_cltv() {
    // Test SegWit transaction with CLTV locktime
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x00], // SegWit marker
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: {
                // ScriptPubkey with CLTV: OP_1 <locktime> OP_CHECKLOCKTIMEVERIFY
                let mut script = vec![0x51].into(); // OP_1
                script.extend_from_slice(&encode_script_int(400000));
                script.push(0xb1); // OP_CHECKLOCKTIMEVERIFY
                script
            },
        }].into(),
        lock_time: 500000, // >= required locktime
    };
    
    let witness = vec![vec![0x51]]; // Witness data
    
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![0x00, 0x14].into(), // P2WPKH
            height: 0,
        }),
    );
    
    // Validate SegWit transaction with CLTV
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let pv = vec![utxo.value];
    let psp: Vec<&blvm_consensus::types::ByteString> = vec![&utxo.script_pubkey];

    let witness_script = witness[0].clone();

    let result = verify_script_with_context_full(
        &input.script_sig,
        &tx.outputs[0].script_pubkey, // Validate output script with CLTV
        Some(&witness_script),
        0,
        &tx,
        0,
        &pv,
        &psp,
        Some(500000), // Block height for CLTV validation
        None,
        blvm_consensus::types::Network::Mainnet,
        blvm_consensus::script::SigVersion::WitnessV0,
        None,
        None,
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")] None,
    );

    assert!(result.is_ok());
}

#[test]
fn test_segwit_with_csv() {
    // Test SegWit transaction with CSV relative locktime
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x00], // SegWit marker
            sequence: 0x00050000, // 5 blocks relative locktime
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: {
                // ScriptPubkey with CSV: OP_1 <sequence> OP_CHECKSEQUENCEVERIFY
                let mut script = vec![0x51].into(); // OP_1
                script.extend_from_slice(&encode_script_int(0x00040000)); // 4 blocks required
                script.push(0xb2); // OP_CHECKSEQUENCEVERIFY
                script
            },
        }].into(),
        lock_time: 0,
    };
    
    let witness = vec![vec![0x51]];
    
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![0x00, 0x14].into(), // P2WPKH
            height: 0,
        }),
    );
    
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let pv = vec![utxo.value];
    let psp: Vec<&blvm_consensus::types::ByteString> = vec![&utxo.script_pubkey];

    let witness_script = witness[0].clone();

    let result = verify_script_with_context_full(
        &input.script_sig,
        &tx.outputs[0].script_pubkey, // Validate output with CSV
        Some(&witness_script),
        0,
        &tx,
        0,
        &pv,
        &psp,
        None,
        None,
        blvm_consensus::types::Network::Mainnet,
        blvm_consensus::script::SigVersion::WitnessV0,
        None,
        None,
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")] None,
    );

    // CSV validation: input sequence (5 blocks) >= required (4 blocks)
    assert!(result.is_ok());
}

#[test]
fn test_taproot_with_csv() {
    // Test Taproot transaction with CSV relative locktime
    use blvm_consensus::taproot::*;
    
    let output_key = [0x42u8; 32];
    let mut p2tr_script = vec![TAPROOT_SCRIPT_PREFIX];
    p2tr_script.extend_from_slice(&output_key);
    p2tr_script.push(0x00);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![], // Empty for Taproot
            sequence: 0x00060000, // 6 blocks relative locktime
        }].into(),
        outputs: vec![
            TransactionOutput {
                value: 1000,
                script_pubkey: p2tr_script.clone(),
            },
            TransactionOutput {
                value: 2000,
                script_pubkey: {
                    // Output with CSV requirement
                    let mut script = vec![0x51].into();
                    script.extend_from_slice(&encode_script_int(0x00050000)); // 5 blocks required
                    script.push(0xb2); // CSV
                    script
                },
            },
        ].into(),
        lock_time: 0,
    };
    
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: p2tr_script,
            height: 0,
        }),
    );
    
    // Validate Taproot transaction
    assert!(validate_taproot_transaction(&tx).unwrap());
    
    // Validate CSV in second output
    let p2tr_script: blvm_consensus::types::ByteString = create_p2tr_script(&output_key).into();
    let pv = vec![1000000i64];
    let psp: Vec<&blvm_consensus::types::ByteString> = vec![&p2tr_script];

    // CSV validation: input sequence (6 blocks) >= required (5 blocks)
    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &tx.outputs[1].script_pubkey, // CSV script
        None,
        0,
        &tx,
        0,
        &pv,
        &psp,
        None,
        None,
        blvm_consensus::types::Network::Mainnet,
        blvm_consensus::script::SigVersion::Base,
        None,
        None,
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")] None,
    );
    
    assert!(result.is_ok());
}

#[test]
fn test_mixed_block_segwit_and_taproot() {
    // Test block with both SegWit and Taproot transactions
    use blvm_consensus::taproot::*;
    
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
                    script_sig: vec![0x00],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0x00, 0x14].into(), // P2WPKH
                }].into(),
                lock_time: 0,
            },
            Transaction {
                // Taproot transaction
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [2; 32].into(), index: 0 },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: create_p2tr_script(&[1u8; 32].into()),
                }].into(),
                lock_time: 0,
            },
        ],
    };
    
    // Validate all transactions
    for (i, tx) in block.transactions.iter().enumerate() {
        if i == 0 {
            // Coinbase - skip Taproot validation
            continue;
        }
        
        // SegWit transaction
        if i == 1 {
            assert!(is_segwit_transaction(tx));
        }
        
        // Taproot transaction
        if i == 2 {
            assert!(validate_taproot_transaction(tx).unwrap());
            assert!(is_taproot_output(&tx.outputs[0]));
        }
    }
}

#[test]
fn test_segwit_taproot_cltv_combined() {
    // Test complex scenario: SegWit transaction with Taproot output that has CLTV
    use blvm_consensus::taproot::*;
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x00], // SegWit marker
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![
            TransactionOutput {
                value: 1000,
                script_pubkey: create_p2tr_script(&[1u8; 32].into()), // Taproot output
            },
            TransactionOutput {
                value: 2000,
                script_pubkey: {
                    // CLTV script
                    let mut script = vec![0x51];
                    script.extend_from_slice(&encode_script_int(400000));
                    script.push(0xb1); // CLTV
                    script
                },
            },
        ].into(),
        lock_time: 500000, // >= required for CLTV
    };
    
    let witness = vec![vec![0x51]];
    
    // Validate SegWit transaction
    assert!(is_segwit_transaction(&tx));
    
    // Validate Taproot output
    assert!(validate_taproot_transaction(&tx).unwrap());
    assert!(is_taproot_output(&tx.outputs[0]));
    
    // Validate CLTV in second output
    let cltv_sp: blvm_consensus::types::ByteString = vec![0x00, 0x14].into();
    let pv = vec![1000000i64];
    let psp: Vec<&blvm_consensus::types::ByteString> = vec![&cltv_sp];

    let witness_script = witness[0].clone();

    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &tx.outputs[1].script_pubkey, // CLTV script
        Some(&witness_script),
        0,
        &tx,
        0,
        &pv,
        &psp,
        Some(500000), // Block height for CLTV
        None,
        blvm_consensus::types::Network::Mainnet,
        blvm_consensus::script::SigVersion::WitnessV0,
        None,
        None,
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")] None,
    );
    
    assert!(result.is_ok());
}

#[test]
fn test_cltv_csv_combined() {
    // Test transaction with both CLTV and CSV in different outputs
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x51],
            sequence: 0x00050000, // 5 blocks for CSV
        }].into(),
        outputs: vec![
            TransactionOutput {
                value: 1000,
                script_pubkey: {
                    // CLTV output
                    let mut script = vec![0x51].into();
                    script.extend_from_slice(&encode_script_int(400000));
                    script.push(0xb1); // CLTV
                    script
                },
            },
            TransactionOutput {
                value: 2000,
                script_pubkey: {
                    // CSV output
                    let mut script = vec![0x51];
                    script.extend_from_slice(&encode_script_int(0x00040000)); // 4 blocks
                    script.push(0xb2); // CSV
                    script
                },
            },
        ].into(),
        lock_time: 500000, // For CLTV
    };
    
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![0x51].into(),
            height: 0,
        }),
    );
    
    let pv = vec![1000000i64];
    let base_sp: blvm_consensus::types::ByteString = vec![0x51].into();
    let psp: Vec<&blvm_consensus::types::ByteString> = vec![&base_sp];

    // Validate CLTV output
    let result_cltv = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &tx.outputs[0].script_pubkey,
        None,
        0,
        &tx,
        0,
        &pv,
        &psp,
        Some(500000), // Block height
        None,
        blvm_consensus::types::Network::Mainnet,
        blvm_consensus::script::SigVersion::Base,
        None,
        None,
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")] None,
    );
    assert!(result_cltv.is_ok());

    // Validate CSV output
    let result_csv = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &tx.outputs[1].script_pubkey,
        None,
        0,
        &tx,
        0,
        &pv,
        &psp,
        None,
        None,
        blvm_consensus::types::Network::Mainnet,
        blvm_consensus::script::SigVersion::Base,
        None,
        None,
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")] None,
    );
    // CSV: input sequence (5 blocks) >= required (4 blocks)
    assert!(result_csv.is_ok());
}

#[test]
fn test_block_weight_with_segwit_and_taproot() {
    // Test block weight calculation with both SegWit and Taproot transactions
    use blvm_consensus::segwit::calculate_block_weight;
    use blvm_consensus::taproot::*;
    
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
                    script_sig: vec![0x00],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0x00, 0x14].into(),
                }].into(),
                lock_time: 0,
            },
            Transaction {
                // Taproot transaction
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [2; 32].into(), index: 0 },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: create_p2tr_script(&[1u8; 32].into()),
                }].into(),
                lock_time: 0,
            },
        ],
    };
    
    // Create witnesses (SegWit has witness, Taproot has empty scriptSig)
    let witnesses = vec![
        vec![], // Coinbase
        vec![vec![0x51]], // SegWit witness
        vec![], // Taproot (no witness data in test)
    ];
    
    let block_weight = calculate_block_weight(&block, &witnesses).unwrap();
    
    assert!(block_weight > 0);
}

// Helper function for Taproot tests
fn create_p2tr_script(output_key: &[u8; 32]) -> Vec<u8> {
    let mut script = vec![blvm_consensus::taproot::TAPROOT_SCRIPT_PREFIX];
    script.extend_from_slice(output_key);
    script.push(0x00);
    script
}

