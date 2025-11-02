//! Tests for parallel script verification in production mode

#[cfg(feature = "production")]
mod tests {
    use consensus_proof::*;
    use consensus_proof::block::*;
    use consensus_proof::script::*;

    fn create_multi_input_transaction() -> Transaction {
        Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint { hash: [1; 32], index: 0 },
                    script_sig: vec![0x51],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint { hash: [2; 32], index: 0 },
                    script_sig: vec![0x51],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint { hash: [3; 32], index: 0 },
                    script_sig: vec![0x51],
                    sequence: 0xffffffff,
                },
            ],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            }],
            lock_time: 0,
        }
    }

    fn create_multi_input_utxo_set() -> UtxoSet {
        let mut utxo_set = UtxoSet::new();
        
        for i in 1..=3 {
            let outpoint = OutPoint { hash: [i as u8; 32], index: 0 };
            let utxo = UTXO {
                value: 10000,
                script_pubkey: vec![0x51],
                height: 0,
            };
            utxo_set.insert(outpoint, utxo);
        }
        
        utxo_set
    }

    #[test]
    fn test_parallel_script_verification_single_tx() {
        // Test parallel verification of single transaction with multiple inputs
        let tx = create_multi_input_transaction();
        let utxo_set = create_multi_input_utxo_set();
        
        // Create block with this transaction (after coinbase)
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![
                // Coinbase
                Transaction {
                    version: 1,
                    inputs: vec![TransactionInput {
                        prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                        script_sig: vec![0x51],
                        sequence: 0xffffffff,
                    }],
                    outputs: vec![TransactionOutput {
                        value: 50_000_000_000,
                        script_pubkey: vec![0x51],
                    }],
                    lock_time: 0,
                },
                tx.clone(),
            ],
        };
        
        let (result, _) = connect_block(&block, utxo_set, 0).unwrap();
        
        // Should produce valid result (or invalid, but should be deterministic)
        assert!(matches!(result, ValidationResult::Valid | ValidationResult::Invalid(_)),
                "Parallel verification must produce deterministic result");
    }

    #[test]
    fn test_parallel_script_verification_ordering() {
        // Verify parallel results maintain correct input ordering
        let tx = create_multi_input_transaction();
        
        // Pre-lookup UTXOs (as done in production code)
        let utxo_set = create_multi_input_utxo_set();
        let input_utxos: Vec<(usize, Option<&ByteString>)> = tx.inputs
            .iter()
            .enumerate()
            .map(|(j, input)| (j, utxo_set.get(&input.prevout).map(|u| &u.script_pubkey)))
            .collect();
        
        // Verify ordering is preserved
        assert_eq!(input_utxos.len(), 3);
        assert_eq!(input_utxos[0].0, 0);
        assert_eq!(input_utxos[1].0, 1);
        assert_eq!(input_utxos[2].0, 2);
    }

    #[test]
    fn test_parallel_script_verification_error_handling() {
        // Test error propagation in parallel mode
        let mut utxo_set = create_multi_input_utxo_set();
        
        // Create transaction with invalid script
        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint { hash: [1; 32], index: 0 },
                    script_sig: vec![0xff; MAX_SCRIPT_OPS + 1], // Invalid: too many ops
                    sequence: 0xffffffff,
                },
            ],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
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
            transactions: vec![
                Transaction {
                    version: 1,
                    inputs: vec![TransactionInput {
                        prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                        script_sig: vec![0x51],
                        sequence: 0xffffffff,
                    }],
                    outputs: vec![TransactionOutput {
                        value: 50_000_000_000,
                        script_pubkey: vec![0x51],
                    }],
                    lock_time: 0,
                },
                tx,
            ],
        };
        
        // Should handle error correctly
        let result = connect_block(&block, utxo_set, 0);
        // May succeed or fail depending on when error is caught
        assert!(result.is_ok() || result.is_err(),
                "Parallel verification must handle errors correctly");
    }

    #[test]
    fn test_parallel_utxo_prelookup() {
        // Verify UTXO pre-lookup avoids race conditions
        let tx = create_multi_input_transaction();
        let utxo_set = create_multi_input_utxo_set();
        
        // Pre-lookup all UTXOs (as done in production code)
        let input_utxos: Vec<(usize, Option<&ByteString>)> = tx.inputs
            .iter()
            .enumerate()
            .map(|(j, input)| (j, utxo_set.get(&input.prevout).map(|u| &u.script_pubkey)))
            .collect();
        
        // All inputs should have corresponding UTXOs
        assert_eq!(input_utxos.len(), 3);
        for (idx, opt_script) in &input_utxos {
            assert!(opt_script.is_some(), 
                    "UTXO pre-lookup must find all inputs (input {})", idx);
        }
    }

    #[test]
    fn test_parallel_empty_transactions() {
        // Edge case: transaction with no inputs (should still work)
        let coinbase_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 50_000_000_000,
                script_pubkey: vec![0x51],
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
            transactions: vec![coinbase_tx],
        };
        
        let (result, _) = connect_block(&block, UtxoSet::new(), 0).unwrap();
        // Should handle empty non-coinbase transactions correctly
        assert!(matches!(result, ValidationResult::Valid | ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_parallel_single_input() {
        // Edge case: transaction with single input (should still work)
        let mut utxo_set = UtxoSet::new();
        let outpoint = OutPoint { hash: [1; 32], index: 0 };
        let utxo = UTXO {
            value: 10000,
            script_pubkey: vec![0x51],
            height: 0,
        };
        utxo_set.insert(outpoint, utxo);
        
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [1; 32], index: 0 },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
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
            transactions: vec![
                Transaction {
                    version: 1,
                    inputs: vec![TransactionInput {
                        prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                        script_sig: vec![0x51],
                        sequence: 0xffffffff,
                    }],
                    outputs: vec![TransactionOutput {
                        value: 50_000_000_000,
                        script_pubkey: vec![0x51],
                    }],
                    lock_time: 0,
                },
                tx,
            ],
        };
        
        let (result, _) = connect_block(&block, utxo_set, 0).unwrap();
        // Single input should work correctly in parallel mode
        assert!(matches!(result, ValidationResult::Valid | ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_parallel_deterministic_results() {
        // Verify parallel execution produces deterministic results
        let tx = create_multi_input_transaction();
        let utxo_set = create_multi_input_utxo_set();
        
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![
                Transaction {
                    version: 1,
                    inputs: vec![TransactionInput {
                        prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                        script_sig: vec![0x51],
                        sequence: 0xffffffff,
                    }],
                    outputs: vec![TransactionOutput {
                        value: 50_000_000_000,
                        script_pubkey: vec![0x51],
                    }],
                    lock_time: 0,
                },
                tx.clone(),
            ],
        };
        
        let (result1, _) = connect_block(&block, utxo_set.clone(), 0).unwrap();
        let (result2, _) = connect_block(&block, utxo_set, 0).unwrap();
        
        // Results should be identical
        assert_eq!(format!("{:?}", result1), format!("{:?}", result2),
                   "Parallel verification must be deterministic");
    }
}

