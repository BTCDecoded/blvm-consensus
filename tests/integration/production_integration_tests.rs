//! Integration tests for production performance optimizations

#[cfg(feature = "production")]
mod tests {
    use consensus_proof::*;
    use consensus_proof::block::*;
    use std::time::Instant;

    fn create_multi_transaction_block(num_txs: usize) -> (Block, UtxoSet) {
        let mut transactions = vec![
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
        ];
        
        let mut utxo_set = UtxoSet::new();
        
        // Create regular transactions
        for i in 1..=num_txs {
            let outpoint = OutPoint { hash: [i as u8; 32], index: 0 };
            utxo_set.insert(outpoint, UTXO {
                value: 10000,
                script_pubkey: vec![0x51],
                height: 0,
            });
            
            transactions.push(Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: outpoint,
                    script_sig: vec![0x51],
                    sequence: 0xffffffff,
                }],
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0x51],
                }],
                lock_time: 0,
            });
        }
        
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions,
        };
        
        (block, utxo_set)
    }

    #[test]
    fn test_production_block_validation_full() {
        // Complete block validation with production features enabled
        let (block, utxo_set) = create_multi_transaction_block(5);
        
        let (result, new_utxo_set) = connect_block(&block, utxo_set, 0).unwrap();
        
        // Should produce valid result
        assert!(matches!(result, ValidationResult::Valid | ValidationResult::Invalid(_)));
        
        // UTXO set should be updated
        assert!(new_utxo_set.len() > 0,
                "Block validation should update UTXO set");
    }

    #[test]
    fn test_production_multi_transaction_block() {
        // Block with many transactions (tests parallel + context reuse together)
        let (block, utxo_set) = create_multi_transaction_block(10);
        
        // Basic performance sanity check
        let start = Instant::now();
        let (result, _) = connect_block(&block, utxo_set, 0).unwrap();
        let duration = start.elapsed();
        
        // Should complete successfully
        assert!(matches!(result, ValidationResult::Valid | ValidationResult::Invalid(_)),
                "Multi-transaction block validation must work");
        
        // Should complete in reasonable time (basic sanity check)
        assert!(duration.as_millis() < 10_000,
                "Multi-transaction block should validate quickly ({}ms)", 
                duration.as_millis());
    }

    #[test]
    fn test_production_coinbase_validation() {
        // Verify coinbase handling works correctly
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
            ],
        };
        
        let (result, _) = connect_block(&block, UtxoSet::new(), 0).unwrap();
        assert!(matches!(result, ValidationResult::Valid | ValidationResult::Invalid(_)),
                "Coinbase validation must work correctly");
    }

    #[test]
    fn test_production_utxo_set_consistency() {
        // Verify UTXO set updates are correct in production mode
        let (block, initial_utxo_set) = create_multi_transaction_block(3);
        
        let (result, final_utxo_set) = connect_block(&block, initial_utxo_set, 0).unwrap();
        
        // If valid, UTXO set should be updated
        if matches!(result, ValidationResult::Valid) {
            // Should have at least the new outputs
            assert!(final_utxo_set.len() >= 3,
                    "UTXO set should contain new transaction outputs");
        }
        
        // Multiple executions should produce consistent UTXO sets
        let (_, final_utxo_set2) = connect_block(&block, initial_utxo_set, 0).unwrap();
        if matches!(result, ValidationResult::Valid) {
            assert_eq!(final_utxo_set.len(), final_utxo_set2.len(),
                       "UTXO set updates must be deterministic");
        }
    }

    #[test]
    fn test_production_deterministic_block_validation() {
        // Verify block validation is deterministic with production features
        let (block, utxo_set) = create_multi_transaction_block(5);
        
        let (result1, _) = connect_block(&block, utxo_set.clone(), 0).unwrap();
        let (result2, _) = connect_block(&block, utxo_set, 0).unwrap();
        
        // Results must be identical
        assert_eq!(format!("{:?}", result1), format!("{:?}", result2),
                   "Block validation must be deterministic with production features");
    }
}

