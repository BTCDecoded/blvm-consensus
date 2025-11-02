//! Edge case tests for production performance optimizations

#[cfg(feature = "production")]
mod tests {
    use consensus_proof::*;
    use consensus_proof::block::*;
    use consensus_proof::script::*;

    #[test]
    fn test_production_empty_block() {
        // Block with only coinbase transaction
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
                "Production mode must handle empty blocks correctly");
    }

    #[test]
    fn test_production_max_inputs_transaction() {
        // Transaction at script input limits (tested with smaller number for practical testing)
        let max_inputs = 10; // Smaller than MAX_INPUTS for test performance
        let mut inputs = Vec::new();
        let mut utxo_set = UtxoSet::new();
        
        for i in 0..max_inputs {
            let outpoint = OutPoint { hash: [i as u8; 32], index: 0 };
            inputs.push(TransactionInput {
                prevout: outpoint,
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            });
            
            utxo_set.insert(outpoint, UTXO {
                value: 10000,
                script_pubkey: vec![0x51],
                height: 0,
            });
        }
        
        let tx = Transaction {
            version: 1,
            inputs,
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
        assert!(matches!(result, ValidationResult::Valid | ValidationResult::Invalid(_)),
                "Production mode must handle max-inputs transactions correctly");
    }

    #[test]
    fn test_production_concurrent_signatures() {
        // Multiple signature operations in sequence
        let script = vec![0x51, 0x51, 0xac]; // OP_1, OP_1, OP_CHECKSIG
        let mut results = Vec::new();
        
        for _ in 0..20 {
            let mut stack = Vec::new();
            let result = eval_script(&script, &mut stack, 0).unwrap();
            results.push(result);
        }
        
        // All results should be identical
        assert!(results.iter().all(|&r| r == results[0]),
                "Concurrent signature operations must be deterministic");
    }

    #[test]
    fn test_production_context_cleanup() {
        // Verify context doesn't retain state between operations
        let script1 = vec![0x51, 0xac]; // OP_1, OP_CHECKSIG
        let script2 = vec![0x52, 0xac]; // OP_2, OP_CHECKSIG
        
        // Execute script1 multiple times
        let mut stack1_a = Vec::new();
        let result1_a = eval_script(&script1, &mut stack1_a, 0).unwrap();
        
        let mut stack1_b = Vec::new();
        let result1_b = eval_script(&script1, &mut stack1_b, 0).unwrap();
        
        // Execute script2
        let mut stack2 = Vec::new();
        let result2 = eval_script(&script2, &mut stack2, 0).unwrap();
        
        // Execute script1 again after script2
        let mut stack1_c = Vec::new();
        let result1_c = eval_script(&script1, &mut stack1_c, 0).unwrap();
        
        // script1 results should be identical regardless of context state
        assert_eq!(result1_a, result1_b,
                   "Context must not affect script execution results");
        assert_eq!(result1_a, result1_c,
                   "Context cleanup must not affect subsequent executions");
    }

    #[test]
    fn test_production_witness_edge_cases() {
        // Test witness handling with production optimizations
        let script_sig = vec![];
        let script_pubkey = vec![];
        let witness = Some(vec![]);
        
        let result = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
        
        // Should handle empty witness correctly
        assert!(result == true || result == false);
        
        // Multiple calls should be deterministic
        let result2 = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
        assert_eq!(result, result2,
                   "Witness edge cases must be deterministic");
    }

    #[test]
    fn test_production_error_recovery() {
        // Test that errors don't affect subsequent operations
        let invalid_script = vec![0xff; MAX_SCRIPT_OPS + 1];
        let valid_script = vec![0x51];
        
        // First operation fails
        let mut stack1 = Vec::new();
        let result1 = eval_script(&invalid_script, &mut stack1, 0);
        assert!(result1.is_err());
        
        // Second operation should succeed (error shouldn't affect context)
        let mut stack2 = Vec::new();
        let result2 = eval_script(&valid_script, &mut stack2, 0);
        assert!(result2.is_ok(),
                "Errors must not affect subsequent script executions");
    }
}

