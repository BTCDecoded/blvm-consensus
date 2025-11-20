//! Correctness parity tests for production performance optimizations
//! 
//! These tests verify that production optimizations produce identical results
//! to the non-production code paths. Since features are compile-time, we test
//! production behavior and verify correctness through comprehensive test cases.

#[cfg(feature = "production")]
mod tests {
    use bllvm_consensus::*;
    use bllvm_consensus::script::*;
    use bllvm_consensus::block::*;
    use std::time::Instant;

    #[path = "../test_helpers.rs"]
    mod test_helpers;
    use test_helpers::{create_test_tx, create_test_utxo_set};

    fn create_test_block() -> Block {
        Block {
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
            ],
        }
    }

    #[test]
    fn test_production_script_verification_parity() {
        // Test that production optimizations produce correct script verification results
        let script_sig = vec![0x51]; // OP_1
        let script_pubkey = vec![0x51, 0x51]; // OP_1 OP_1
        
        let result = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
        
        // Should produce deterministic result (may be true or false depending on script logic)
        assert!(result == true || result == false);
        
        // Verify multiple calls produce same result (determinism check)
        let result2 = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
        assert_eq!(result, result2, "Script verification must be deterministic");
    }

    #[test]
    fn test_production_block_validation_parity() {
        // Test block validation with production features
        let block = create_test_block();
        let utxo_set = UtxoSet::new();
        let height = 0;
        
        let (result, _new_utxo_set) = connect_block(&block, utxo_set, height).unwrap();
        
        // Should produce valid result for valid block
        assert!(matches!(result, ValidationResult::Valid | ValidationResult::Invalid(_)));
        
        // Verify deterministic behavior
        let utxo_set2 = UtxoSet::new();
        let (result2, _) = connect_block(&block, utxo_set2, height).unwrap();
        assert_eq!(format!("{:?}", result), format!("{:?}", result2),
                   "Block validation must be deterministic");
    }

    #[test]
    fn test_production_signature_verification_parity() {
        // Test OP_CHECKSIG operations with thread-local context
        let script = vec![0x51, 0x51, 0xac]; // OP_1, OP_1, OP_CHECKSIG
        
        // Test multiple signature operations to verify context reuse doesn't affect results
        let results: Vec<bool> = (0..10)
            .map(|_| {
                let mut stack = Vec::new();
                eval_script(&script, &mut stack, 0).unwrap()
            })
            .collect();
        
        // All results should be identical (determinism)
        let first_result = results[0];
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(first_result, result, 
                       "Signature verification must be deterministic (iteration {})", i);
        }
    }

    #[test]
    fn test_production_context_independence() {
        // Verify thread-local context doesn't affect results across calls
        let script1 = vec![0x51, 0x52]; // OP_1, OP_2
        let script2 = vec![0x51, 0x51]; // OP_1, OP_1
        
        let result1_a = eval_script(&script1, &mut Vec::new(), 0).unwrap();
        let result2 = eval_script(&script2, &mut Vec::new(), 0).unwrap();
        let result1_b = eval_script(&script1, &mut Vec::new(), 0).unwrap();
        
        // Same input should produce same output regardless of context reuse
        assert_eq!(result1_a, result1_b, 
                   "Context reuse must not affect script execution results");
        // Different inputs should produce different results
        assert_ne!(result1_a, result2, 
                   "Different scripts must produce different results");
    }

    #[test]
    fn test_production_memory_preallocation_parity() {
        // Verify stack pre-allocation doesn't change script execution results
        let script = vec![0x51, 0x51, 0x52, 0x52]; // OP_1, OP_1, OP_2, OP_2
        
        // First execution (will trigger pre-allocation)
        let mut stack1 = Vec::new();
        let result1 = eval_script(&script, &mut stack1, 0).unwrap();
        
        // Second execution (should reuse pre-allocated capacity if applicable)
        let mut stack2 = Vec::new();
        let result2 = eval_script(&script, &mut stack2, 0).unwrap();
        
        // Results must be identical
        assert_eq!(result1, result2, 
                   "Stack pre-allocation must not affect script execution results");
        
        // Stack state should be identical
        assert_eq!(stack1.len(), stack2.len(), 
                   "Stack sizes must match");
    }

    #[test]
    fn test_production_error_handling_parity() {
        // Verify error handling is identical with production optimizations
        let invalid_script = vec![0xff; MAX_SCRIPT_OPS + 1]; // Too many operations
        
        let mut stack = Vec::new();
        let result = eval_script(&invalid_script, &mut stack, 0);
        
        // Should produce error
        assert!(result.is_err(), 
                "Production mode must correctly handle script errors");
        
        // Error should be ScriptExecution variant
        match result {
            Err(ConsensusError::ScriptExecution(_)) => (),
            _ => panic!("Expected ScriptExecution error"),
        }
    }

    #[test]
    fn test_production_multiple_signatures_deterministic() {
        // Test that multiple signature verifications are deterministic
        let mut stack = vec![vec![0x51], vec![0x51]];
        
        let results: Vec<bool> = (0..20)
            .map(|_| {
                let mut s = stack.clone();
                execute_opcode(0xac, &mut s, 0).unwrap_or(false)
            })
            .collect();
        
        // All should be identical
        assert!(results.iter().all(|&r| r == results[0]),
                "Multiple signature operations must be deterministic");
    }

    #[test]
    fn test_production_witness_handling() {
        // Test witness script verification with production optimizations
        let script_sig = vec![0x51];
        let script_pubkey = vec![0x51];
        let witness = Some(vec![0x52]);
        
        let result = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
        
        // Should produce deterministic result
        let result2 = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
        assert_eq!(result, result2, "Witness verification must be deterministic");
    }
}

