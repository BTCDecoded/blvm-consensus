//! Correctness parity tests for production performance optimizations
//!
//! These tests verify that production optimizations produce identical results
//! to the non-production code paths. Since features are compile-time, we test
//! production behavior and verify correctness through comprehensive test cases.

#[cfg(feature = "production")]
mod tests {
    use blvm_consensus::block::*;
    use blvm_consensus::error::ScriptErrorCode;
    use blvm_consensus::opcodes::{OP_1, OP_2, OP_CHECKSIG, OP_NOP};
    use blvm_consensus::script::{SigVersion, *};
    use blvm_consensus::types::Network;
    use blvm_consensus::*;

    fn create_test_block() -> Block {
        let coinbase = Transaction {
            version: 2,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![0x01, 0x00],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5_000_000_000,
                script_pubkey: vec![OP_1].into(),
            }]
            .into(),
            lock_time: 0,
        };
        let merkle_root =
            mining::calculate_merkle_root(std::slice::from_ref(&coinbase)).expect("merkle root");
        Block {
            header: BlockHeader {
                version: 4,
                prev_block_hash: [0; 32],
                merkle_root,
                timestamp: 1_231_006_505,
                bits: 0x0300ffff,
                nonce: 0,
            },
            transactions: vec![coinbase].into(),
        }
    }

    #[test]
    fn test_production_script_verification_parity() {
        // Test that production optimizations produce correct script verification results
        let script_sig = vec![OP_1]; // OP_1
        let script_pubkey = vec![OP_1, OP_1]; // OP_1 OP_1

        let result = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
        assert!(result, "OP_1 vs OP_1 OP_1 must verify true");

        let result2 = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
        assert_eq!(result, result2, "Script verification must be deterministic");
    }

    #[test]
    fn test_production_block_validation_parity() {
        // Test block validation with production features
        let block = create_test_block();
        let utxo_set = UtxoSet::default();
        let height = 0;

        let witnesses: Vec<Vec<segwit::Witness>> = block
            .transactions
            .iter()
            .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
            .collect();
        let (result, _new_utxo_set, _undo) = {
            let ctx = block::BlockValidationContext::for_network(Network::Regtest);
            connect_block(&block, witnesses.as_slice(), utxo_set, height, &ctx)
        }
        .unwrap();

        assert_eq!(result, ValidationResult::Valid);

        // Verify deterministic behavior
        let utxo_set2 = UtxoSet::default();
        let (result2, _, _) = {
            let ctx = block::BlockValidationContext::for_network(Network::Regtest);
            connect_block(&block, witnesses.as_slice(), utxo_set2, height, &ctx)
        }
        .unwrap();
        assert_eq!(
            format!("{:?}", result),
            format!("{:?}", result2),
            "Block validation must be deterministic"
        );
    }

    #[test]
    fn test_production_signature_verification_parity() {
        // Test OP_CHECKSIG operations with thread-local context
        let script = vec![OP_1, OP_1, OP_CHECKSIG]; // OP_1, OP_1, OP_CHECKSIG

        // Test multiple signature operations to verify context reuse doesn't affect results
        let results: Vec<bool> = (0..10)
            .map(|_| {
                let mut stack = Vec::new();
                eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap()
            })
            .collect();

        // All results should be identical (determinism)
        let first_result = results[0];
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(
                first_result, result,
                "Signature verification must be deterministic (iteration {})",
                i
            );
        }
    }

    fn eval_stack(script: &[u8]) -> Vec<StackElement> {
        let mut stack = Vec::new();
        eval_script(script, &mut stack, 0, SigVersion::Base).unwrap();
        stack
    }

    #[test]
    fn test_production_context_independence() {
        // Verify thread-local context doesn't affect results across calls
        let script1 = vec![OP_1, OP_2];
        let script2 = vec![OP_1, OP_1];

        let stack1_a = eval_stack(&script1);
        let stack2 = eval_stack(&script2);
        let stack1_b = eval_stack(&script1);

        // Same input should produce same stack regardless of context reuse
        assert_eq!(
            stack1_a, stack1_b,
            "Context reuse must not affect script execution results"
        );
        // Different inputs should produce different stacks
        assert_ne!(
            stack1_a, stack2,
            "Different scripts must produce different results"
        );
    }

    #[test]
    fn test_production_memory_preallocation_parity() {
        // Verify stack pre-allocation doesn't change script execution results
        let script = vec![OP_1, OP_1, OP_2, OP_2]; // OP_1, OP_1, OP_2, OP_2

        // First execution (will trigger pre-allocation)
        let mut stack1 = Vec::new();
        let result1 = eval_script(&script, &mut stack1, 0, SigVersion::Base).unwrap();

        // Second execution (should reuse pre-allocated capacity if applicable)
        let mut stack2 = Vec::new();
        let result2 = eval_script(&script, &mut stack2, 0, SigVersion::Base).unwrap();

        // Results must be identical
        assert_eq!(
            result1, result2,
            "Stack pre-allocation must not affect script execution results"
        );

        // Stack state should be identical
        assert_eq!(stack1.len(), stack2.len(), "Stack sizes must match");
    }

    #[test]
    fn test_production_error_handling_parity() {
        // Verify error handling is identical with production optimizations
        let invalid_script = vec![OP_NOP; MAX_SCRIPT_OPS + 1];

        let mut stack = Vec::new();
        let result = eval_script(&invalid_script, &mut stack, 0, SigVersion::Base);

        // Should produce error (op count limit)
        assert!(
            result.is_err(),
            "Production mode must correctly handle script errors"
        );

        match result {
            Err(ConsensusError::ScriptErrorWithCode {
                code: ScriptErrorCode::OpCount,
                ..
            })
            | Err(ConsensusError::ScriptExecution(_)) => (),
            other => panic!("Expected op-count script error, got {other:?}"),
        }
    }

    #[test]
    fn test_production_multiple_signatures_deterministic() {
        // Test that multiple signature verifications are deterministic
        let script = vec![OP_1, OP_1, OP_CHECKSIG];

        let results: Vec<bool> = (0..20)
            .map(|_| {
                let mut stack = Vec::new();
                eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap()
            })
            .collect();

        // All should be identical
        assert!(
            results.iter().all(|&r| r == results[0]),
            "Multiple signature operations must be deterministic"
        );
    }

    #[test]
    fn test_production_witness_handling() {
        // Test witness script verification with production optimizations
        let script_sig = vec![OP_1];
        let script_pubkey = vec![OP_1];
        let witness = Some(vec![OP_2]);

        let result = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();

        // Should produce deterministic result
        let result2 = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
        assert_eq!(
            result, result2,
            "Witness verification must be deterministic"
        );
    }
}
