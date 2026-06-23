//! Edge case tests for production performance optimizations

#[cfg(feature = "production")]
mod tests {
    use blvm_consensus::block::*;
    use blvm_consensus::error::ScriptErrorCode;
    use blvm_consensus::opcodes::{OP_1, OP_2, OP_CHECKSIG, OP_NOP};
    use blvm_consensus::script::{SigVersion, *};
    use blvm_consensus::types::Network;
    use blvm_consensus::*;

    fn regtest_coinbase(height: u64) -> Transaction {
        let mut script_sig = vec![0x01];
        script_sig.push(height as u8);
        Transaction {
            version: 2,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig,
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5_000_000_000,
                script_pubkey: vec![OP_1].into(),
            }]
            .into(),
            lock_time: 0,
        }
    }

    fn witnesses_for_block(block: &Block) -> Vec<Vec<segwit::Witness>> {
        block
            .transactions
            .iter()
            .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
            .collect()
    }

    fn connect_regtest(
        block: &Block,
        utxo_set: UtxoSet,
        height: u64,
    ) -> (ValidationResult, UtxoSet, reorganization::BlockUndoLog) {
        let witnesses = witnesses_for_block(block);
        let ctx = BlockValidationContext::for_network(Network::Regtest);
        connect_block(block, witnesses.as_slice(), utxo_set, height, &ctx).unwrap()
    }

    fn eval_stack(script: &[u8]) -> Vec<StackElement> {
        let mut stack = Vec::new();
        eval_script(script, &mut stack, 0, SigVersion::Base).unwrap();
        stack
    }

    #[test]
    fn test_production_empty_block() {
        let coinbase = regtest_coinbase(0);
        let merkle_root =
            mining::calculate_merkle_root(std::slice::from_ref(&coinbase)).expect("merkle root");
        let block = Block {
            header: BlockHeader {
                version: 4,
                prev_block_hash: [0; 32],
                merkle_root,
                timestamp: 1_231_006_505,
                bits: 0x0300ffff,
                nonce: 0,
            },
            transactions: vec![coinbase].into(),
        };

        let (result, _, _) = connect_regtest(&block, UtxoSet::default(), 0);
        assert_eq!(result, ValidationResult::Valid);
    }

    #[test]
    fn test_production_max_inputs_transaction() {
        let max_inputs = 10;
        let mut inputs = Vec::new();
        let mut utxo_set = UtxoSet::default();

        for i in 0..max_inputs {
            let outpoint = OutPoint {
                hash: [i as u8; 32].into(),
                index: 0,
            };
            inputs.push(TransactionInput {
                prevout: outpoint,
                script_sig: vec![OP_1].into(),
                sequence: 0xffffffff,
            });
            utxo_set.insert(
                outpoint,
                std::sync::Arc::new(UTXO {
                    value: 1_000_000,
                    script_pubkey: vec![OP_1].into(),
                    height: 0,
                    is_coinbase: false,
                }),
            );
        }

        let tx = Transaction {
            version: 2,
            inputs: inputs.into(),
            outputs: vec![TransactionOutput {
                value: 900_000,
                script_pubkey: vec![OP_1].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let coinbase = regtest_coinbase(0);
        let transactions = vec![coinbase, tx];
        let merkle_root = mining::calculate_merkle_root(&transactions).expect("merkle root");
        let block = Block {
            header: BlockHeader {
                version: 4,
                prev_block_hash: [0; 32],
                merkle_root,
                timestamp: 1_231_006_505,
                bits: 0x0300ffff,
                nonce: 0,
            },
            transactions: transactions.into(),
        };

        let (result, _, _) = connect_regtest(&block, utxo_set, 0);
        assert_eq!(result, ValidationResult::Valid);
    }

    #[test]
    fn test_production_concurrent_signatures() {
        let script = vec![OP_1, OP_1, OP_CHECKSIG];
        let results: Vec<bool> = (0..20)
            .map(|_| {
                let mut stack = Vec::new();
                eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap()
            })
            .collect();

        assert!(
            results.iter().all(|&r| r == results[0]),
            "Concurrent signature operations must be deterministic"
        );
    }

    #[test]
    fn test_production_context_cleanup() {
        let script1 = vec![OP_1, OP_2];
        let script2 = vec![OP_1, OP_1];

        let stack1_a = eval_stack(&script1);
        let stack1_b = eval_stack(&script1);
        let stack2 = eval_stack(&script2);
        let stack1_c = eval_stack(&script1);

        assert_eq!(stack1_a, stack1_b);
        assert_eq!(stack1_a, stack1_c);
        assert_ne!(stack1_a, stack2);
    }

    #[test]
    fn test_production_witness_edge_cases() {
        let script_sig = vec![];
        let script_pubkey = vec![];
        let witness = Some(vec![]);

        let result = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
        let result2 = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
        assert_eq!(result, result2);
    }

    #[test]
    fn test_production_error_recovery() {
        let invalid_script = vec![OP_NOP; MAX_SCRIPT_OPS + 1];
        let valid_script = vec![OP_1];

        let mut stack1 = Vec::new();
        let result1 = eval_script(&invalid_script, &mut stack1, 0, SigVersion::Base);
        assert!(matches!(
            result1,
            Err(ConsensusError::ScriptErrorWithCode {
                code: ScriptErrorCode::OpCount,
                ..
            })
        ));

        let mut stack2 = Vec::new();
        let result2 = eval_script(&valid_script, &mut stack2, 0, SigVersion::Base);
        assert!(result2.is_ok());
    }
}
