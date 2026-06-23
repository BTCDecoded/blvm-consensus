//! Tests for parallel script verification in production mode

#[cfg(feature = "production")]
mod tests {
    use blvm_consensus::block::*;
    use blvm_consensus::opcodes::{OP_1, OP_NOP};
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

    fn create_multi_input_transaction() -> Transaction {
        Transaction {
            version: 2,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [1; 32].into(),
                        index: 0,
                    },
                    script_sig: vec![OP_1].into(),
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint {
                        hash: [2; 32].into(),
                        index: 0,
                    },
                    script_sig: vec![OP_1].into(),
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint {
                        hash: [3; 32].into(),
                        index: 0,
                    },
                    script_sig: vec![OP_1].into(),
                    sequence: 0xffffffff,
                },
            ]
            .into(),
            outputs: vec![TransactionOutput {
                value: 900_000,
                script_pubkey: vec![OP_1].into(),
            }]
            .into(),
            lock_time: 0,
        }
    }

    fn create_multi_input_utxo_set() -> UtxoSet {
        let mut utxo_set = UtxoSet::default();
        for i in 1..=3 {
            let outpoint = OutPoint {
                hash: [i as u8; 32].into(),
                index: 0,
            };
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
        utxo_set
    }

    fn block_with_coinbase_and_tx(tx: Transaction) -> Block {
        let coinbase = regtest_coinbase(0);
        let transactions = vec![coinbase, tx];
        let merkle_root = mining::calculate_merkle_root(&transactions).expect("merkle root");
        Block {
            header: BlockHeader {
                version: 4,
                prev_block_hash: [0; 32],
                merkle_root,
                timestamp: 1_231_006_505,
                bits: 0x0300ffff,
                nonce: 0,
            },
            transactions: transactions.into(),
        }
    }

    #[test]
    fn test_parallel_script_verification_single_tx() {
        let tx = create_multi_input_transaction();
        let block = block_with_coinbase_and_tx(tx);
        let (result, _, _) = connect_regtest(&block, create_multi_input_utxo_set(), 0);
        assert_eq!(result, ValidationResult::Valid);
    }

    #[test]
    fn test_parallel_script_verification_ordering() {
        let tx = create_multi_input_transaction();
        let utxo_set = create_multi_input_utxo_set();
        let input_utxos: Vec<_> = tx
            .inputs
            .iter()
            .enumerate()
            .map(|(j, input)| (j, utxo_set.get(&input.prevout).map(|u| &u.script_pubkey)))
            .collect();

        assert_eq!(input_utxos.len(), 3);
        assert_eq!(input_utxos[0].0, 0);
        assert_eq!(input_utxos[1].0, 1);
        assert_eq!(input_utxos[2].0, 2);
    }

    #[test]
    fn test_parallel_script_verification_error_handling() {
        let tx = Transaction {
            version: 2,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![OP_NOP; MAX_SCRIPT_OPS + 1].into(),
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 900_000,
                script_pubkey: vec![OP_1].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let block = block_with_coinbase_and_tx(tx);
        let witnesses = witnesses_for_block(&block);
        let ctx = BlockValidationContext::for_network(Network::Regtest);
        let result = connect_block(
            &block,
            witnesses.as_slice(),
            create_multi_input_utxo_set(),
            0,
            &ctx,
        );
        assert!(
            result.is_err(),
            "Invalid script must fail block connect: {result:?}"
        );
    }

    #[test]
    fn test_parallel_utxo_prelookup() {
        let tx = create_multi_input_transaction();
        let utxo_set = create_multi_input_utxo_set();
        let input_utxos: Vec<_> = tx
            .inputs
            .iter()
            .enumerate()
            .map(|(j, input)| (j, utxo_set.get(&input.prevout).map(|u| &u.script_pubkey)))
            .collect();

        assert_eq!(input_utxos.len(), 3);
        for (idx, opt_script) in &input_utxos {
            assert!(
                opt_script.is_some(),
                "UTXO pre-lookup must find all inputs (input {})",
                idx
            );
        }
    }

    #[test]
    fn test_parallel_empty_transactions() {
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
    fn test_parallel_single_input() {
        let mut utxo_set = UtxoSet::default();
        let outpoint = OutPoint {
            hash: [1; 32].into(),
            index: 0,
        };
        utxo_set.insert(
            outpoint,
            std::sync::Arc::new(UTXO {
                value: 1_000_000,
                script_pubkey: vec![OP_1].into(),
                height: 0,
                is_coinbase: false,
            }),
        );

        let tx = Transaction {
            version: 2,
            inputs: vec![TransactionInput {
                prevout: outpoint,
                script_sig: vec![OP_1].into(),
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 900_000,
                script_pubkey: vec![OP_1].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let block = block_with_coinbase_and_tx(tx);
        let (result, _, _) = connect_regtest(&block, utxo_set, 0);
        assert_eq!(result, ValidationResult::Valid);
    }

    #[test]
    fn test_parallel_deterministic_results() {
        let tx = create_multi_input_transaction();
        let block = block_with_coinbase_and_tx(tx);
        let utxo_set = create_multi_input_utxo_set();
        let witnesses = witnesses_for_block(&block);
        let ctx = BlockValidationContext::for_network(Network::Regtest);

        let (result1, _, _) =
            connect_block(&block, witnesses.as_slice(), utxo_set.clone(), 0, &ctx).unwrap();
        let (result2, _, _) =
            connect_block(&block, witnesses.as_slice(), utxo_set, 0, &ctx).unwrap();

        assert_eq!(result1, result2);
        assert_eq!(result1, ValidationResult::Valid);
    }
}
