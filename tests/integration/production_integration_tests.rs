//! Integration tests for production performance optimizations

#[cfg(feature = "production")]
#[path = "../test_helpers.rs"]
mod test_helpers;

#[cfg(feature = "production")]
mod tests {
    use super::test_helpers::adjusted_timeout;
    use blvm_consensus::block::*;
    use blvm_consensus::opcodes::OP_1;
    use blvm_consensus::types::Network;
    use blvm_consensus::*;
    use std::time::Instant;

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

    fn create_multi_transaction_block(num_txs: usize) -> (Block, UtxoSet) {
        let coinbase = regtest_coinbase(0);
        let mut transactions = vec![coinbase.clone()];
        let mut utxo_set = UtxoSet::default();

        for i in 1..=num_txs {
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

            transactions.push(Transaction {
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
            });
        }

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

        (block, utxo_set)
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

    #[test]
    fn test_production_block_validation_full() {
        let (block, utxo_set) = create_multi_transaction_block(5);
        let (result, new_utxo_set, _undo) = connect_regtest(&block, utxo_set, 0);

        assert_eq!(result, ValidationResult::Valid);
        assert!(
            new_utxo_set.len() > 0,
            "Block validation should update UTXO set"
        );
    }

    #[test]
    fn test_production_multi_transaction_block() {
        let (block, utxo_set) = create_multi_transaction_block(10);

        let start = Instant::now();
        let (result, _, _) = connect_regtest(&block, utxo_set, 0);
        let duration = start.elapsed();

        assert_eq!(result, ValidationResult::Valid);

        let max_duration_ms = adjusted_timeout(10_000);
        assert!(
            duration.as_millis() < max_duration_ms as u128,
            "Multi-transaction block should validate quickly ({}ms, max: {}ms)",
            duration.as_millis(),
            max_duration_ms
        );
    }

    #[test]
    fn test_production_coinbase_validation() {
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
    fn test_production_utxo_set_consistency() {
        let (block, initial_utxo_set) = create_multi_transaction_block(3);
        let (result, final_utxo_set, _) = connect_regtest(&block, initial_utxo_set.clone(), 0);

        assert_eq!(result, ValidationResult::Valid);
        assert!(
            final_utxo_set.len() >= 3,
            "UTXO set should contain new transaction outputs"
        );

        let (_, final_utxo_set2, _) = connect_regtest(&block, initial_utxo_set, 0);
        assert_eq!(
            final_utxo_set.len(),
            final_utxo_set2.len(),
            "UTXO set updates must be deterministic"
        );
    }

    #[test]
    fn test_production_deterministic_block_validation() {
        let (block, utxo_set) = create_multi_transaction_block(5);

        let (result1, _, _) = connect_regtest(&block, utxo_set.clone(), 0);
        let (result2, _, _) = connect_regtest(&block, utxo_set, 0);

        assert_eq!(result1, result2);
        assert_eq!(result1, ValidationResult::Valid);
    }
}
