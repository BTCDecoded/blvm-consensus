//! Property tests for block validation edge cases

use blvm_consensus::ConsensusProof;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::types::*;
use proptest::prelude::*;

fn make_coinbase() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffffu32,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    }
}

fn per_tx_witnesses(block: &Block) -> Vec<Vec<blvm_consensus::segwit::Witness>> {
    block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect()
}

proptest! {
    #[test]
    fn prop_block_max_transactions(tx_count in 1..10usize) {
        let consensus = ConsensusProof::new();
        let mut txs: Vec<Transaction> = vec![make_coinbase()];
        for i in 1..tx_count {
            txs.push(Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [i as u8; 32], index: 0u32 },
                    script_sig: vec![OP_1],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![OP_1],
                }].into(),
                lock_time: 0,
            });
        }
        let block = Block {
            header: BlockHeader {
                version: 1i64,
                prev_block_hash: [0; 32],
                merkle_root: [1; 32],
                timestamp: 1234567890,
                bits: 0x1d00ffff_u64,
                nonce: 0,
            },
            transactions: txs.into_boxed_slice(),
        };
        let utxo_set = UtxoSet::default();
        let witnesses = per_tx_witnesses(&block);
        let result = consensus.validate_block_with_time_context(
            &block,
            &witnesses,
            utxo_set,
            0,
            None,
            blvm_consensus::types::Network::Mainnet,
        );
        prop_assert!(result.is_ok());
        if let Ok((validation, _)) = result {
            prop_assert!(
                matches!(validation, ValidationResult::Invalid(_)),
                "synthetic block with bad merkle / missing UTXOs must not validate: {validation:?}"
            );
        }
    }
}

#[test]
fn block_empty_transactions() {
    let block = Block {
        header: BlockHeader {
            version: 1i64,
            prev_block_hash: [0; 32],
            merkle_root: [1; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff_u64,
            nonce: 0,
        },
        transactions: vec![].into_boxed_slice(),
    };
    let consensus = ConsensusProof::new();
    let utxo_set = UtxoSet::default();
    let witnesses: Vec<Vec<blvm_consensus::segwit::Witness>> = vec![];
    let result = consensus.validate_block_with_time_context(
        &block,
        &witnesses,
        utxo_set,
        0,
        None,
        blvm_consensus::types::Network::Mainnet,
    );
    let (validation_result, _) = result.expect("empty block returns a validation result");
    assert!(
        matches!(validation_result, ValidationResult::Invalid(_)),
        "a block with no transactions must be Invalid, got {validation_result:?}"
    );
}

#[test]
fn block_coinbase_only() {
    let coinbase = make_coinbase();
    let block = Block {
        header: BlockHeader {
            version: 1i64,
            prev_block_hash: [0; 32],
            merkle_root: [1; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff_u64,
            nonce: 0,
        },
        transactions: vec![coinbase].into_boxed_slice(),
    };
    let consensus = ConsensusProof::new();
    let utxo_set = UtxoSet::default();
    let witnesses = per_tx_witnesses(&block);
    let result = consensus.validate_block_with_time_context(
        &block,
        &witnesses,
        utxo_set,
        0,
        None,
        blvm_consensus::types::Network::Mainnet,
    );
    let (validation, _) = result.expect("coinbase-only block should return validation result");
    assert!(
        matches!(validation, ValidationResult::Invalid(_)),
        "naive genesis-height coinbase block must not validate as accepted: {validation:?}"
    );
}

proptest! {
    #[test]
    fn prop_block_height_subsidy(height in 0u64..1000000u64) {
        let consensus = ConsensusProof::new();
        let subsidy = consensus.get_block_subsidy(height);
        prop_assert!(subsidy >= 0);
        use blvm_consensus::orange_paper_constants::{C, H};
        let initial_subsidy = 50 * C;
        prop_assert!(subsidy <= initial_subsidy as i64);
        if height > H {
            let earlier_subsidy = consensus.get_block_subsidy(height - H);
            prop_assert!(earlier_subsidy >= subsidy || subsidy == 0);
        }
    }
}
