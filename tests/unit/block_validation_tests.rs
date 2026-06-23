//! connect_block / apply_transaction unit vectors (REV-TC-01 strict asserts).

use blvm_consensus::block::{self, BlockValidationContext};
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::types::{Network, ValidationResult};
use blvm_consensus::*;

fn empty_block(timestamp: u64) -> Block {
    Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp,
            bits: 0x0300ffff,
            nonce: 0,
        },
        transactions: vec![].into(),
    }
}

fn naive_coinbase_block() -> Block {
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1_231_006_505,
            bits: 0x0300ffff,
            nonce: 0,
        },
        transactions: vec![coinbase].into(),
    }
}

fn witnesses_for(block: &Block) -> Vec<Vec<segwit::Witness>> {
    block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect()
}

fn connect_mainnet(
    block: &Block,
    utxo: UtxoSet,
    height: u64,
) -> (ValidationResult, UtxoSet, reorganization::BlockUndoLog) {
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    block::connect_block(block, &witnesses_for(block), utxo, height, &ctx).unwrap()
}

#[test]
fn test_connect_block_empty_transactions() {
    let block = empty_block(0);
    let (result, _, _) = connect_mainnet(&block, UtxoSet::default(), 1);
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("no transactions")),
        "empty block must not connect: {result:?}"
    );
}

#[test]
fn test_connect_block_invalid_timestamp() {
    let block = empty_block(0);
    let (result, _, _) = connect_mainnet(&block, UtxoSet::default(), 1);
    assert!(
        matches!(result, ValidationResult::Invalid(_)),
        "empty block with timestamp 0 must not connect at height 1: {result:?}"
    );
}

#[test]
fn test_connect_block_naive_coinbase_rejected() {
    let block = naive_coinbase_block();
    let (result, _, _) = connect_mainnet(&block, UtxoSet::default(), 1);
    assert!(
        matches!(result, ValidationResult::Invalid(_)),
        "coinbase without BIP34/merkle/version rules must not connect: {result:?}"
    );
}

#[test]
fn test_apply_transaction_coinbase() {
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = block::apply_transaction(&coinbase_tx, UtxoSet::default(), 1);
    assert!(result.is_ok(), "coinbase apply should succeed: {result:?}");
    let (new_utxo, undo) = result.unwrap();
    assert!(!new_utxo.is_empty(), "coinbase must create UTXOs");
    assert!(!undo.is_empty(), "coinbase apply must emit undo entries");
}
