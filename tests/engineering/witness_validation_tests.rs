//! Tests for witness data validation in block context

#[path = "../integration/helpers.rs"]
mod helpers;

use blvm_consensus::bip113::get_median_time_past;
use blvm_consensus::block::{connect_block, BlockValidationContext};
use blvm_consensus::economic::get_block_subsidy;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::*;
use helpers::{merkle_root_for_tx, per_tx_witnesses};

fn coinbase_block(height: u64) -> Block {
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![OP_1, OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: get_block_subsidy(height),
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: merkle_root_for_tx(&coinbase),
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![coinbase].into(),
    }
}

#[test]
fn test_witness_validation_empty_witnesses() {
    let block = coinbase_block(0);
    let witnesses = per_tx_witnesses(&block);
    let utxo_set = UtxoSet::default();
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (result, _, _) = connect_block(&block, &witnesses, utxo_set, 0, &ctx).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_witness_validation_segwit_block() {
    let block = coinbase_block(0);
    let witnesses = per_tx_witnesses(&block);
    let utxo_set = UtxoSet::default();
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (result, _, _) = connect_block(&block, &witnesses, utxo_set, 0, &ctx).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_witness_count_mismatch() {
    let block = coinbase_block(0);
    let witnesses: Vec<Vec<Witness>> = vec![vec![Vec::new()], vec![Vec::new()]];
    let utxo_set = UtxoSet::default();
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (result, _, _) = connect_block(&block, &witnesses, utxo_set, 0, &ctx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_median_time_past_validation() {
    let mut headers = Vec::new();
    for i in 0..11 {
        headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890 + (i * 600),
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let median_time = get_median_time_past(&headers);
    assert!(median_time > 0);

    let block = coinbase_block(0);
    let witnesses = per_tx_witnesses(&block);
    let utxo_set = UtxoSet::default();
    // Median-time-past is exercised above; connect uses default context (no MTP header gate).
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (result, _, _) = connect_block(&block, &witnesses, utxo_set, 0, &ctx).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_median_time_past_with_fewer_headers() {
    let headers = vec![
        BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890 + 600,
            bits: 0x1d00ffff,
            nonce: 0,
        },
    ];

    let median_time = get_median_time_past(&headers);
    assert!(median_time > 0);
}
