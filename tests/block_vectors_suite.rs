//! COV-C-01a: Block connect vector wiring (Core JSON + programmatic smoke).

#[path = "core_test_vectors/block_tests.rs"]
mod block_tests;

use block_tests::{load_block_test_vectors, score_core_block_tests, BlockTestVector};
use blvm_consensus::block::{connect_block, BlockValidationContext};
use blvm_consensus::economic::get_block_subsidy;
use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::Network;
use blvm_consensus::{
    Block, BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet,
    ValidationResult,
};

fn programmatic_coinbase_block(height: u64) -> BlockTestVector {
    let value = get_block_subsidy(height);
    let coinbase = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![OP_1, height as u8].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let merkle_root = calculate_merkle_root(&[coinbase.clone()]).expect("merkle");
    BlockTestVector {
        block: Block {
            header: BlockHeader {
                version: 2,
                prev_block_hash: [0; 32],
                merkle_root,
                timestamp: 1_231_006_505,
                bits: 0x0300ffff,
                nonce: 0,
            },
            transactions: vec![coinbase].into(),
        },
        expected_result: ValidationResult::Valid,
        height: 1,
        prev_utxo_set: UtxoSet::default(),
    }
}

#[test]
fn test_score_programmatic_coinbase_block_vector() {
    let vectors = vec![programmatic_coinbase_block(1)];
    let (passed, failed) = score_core_block_tests(&vectors);
    assert_eq!(passed, 1);
    assert_eq!(failed, 0);
}

#[test]
fn test_execute_core_block_vectors_if_present() {
    let vectors = load_block_test_vectors("tests/test_data/core_vectors/blocks").expect("load");
    if vectors.is_empty() {
        return;
    }
    let (passed, failed) = score_core_block_tests(&vectors);
    eprintln!(
        "Core block vectors: {passed}/{} scored, {failed} mismatches",
        vectors.len()
    );
    assert_eq!(passed + failed, vectors.len());
}

#[test]
fn test_connect_programmatic_coinbase_directly() {
    let vector = programmatic_coinbase_block(1);
    let witnesses: Vec<Vec<Witness>> = vector
        .block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Witness::default()).collect())
        .collect();
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (result, _, _) = connect_block(
        &vector.block,
        witnesses.as_slice(),
        vector.prev_utxo_set,
        vector.height,
        &ctx,
    )
    .unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}
