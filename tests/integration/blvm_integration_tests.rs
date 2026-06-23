//! Integration tests for BLVM optimizations (serialization, merkle, validation).

use blvm_consensus::opcodes::OP_1;
use blvm_consensus::{
    ConsensusProof, OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet,
    ValidationResult,
    mining::calculate_merkle_root,
    optimizations::simd_vectorization,
    serialization::{block::serialize_block_header, transaction::serialize_transaction},
    types::{Block, BlockHeader, Network},
};

fn create_test_transaction() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

fn create_test_transactions(count: usize) -> Vec<Transaction> {
    (0..count)
        .map(|i| {
            let mut tx = create_test_transaction();
            tx.inputs[0].prevout.index = i as u32;
            tx
        })
        .collect()
}

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
    let merkle_root = calculate_merkle_root(std::slice::from_ref(&coinbase)).expect("merkle root");
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

/// Regtest genesis-height coinbase block must validate through the public API.
#[test]
fn test_block_validation_correctness() {
    let block = create_test_block();
    let utxo_set = UtxoSet::default();
    let consensus = ConsensusProof::new();
    let witnesses: Vec<Vec<blvm_consensus::segwit::Witness>> = block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();

    let (validation, new_utxo) = consensus
        .validate_block_with_time_context(&block, &witnesses, utxo_set, 0, None, Network::Regtest)
        .expect("validation should return a result");

    assert_eq!(validation, ValidationResult::Valid);
    assert!(!new_utxo.is_empty(), "valid coinbase must create UTXOs");
}

#[test]
fn test_batch_hashing_integration() {
    #[cfg(feature = "production")]
    {
        let transactions = create_test_transactions(100);
        let serialized: Vec<Vec<u8>> = transactions.iter().map(serialize_transaction).collect();
        let tx_refs: Vec<&[u8]> = serialized.iter().map(|v| v.as_slice()).collect();
        let aligned_hashes = simd_vectorization::batch_double_sha256_aligned(&tx_refs);
        let regular_hashes = simd_vectorization::batch_double_sha256(&tx_refs);
        assert_eq!(aligned_hashes.len(), transactions.len());
        assert_eq!(regular_hashes.len(), transactions.len());
        for (aligned, regular) in aligned_hashes.iter().zip(regular_hashes.iter()) {
            assert_eq!(aligned.as_bytes(), regular);
        }
    }
}

#[test]
fn test_merkle_root_large_transaction_set() {
    let transactions = create_test_transactions(2000);
    let root = calculate_merkle_root(&transactions).expect("merkle root");
    assert_eq!(root.len(), 32);
    let root2 = calculate_merkle_root(&transactions).expect("merkle root again");
    assert_eq!(root, root2, "merkle root must be deterministic");
}

#[test]
fn test_serialization_round_trip() {
    use blvm_consensus::serialization::transaction::deserialize_transaction;

    let tx = create_test_transaction();
    let serialized = serialize_transaction(&tx);
    let deserialized = deserialize_transaction(&serialized).expect("deserialize");
    assert_eq!(tx.version, deserialized.version);
    assert_eq!(tx.inputs.len(), deserialized.inputs.len());
    assert_eq!(tx.outputs.len(), deserialized.outputs.len());
    assert_eq!(tx.lock_time, deserialized.lock_time);
}

#[test]
#[cfg(feature = "production")]
#[cfg(feature = "rayon")]
fn test_batch_operations_parallel() {
    use rayon::prelude::*;

    let transactions = create_test_transactions(1000);
    let serialized: Vec<Vec<u8>> = transactions.par_iter().map(serialize_transaction).collect();
    let tx_refs: Vec<&[u8]> = serialized.iter().map(|v| v.as_slice()).collect();
    let hashes = simd_vectorization::batch_double_sha256(&tx_refs);
    assert_eq!(hashes.len(), transactions.len());
}

#[test]
fn test_optimizations_edge_cases() {
    let empty_txs: Vec<Transaction> = vec![];
    assert!(calculate_merkle_root(&empty_txs).is_err());

    let single_tx = vec![create_test_transaction()];
    let root = calculate_merkle_root(&single_tx).expect("single tx merkle");
    assert_eq!(root.len(), 32);

    let mut large_tx = create_test_transaction();
    large_tx.outputs[0].script_pubkey = vec![0u8; 10000].into();
    let serialized = serialize_transaction(&large_tx);
    assert!(serialized.len() > 10000);

    let block = create_test_block();
    let wire = serialize_block_header(&block.header);
    assert_eq!(wire.len(), 80);
}
