//! Integration tests for mempool and mining functions

#[path = "../test_helpers.rs"]
mod test_helpers;

use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::transaction::is_coinbase;
use blvm_consensus::types::*;
use blvm_consensus::*;
use test_helpers::{create_invalid_transaction, create_test_tx};

fn encode_bip34_height(height: u64) -> Vec<u8> {
    if height == 0 {
        return vec![0x00, 0xff];
    }
    let mut height_bytes = Vec::new();
    let mut n = height;
    while n > 0 {
        height_bytes.push((n & 0xff) as u8);
        n >>= 8;
    }
    if height_bytes.last().is_some_and(|&b| b & 0x80 != 0) {
        height_bytes.push(0x00);
    }
    let mut script_sig = Vec::with_capacity(1 + height_bytes.len() + 1);
    script_sig.push(height_bytes.len() as u8);
    script_sig.extend_from_slice(&height_bytes);
    if script_sig.len() < 2 {
        script_sig.push(0xff);
    }
    script_sig
}

fn prev_header() -> BlockHeader {
    BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1_231_006_505,
        bits: 0x207fffff,
        nonce: 0,
    }
}

fn fund_utxo_set() -> UtxoSet {
    let mut utxo_set = UtxoSet::default();
    for (hash_byte, value) in [([1u8; 32], 10_000_i64), ([2u8; 32], 10_000_i64)] {
        utxo_set.insert(
            OutPoint {
                hash: hash_byte,
                index: 0,
            },
            std::sync::Arc::new(UTXO {
                value,
                script_pubkey: vec![OP_1].into(),
                height: 99,
                is_coinbase: false,
            }),
        );
    }
    utxo_set
}

#[test]
fn test_mempool_to_block_integration() {
    let consensus = ConsensusProof::new();
    const HEIGHT: u64 = 100;

    let tx1 = create_test_tx(1000, None, Some([1; 32]), Some(0));
    let tx2 = create_test_tx(2000, None, Some([2; 32]), Some(0));
    let utxo_set = fund_utxo_set();
    let mempool = mempool::Mempool::new();

    assert_eq!(
        consensus
            .accept_to_memory_pool(&tx1, &utxo_set, &mempool, HEIGHT, None, Network::Mainnet)
            .unwrap(),
        mempool::MempoolResult::Accepted
    );
    assert_eq!(
        consensus
            .accept_to_memory_pool(&tx2, &utxo_set, &mempool, HEIGHT, None, Network::Mainnet)
            .unwrap(),
        mempool::MempoolResult::Accepted
    );

    let header = prev_header();
    let prev_headers = vec![header.clone(), header.clone()];
    let block = consensus
        .create_new_block(
            &utxo_set,
            &[tx1, tx2],
            HEIGHT,
            &header,
            &prev_headers,
            &encode_bip34_height(HEIGHT),
            &vec![OP_1],
        )
        .unwrap();

    assert_eq!(block.transactions.len(), 3);
    assert!(is_coinbase(&block.transactions[0]));
}

#[test]
fn test_economic_mining_integration() {
    let consensus = ConsensusProof::new();
    let subsidy = consensus.get_block_subsidy(0);
    assert_eq!(subsidy, 5_000_000_000);

    let header = prev_header();
    let prev_headers = vec![header.clone(), header.clone()];
    let template = consensus
        .create_block_template(
            &UtxoSet::default(),
            &[],
            0,
            &header,
            &prev_headers,
            &encode_bip34_height(0),
            &vec![OP_1],
            Network::Mainnet,
            None,
        )
        .unwrap();

    assert_eq!(template.coinbase_tx.outputs[0].value, subsidy);
}

#[test]
fn test_script_transaction_integration() {
    let consensus = ConsensusProof::new();
    let tx = create_test_tx(1000, None, None, None);

    assert_eq!(
        consensus.validate_transaction(&tx).unwrap(),
        ValidationResult::Valid
    );

    let script_result = consensus
        .verify_script(
            &tx.inputs[0].script_sig,
            &tx.outputs[0].script_pubkey,
            None,
            0,
        )
        .unwrap();
    assert!(script_result, "OP_1/OP_1 must verify");
}

#[test]
fn test_pow_block_integration() {
    let consensus = ConsensusProof::new();
    const HEIGHT: u64 = 0;

    let coinbase = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: encode_bip34_height(HEIGHT),
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
    let merkle_root = calculate_merkle_root(std::slice::from_ref(&coinbase)).unwrap();
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

    let pow_result = consensus.check_proof_of_work(&block.header).unwrap();
    let again = consensus.check_proof_of_work(&block.header).unwrap();
    assert_eq!(pow_result, again, "PoW check must be deterministic");

    let witnesses: Vec<Vec<blvm_consensus::segwit::Witness>> = block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();
    let (validation, new_utxo) = consensus
        .validate_block_with_time_context(
            &block,
            &witnesses,
            UtxoSet::default(),
            HEIGHT,
            None,
            Network::Regtest,
        )
        .unwrap();
    assert_eq!(validation, ValidationResult::Valid);
    assert!(!new_utxo.is_empty());
}

#[test]
fn test_cross_system_error_handling() {
    let consensus = ConsensusProof::new();
    let invalid_tx = create_invalid_transaction();
    let utxo_set = UtxoSet::default();
    let mempool = mempool::Mempool::new();

    let mempool_result = consensus
        .accept_to_memory_pool(
            &invalid_tx,
            &utxo_set,
            &mempool,
            100,
            None,
            Network::Mainnet,
        )
        .unwrap();
    assert!(
        matches!(mempool_result, mempool::MempoolResult::Rejected(_)),
        "empty-input transaction must be rejected"
    );

    let header = prev_header();
    let block = consensus
        .create_new_block(
            &utxo_set,
            &[invalid_tx],
            100,
            &header,
            &[header.clone(), header.clone()],
            &encode_bip34_height(100),
            &vec![OP_1],
        )
        .unwrap();
    assert_eq!(block.transactions.len(), 1);
    assert!(is_coinbase(&block.transactions[0]));
}

#[test]
fn test_performance_integration() {
    let consensus = ConsensusProof::new();
    const HEIGHT: u64 = 100;
    let mut utxo_set = UtxoSet::default();
    let mut mempool_txs = Vec::new();

    for i in 0..10 {
        utxo_set.insert(
            OutPoint {
                hash: [i as u8; 32],
                index: 0,
            },
            std::sync::Arc::new(UTXO {
                value: 10_000,
                script_pubkey: vec![OP_1].into(),
                height: HEIGHT - 1,
                is_coinbase: false,
            }),
        );
        let mut tx = create_test_tx(1000, None, None, None);
        tx.inputs[0].prevout = OutPoint {
            hash: [i as u8; 32],
            index: 0,
        };
        mempool_txs.push(tx);
    }

    for tx in &mempool_txs {
        assert_eq!(
            consensus.validate_transaction(tx).unwrap(),
            ValidationResult::Valid
        );
    }

    let header = prev_header();
    let block = consensus
        .create_new_block(
            &utxo_set,
            &mempool_txs,
            HEIGHT,
            &header,
            &[header.clone(), header.clone()],
            &encode_bip34_height(HEIGHT),
            &vec![OP_1],
        )
        .unwrap();

    assert_eq!(block.transactions.len(), 11);
    assert!(is_coinbase(&block.transactions[0]));
}
