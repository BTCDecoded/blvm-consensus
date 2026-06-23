//! COV-C-05a: Mempool policy coverage (finality, RBF, conflicts, block updates).

#[path = "test_helpers.rs"]
mod test_helpers;

use blvm_consensus::block::calculate_tx_id;
use blvm_consensus::constants::{LOCKTIME_THRESHOLD, MIN_RELAY_FEE};
use blvm_consensus::mempool::{
    Mempool, MempoolResult, accept_to_memory_pool, has_conflict_with_tx, is_final_tx,
    replacement_checks, signals_rbf, update_mempool_after_block,
    update_mempool_after_block_with_lookup,
};
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::opcodes::{OP_0, OP_VERIFY};
use blvm_consensus::{
    Block, BlockHeader, Network, OutPoint, Transaction, TransactionInput, TransactionOutput,
    UtxoSet,
};
use test_helpers::{create_coinbase_tx, create_rbf_tx, create_test_tx, create_test_utxo};

#[test]
fn test_is_final_tx_height_based() {
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x01; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xfffffffe,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 100,
    };
    assert!(!is_final_tx(&tx, 100, 0));
    assert!(is_final_tx(&tx, 101, 0));
}

#[test]
fn test_is_final_tx_timestamp_based() {
    let lock_time = LOCKTIME_THRESHOLD as u64 + 1_000;
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x02; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xfffffffe,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time,
    };
    assert!(!is_final_tx(&tx, 200_000, lock_time - 1));
    assert!(is_final_tx(&tx, 200_000, lock_time + 1));
}

#[test]
fn test_is_final_tx_all_inputs_final_sequence() {
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x03; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 999_999,
    };
    assert!(is_final_tx(&tx, 1, 1));
}

#[test]
fn test_signals_rbf_and_conflict() {
    let rbf = create_rbf_tx(0xfffffffe);
    let non_rbf = create_test_tx(1_000, Some(0xffffffff), None, None);
    assert!(signals_rbf(&rbf));
    assert!(!signals_rbf(&non_rbf));

    let mut conflict = create_test_tx(900, Some(0xfffffffe), None, None);
    conflict.inputs[0].prevout = rbf.inputs[0].prevout.clone();
    assert!(has_conflict_with_tx(&conflict, &rbf));
}

#[test]
fn test_replacement_checks_rejects_non_rbf_existing() {
    let (set, _) = create_test_utxo(10_000);
    let existing = create_test_tx(8_000, Some(0xffffffff), None, None);
    let replacement = create_rbf_tx(0xfffffffe);
    let pool = Mempool::new();
    assert!(!replacement_checks(&replacement, &existing, &set, &pool).unwrap());
}

#[test]
fn test_accept_to_memory_pool_rejects_missing_utxo() {
    let tx = create_test_tx(1_000, None, Some([0x99; 32]), Some(0));
    let pool = Mempool::new();
    let res = accept_to_memory_pool(
        &tx,
        None,
        &UtxoSet::default(),
        &pool,
        1,
        None,
        Network::Mainnet,
    );
    assert!(
        matches!(res, Ok(MempoolResult::Rejected(_)) | Err(_)),
        "missing UTXO should not be accepted: {res:?}"
    );
}

#[test]
fn test_replacement_checks_success() {
    let (set, _) = create_test_utxo(10_000);
    let mut existing = create_rbf_tx(0xfffffffe);
    existing.outputs[0].value = 9_000; // fee = 1000

    let mut replacement = existing.clone();
    replacement.outputs[0].value = 10_000 - 1_000 - MIN_RELAY_FEE - 1; // fee = 2001

    let pool = Mempool::new();
    assert!(
        replacement_checks(&replacement, &existing, &set, &pool).unwrap(),
        "valid RBF replacement should pass all BIP125 checks"
    );
}

#[test]
fn test_accept_to_memory_pool_accepts_valid_tx() {
    let (set, _) = create_test_utxo(10_000);
    let tx = create_test_tx(8_500, None, None, None); // fee = 1500
    let pool = Mempool::new();
    let res = accept_to_memory_pool(&tx, None, &set, &pool, 100, None, Network::Mainnet).unwrap();
    assert_eq!(res, MempoolResult::Accepted);
}

#[test]
fn test_accept_rejects_non_final_tx() {
    let (set, _) = create_test_utxo(10_000);
    let mut tx = create_test_tx(8_500, Some(0xfffffffe), None, None);
    tx.lock_time = 200;
    let pool = Mempool::new();
    let res = accept_to_memory_pool(&tx, None, &set, &pool, 100, None, Network::Mainnet).unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("not final")),
        "non-final tx should be rejected: {res:?}"
    );
}

#[test]
fn test_update_mempool_after_block_with_lookup_removes_spent_conflict() {
    let included = create_test_tx(1_000, None, None, None);
    let included_id = calculate_tx_id(&included);

    let mut conflict = create_test_tx(900, None, None, None);
    conflict.inputs[0].prevout = included.inputs[0].prevout.clone();
    let conflict_id = calculate_tx_id(&conflict);

    let mut pool = Mempool::new();
    pool.insert(included_id);
    pool.insert(conflict_id);

    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1,
            bits: 0,
            nonce: 0,
        },
        transactions: vec![included.clone()].into(),
    };

    let mut store = std::collections::HashMap::new();
    store.insert(included_id, included);
    store.insert(conflict_id, conflict);

    let removed =
        update_mempool_after_block_with_lookup(&mut pool, &block, |id| store.get(id).cloned())
            .unwrap();
    assert!(removed.contains(&included_id));
    assert!(removed.contains(&conflict_id));
    assert!(pool.is_empty());
}

#[test]
fn test_accept_rejects_duplicate_in_mempool() {
    let (set, _) = create_test_utxo(10_000);
    let tx = create_test_tx(8_500, None, None, None);
    let tx_id = calculate_tx_id(&tx);
    let mut pool = Mempool::new();
    pool.insert(tx_id);
    let res = accept_to_memory_pool(&tx, None, &set, &pool, 100, None, Network::Mainnet).unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("already")),
        "duplicate mempool entry should be rejected: {res:?}"
    );
}

#[test]
fn test_update_mempool_after_block_removes_included_tx() {
    let tx = create_test_tx(1_000, None, None, None);
    let tx_id = calculate_tx_id(&tx);
    let mut pool = Mempool::new();
    pool.insert(tx_id);

    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1,
            bits: 0,
            nonce: 0,
        },
        transactions: vec![tx].into(),
    };
    let removed = update_mempool_after_block(&mut pool, &block, &UtxoSet::default()).unwrap();
    assert!(removed.contains(&tx_id));
}

#[test]
fn test_update_mempool_after_block_removes_spent_conflict_without_lookup() {
    let included = create_test_tx(1_000, None, None, None);
    let mut conflict = create_test_tx(900, None, None, None);
    conflict.inputs[0].prevout = included.inputs[0].prevout.clone();
    let conflict_id = calculate_tx_id(&conflict);

    let mut pool = Mempool::new();
    pool.insert_transaction(&conflict);

    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1,
            bits: 0,
            nonce: 0,
        },
        transactions: vec![included].into(),
    };

    let removed = update_mempool_after_block(&mut pool, &block, &UtxoSet::default()).unwrap();
    assert!(removed.contains(&conflict_id));
    assert!(pool.is_empty());
}

#[test]
fn test_accept_rejects_insufficient_fee() {
    let (set, _) = create_test_utxo(10_000);
    let tx = create_test_tx(9_999, None, None, None);
    let pool = Mempool::new();
    let res = accept_to_memory_pool(&tx, None, &set, &pool, 100, None, Network::Mainnet).unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("mempool") || r.contains("fee")),
        "underpaid tx should be rejected: {res:?}"
    );
}

#[test]
fn test_accept_allows_unrelated_mempool_txid_matching_funding_hash() {
    let (set, _) = create_test_utxo(10_000);
    let tx = create_test_tx(8_500, None, None, None);
    let mut pool = Mempool::new();
    // Same bytes as funding txid, but not indexed as spending the outpoint (REV-C-25 fix).
    pool.insert([1; 32]);
    let res = accept_to_memory_pool(&tx, None, &set, &pool, 100, None, Network::Mainnet).unwrap();
    assert!(
        matches!(res, MempoolResult::Accepted),
        "mempool txid alone must not imply double-spend: {res:?}"
    );
}

#[test]
fn test_replacement_checks_rejects_insufficient_fee_increment() {
    let (set, _) = create_test_utxo(10_000);
    let mut existing = create_rbf_tx(0xfffffffe);
    existing.outputs[0].value = 9_000;

    let mut replacement = existing.clone();
    replacement.outputs[0].value = 9_000 - 1;

    let pool = Mempool::new();
    assert!(
        !replacement_checks(&replacement, &existing, &set, &pool).unwrap(),
        "replacement must pay MIN_RELAY_FEE more than original"
    );
}

#[test]
fn test_replacement_checks_rejects_coinbase_replacement() {
    let (set, _) = create_test_utxo(10_000);
    let existing = create_rbf_tx(0xfffffffe);
    let coinbase = create_coinbase_tx(1_000);
    let pool = Mempool::new();
    assert!(replacement_checks(&coinbase, &existing, &set, &pool).is_err());
}

#[test]
fn test_accept_rejects_empty_transaction() {
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    let pool = Mempool::new();
    let res = accept_to_memory_pool(
        &tx,
        None,
        &UtxoSet::default(),
        &pool,
        1,
        None,
        Network::Mainnet,
    )
    .unwrap();
    assert!(matches!(res, MempoolResult::Rejected(_)));
}

#[test]
fn test_accept_rejects_coinbase() {
    let pool = Mempool::new();
    let res = accept_to_memory_pool(
        &create_coinbase_tx(1_000),
        None,
        &UtxoSet::default(),
        &pool,
        1,
        None,
        Network::Mainnet,
    )
    .unwrap();
    assert!(matches!(res, MempoolResult::Rejected(ref r) if r.contains("Coinbase")));
}

#[test]
fn test_accept_rejects_duplicate_mempool_entry() {
    let (set, _) = create_test_utxo(10_000);
    let tx = create_test_tx(8_500, None, None, None);
    let tx_id = calculate_tx_id(&tx);
    let mut pool = Mempool::new();
    pool.insert(tx_id);
    let res = accept_to_memory_pool(&tx, None, &set, &pool, 100, None, Network::Mainnet).unwrap();
    assert!(matches!(res, MempoolResult::Rejected(ref r) if r.contains("already")));
}

#[test]
fn test_accept_rejects_invalid_transaction_structure() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x20; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    let pool = Mempool::new();
    let res = accept_to_memory_pool(
        &tx,
        None,
        &UtxoSet::default(),
        &pool,
        100,
        None,
        Network::Mainnet,
    )
    .unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("Invalid transaction")),
        "empty outputs should fail structure check: {res:?}"
    );
}

#[test]
fn test_accept_rejects_invalid_transaction_inputs() {
    let tx = create_test_tx(8_500, None, None, None);
    let pool = Mempool::new();
    let res = accept_to_memory_pool(
        &tx,
        None,
        &UtxoSet::default(),
        &pool,
        100,
        None,
        Network::Mainnet,
    )
    .unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("Invalid transaction inputs")),
        "missing prevout should fail input check: {res:?}"
    );
}

#[test]
fn test_accept_rejects_invalid_script() {
    use std::sync::Arc;
    let mut set = UtxoSet::default();
    set.insert(
        OutPoint {
            hash: [0x31; 32],
            index: 0,
        },
        Arc::new(blvm_consensus::UTXO {
            value: 10_000,
            script_pubkey: vec![OP_0, OP_VERIFY].into(),
            height: 1,
            is_coinbase: false,
        }),
    );
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x31; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 8_500,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let pool = Mempool::new();
    let res = accept_to_memory_pool(&tx, None, &set, &pool, 100, None, Network::Mainnet).unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("Invalid script")),
        "script verification failure should reject: {res:?}"
    );
}

#[test]
fn test_accept_rejects_zero_fee_below_mempool_minimum() {
    let (set, _) = create_test_utxo(10_000);
    let tx = create_test_tx(10_000, None, None, None);
    let pool = Mempool::new();
    let res = accept_to_memory_pool(&tx, None, &set, &pool, 100, None, Network::Mainnet).unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("Failed mempool rules")),
        "zero-fee tx should fail mempool policy: {res:?}"
    );
}

#[test]
fn test_is_standard_tx_rejects_multiple_op_return_outputs() {
    use blvm_consensus::mempool::is_standard_tx;
    use blvm_consensus::opcodes::OP_RETURN;
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x32; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: 0,
                script_pubkey: vec![OP_RETURN, 0x01, 0x00].into(),
            },
            TransactionOutput {
                value: 0,
                script_pubkey: vec![OP_RETURN, 0x01, 0x01].into(),
            },
        ]
        .into(),
        lock_time: 0,
    };
    assert!(!is_standard_tx(&tx).unwrap());
}

#[test]
fn test_accept_rejects_true_mempool_double_spend() {
    let (set, _) = create_test_utxo(10_000);
    let pool_tx = create_test_tx(9_000, None, None, None);
    let mut pool = Mempool::new();
    pool.insert_transaction(&pool_tx);

    let mut conflicting = create_test_tx(8_500, None, None, None);
    conflicting.version = 2;
    let res = accept_to_memory_pool(&conflicting, None, &set, &pool, 100, None, Network::Mainnet)
        .unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("conflicts with mempool")),
        "expected explicit mempool conflict rejection: {res:?}"
    );
}

#[test]
fn test_replacement_checks_rejects_no_conflict() {
    let (set, _) = create_test_utxo(10_000);
    let mut existing = create_rbf_tx(0xfffffffe);
    existing.outputs[0].value = 9_000;
    let mut replacement = existing.clone();
    replacement.inputs[0].prevout.hash = [0x55; 32];
    replacement.outputs[0].value = 8_500;
    let pool = Mempool::new();
    let res = replacement_checks(&replacement, &existing, &set, &pool);
    assert!(
        matches!(res, Ok(false) | Err(_)),
        "replacement without shared input must not succeed: {res:?}"
    );
}

#[test]
fn test_replacement_checks_rejects_new_unconfirmed_dependency() {
    let (set, _) = create_test_utxo(10_000);
    let mut existing = create_rbf_tx(0xfffffffe);
    existing.outputs[0].value = 9_000;

    let mut replacement = existing.clone();
    replacement.inputs.push(TransactionInput {
        prevout: OutPoint {
            hash: [0x77; 32],
            index: 0,
        },
        script_sig: vec![].into(),
        sequence: 0xfffffffe,
    });
    replacement.outputs[0].value = 8_500;

    let pool = Mempool::new();
    assert!(
        !replacement_checks(&replacement, &existing, &set, &pool).unwrap(),
        "replacement introducing new unconfirmed dependency must fail"
    );
}

#[test]
fn test_accept_rejects_empty_inputs_and_outputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    let pool = Mempool::new();
    let res = accept_to_memory_pool(
        &tx,
        None,
        &UtxoSet::default(),
        &pool,
        1,
        None,
        Network::Mainnet,
    )
    .unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("at least one input or output")),
        "completely empty tx should be rejected: {res:?}"
    );
}

#[test]
fn test_replacement_checks_rejects_coinbase_new_tx() {
    let coinbase = create_coinbase_tx(1);
    let existing = create_test_tx(1_000, Some(0xfffffffe), None, None);
    let (set, _) = create_test_utxo(10_000);
    let pool = Mempool::new();
    assert!(replacement_checks(&coinbase, &existing, &set, &pool).is_err());
}

#[test]
fn test_replacement_checks_rejects_coinbase_existing_tx() {
    let new_tx = create_test_tx(1_000, Some(0xfffffffe), None, None);
    let coinbase = create_coinbase_tx(1);
    let (set, _) = create_test_utxo(10_000);
    let pool = Mempool::new();
    assert!(replacement_checks(&new_tx, &coinbase, &set, &pool).is_err());
}

#[test]
fn test_accept_rejects_witness_count_mismatch() {
    let (set, prev) = create_test_utxo(10_000);
    let tx = create_test_tx(9_000, Some(0xffffffff), None, None);
    let mut tx = tx;
    tx.inputs[0].prevout = prev;
    let pool = Mempool::new();
    let witnesses = vec![vec![vec![0u8; 32]], vec![vec![0u8; 32]]];
    let res = accept_to_memory_pool(
        &tx,
        Some(&witnesses),
        &set,
        &pool,
        1_000_000,
        None,
        Network::Mainnet,
    )
    .unwrap();
    assert!(
        matches!(res, MempoolResult::Rejected(ref r) if r.contains("Witness count")),
        "expected witness mismatch rejection, got {res:?}"
    );
}
