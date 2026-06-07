//! COV-C-01b: UtxoOverlay apply/lookup coverage for block validation paths.

use blvm_consensus::block::calculate_tx_id;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::utxo_overlay::{
    apply_transaction_to_overlay, apply_transaction_to_overlay_no_undo, to_fast_utxo_set,
    utxo_deletion_key_to_outpoint, UtxoOverlay,
};
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet, UTXO};
use std::sync::Arc;

fn seed_utxo(set: &mut UtxoSet, byte: u8, value: i64) {
    set.insert(
        OutPoint {
            hash: [byte; 32],
            index: 0,
        },
        Arc::new(UTXO {
            value,
            script_pubkey: vec![OP_1].into(),
            height: 0,
            is_coinbase: false,
        }),
    );
}

#[test]
fn test_deletion_key_roundtrip() {
    let op = OutPoint {
        hash: [0x42; 32],
        index: 0x01020304,
    };
    let base = UtxoSet::default();
    let mut overlay = UtxoOverlay::new(&base);
    overlay.insert(
        op,
        UTXO {
            value: 1,
            script_pubkey: vec![OP_1].into(),
            height: 0,
            is_coinbase: false,
        },
    );
    overlay.mark_spent(&op);
    let key = {
        let mut k = [0u8; 36];
        k[..32].copy_from_slice(&op.hash);
        k[32..36].copy_from_slice(&op.index.to_be_bytes());
        k
    };
    assert_eq!(utxo_deletion_key_to_outpoint(&key), op);
}

#[test]
fn test_apply_transaction_to_overlay_records_undo() {
    let mut base = UtxoSet::default();
    seed_utxo(&mut base, 0x01, 10_000);

    let spend = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x01; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 9_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let tx_id = calculate_tx_id(&spend);

    let mut overlay = UtxoOverlay::new(&base);
    let undo = apply_transaction_to_overlay(&mut overlay, &spend, tx_id, 100);
    assert_eq!(undo.len(), 2, "one spent input + one new output");
    assert!(overlay
        .get(&OutPoint {
            hash: [0x01; 32],
            index: 0
        })
        .is_none());
    assert!(overlay
        .get(&OutPoint {
            hash: tx_id,
            index: 0
        })
        .is_some());
}

#[test]
fn test_apply_transaction_to_overlay_no_undo_fast_path() {
    let mut base = UtxoSet::default();
    seed_utxo(&mut base, 0x02, 5_000);

    let spend = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x02; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 4_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let tx_id = calculate_tx_id(&spend);

    let mut overlay = UtxoOverlay::new(&base);
    apply_transaction_to_overlay_no_undo(&mut overlay, &spend, tx_id, 200);
    assert!(overlay
        .get(&OutPoint {
            hash: [0x02; 32],
            index: 0
        })
        .is_none());
    assert_eq!(
        overlay
            .get(&OutPoint {
                hash: tx_id,
                index: 0
            })
            .unwrap()
            .value,
        4_000
    );
}

#[test]
fn test_to_fast_utxo_set_copies_base() {
    let mut base = UtxoSet::default();
    seed_utxo(&mut base, 0x03, 1_000);
    let fast = to_fast_utxo_set(&base);
    assert_eq!(fast.len(), 1);
    assert_eq!(
        fast.get(&OutPoint {
            hash: [0x03; 32],
            index: 0
        })
        .unwrap()
        .value,
        1_000
    );
}

#[test]
fn test_overlay_intra_block_spend_from_addition() {
    let base = UtxoSet::default();
    let mut overlay = UtxoOverlay::new(&base);

    let fund = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let fund_id = calculate_tx_id(&fund);
    apply_transaction_to_overlay_no_undo(&mut overlay, &fund, fund_id, 1);

    let prevout = OutPoint {
        hash: fund_id,
        index: 0,
    };
    let spend = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout,
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 900,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let spend_id = calculate_tx_id(&spend);
    apply_transaction_to_overlay_no_undo(&mut overlay, &spend, spend_id, 1);

    assert!(overlay.get(&prevout).is_none());
    assert!(overlay
        .get(&OutPoint {
            hash: spend_id,
            index: 0
        })
        .is_some());
}

#[test]
fn test_utxo_set_lookup_trait_helpers() {
    let mut base = UtxoSet::default();
    seed_utxo(&mut base, 0x05, 100);
    use blvm_consensus::utxo_overlay::UtxoLookup as _;
    assert!(!base.is_empty());
    assert!(base.contains_key(&OutPoint {
        hash: [0x05; 32],
        index: 0
    }));
}

#[test]
fn test_overlay_contains_key_and_apply_to_base() {
    let mut base = UtxoSet::default();
    seed_utxo(&mut base, 0x06, 100);
    let op = OutPoint {
        hash: [0x06; 32],
        index: 0,
    };
    let mut overlay = UtxoOverlay::new(&base);
    assert!(overlay.contains_key(&op));
    overlay.mark_spent(&op);
    assert!(!overlay.contains_key(&op));
    overlay.insert(
        op.clone(),
        UTXO {
            value: 200,
            script_pubkey: vec![OP_1].into(),
            height: 50,
            is_coinbase: false,
        },
    );
    assert!(overlay.contains_key(&op));
    assert_eq!(overlay.get(&op).unwrap().value, 200);
    let merged = overlay.apply_to_base();
    assert_eq!(merged.get(&op).unwrap().value, 200);
}

#[test]
fn test_overlay_remove_returns_spent_utxo() {
    let mut base = UtxoSet::default();
    seed_utxo(&mut base, 0x07, 500);
    let op = OutPoint {
        hash: [0x07; 32],
        index: 0,
    };
    let mut overlay = UtxoOverlay::new(&base);
    let removed = overlay.remove(&op).unwrap();
    assert_eq!(removed.value, 500);
    assert_eq!(overlay.deletions_len(), 1);
    assert!(overlay.get(&op).is_none());
}

#[test]
fn test_overlay_with_capacity_and_into_changes() {
    let base = UtxoSet::default();
    let overlay = UtxoOverlay::with_capacity(&base, 4, 2);
    assert_eq!(overlay.base_len(), 0);
    let (additions, deletions) = overlay.into_changes();
    assert!(additions.is_empty());
    assert!(deletions.is_empty());
}

#[test]
fn test_overlay_additions_and_deletions_accessors() {
    let mut base = UtxoSet::default();
    seed_utxo(&mut base, 0x08, 300);
    let op = OutPoint {
        hash: [0x08; 32],
        index: 0,
    };
    let mut overlay = UtxoOverlay::new(&base);
    overlay.mark_spent(&op);
    assert_eq!(overlay.deletions().len(), 1);
    assert!(overlay.additions().is_empty());
}
