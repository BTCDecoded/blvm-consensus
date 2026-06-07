//! COV-C-01c: `block::apply_transaction` and `calculate_tx_id` coverage.

use blvm_consensus::block::{apply_transaction, calculate_tx_id};
use blvm_consensus::economic::get_block_subsidy;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet, UTXO};
use std::sync::Arc;

fn coinbase(height_byte: u8) -> Transaction {
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![OP_1, height_byte].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: get_block_subsidy(1),
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

fn spend(prevout_byte: u8, _input_value: i64, output_value: i64) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [prevout_byte; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: output_value,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

#[test]
fn test_apply_coinbase_adds_utxo_and_undo() {
    let tx = coinbase(1);
    let (set, undo) = apply_transaction(&tx, UtxoSet::default(), 1).unwrap();
    assert_eq!(set.len(), 1);
    assert_eq!(undo.len(), 1);
    let tx_id = calculate_tx_id(&tx);
    let utxo = set
        .get(&OutPoint {
            hash: tx_id,
            index: 0,
        })
        .unwrap();
    assert!(utxo.is_coinbase);
    assert_eq!(utxo.value, get_block_subsidy(1));
}

#[test]
fn test_apply_spend_moves_value() {
    let fund = coinbase(2);
    let (set, _) = apply_transaction(&fund, UtxoSet::default(), 1).unwrap();
    let fund_id = calculate_tx_id(&fund);
    let prevout = OutPoint {
        hash: fund_id,
        index: 0,
    };

    let mut spend_tx = spend(0x00, fund.outputs[0].value, fund.outputs[0].value - 1_000);
    spend_tx.inputs[0].prevout = prevout;

    let (set, undo) = apply_transaction(&spend_tx, set, 2).unwrap();
    assert!(set.get(&prevout).is_none());
    let spend_id = calculate_tx_id(&spend_tx);
    assert!(set
        .get(&OutPoint {
            hash: spend_id,
            index: 0
        })
        .is_some());
    assert_eq!(undo.len(), 2, "one removed input + one new output");
}

#[test]
fn test_apply_spend_missing_prevout_still_adds_outputs() {
    let mut set = UtxoSet::default();
    set.insert(
        OutPoint {
            hash: [0x99; 32],
            index: 0,
        },
        Arc::new(UTXO {
            value: 10_000,
            script_pubkey: vec![OP_1].into(),
            height: 0,
            is_coinbase: false,
        }),
    );

    let spend_tx = spend(0x01, 10_000, 9_000);
    let (set, _) = apply_transaction(&spend_tx, set, 3).unwrap();
    let spend_id = calculate_tx_id(&spend_tx);
    assert!(set
        .get(&OutPoint {
            hash: spend_id,
            index: 0
        })
        .is_some());
}

#[test]
fn test_calculate_tx_id_stable() {
    let tx = coinbase(3);
    assert_eq!(calculate_tx_id(&tx), calculate_tx_id(&tx));
    assert_ne!(calculate_tx_id(&tx), calculate_tx_id(&coinbase(4)));
}

#[test]
fn test_apply_coinbase_multiple_outputs() {
    let mut tx = coinbase(1);
    tx.outputs = vec![
        TransactionOutput {
            value: get_block_subsidy(1) / 2,
            script_pubkey: vec![OP_1].into(),
        },
        TransactionOutput {
            value: get_block_subsidy(1) / 2,
            script_pubkey: vec![OP_1].into(),
        },
    ]
    .into();
    let (set, undo) = apply_transaction(&tx, UtxoSet::default(), 1).unwrap();
    assert_eq!(set.len(), 2);
    assert_eq!(undo.len(), 2);
}
