//! Rejects `check_transaction` already returns before script or UTXO work.

use blvm_consensus::transaction::check_transaction;
use blvm_consensus::{
    OutPoint, Transaction, TransactionInput, TransactionOutput, ValidationResult,
};

fn one_input() -> TransactionInput {
    TransactionInput {
        prevout: OutPoint {
            hash: [1; 32],
            index: 0,
        },
        script_sig: vec![],
        sequence: 0xffff_ffff,
    }
}

fn one_output(value: i64) -> TransactionOutput {
    TransactionOutput {
        value,
        script_pubkey: vec![0x51],
    }
}

#[test]
fn check_transaction_rejects_empty_inputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![one_output(1)].into(),
        lock_time: 0,
    };
    match check_transaction(&tx) {
        Ok(ValidationResult::Invalid(message)) => {
            assert!(
                message.contains("Empty inputs or outputs"),
                "empty inputs returned {message}"
            );
        }
        Ok(other) => panic!("empty inputs returned {other:?}"),
        Err(err) => panic!("empty inputs returned {err:?}"),
    }
}

#[test]
fn check_transaction_rejects_empty_outputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![one_input()].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    match check_transaction(&tx) {
        Ok(ValidationResult::Invalid(message)) => {
            assert!(
                message.contains("Empty inputs or outputs"),
                "empty outputs returned {message}"
            );
        }
        Ok(other) => panic!("empty outputs returned {other:?}"),
        Err(err) => panic!("empty outputs returned {err:?}"),
    }
}

#[test]
fn check_transaction_rejects_a_negative_output() {
    let tx = Transaction {
        version: 1,
        inputs: vec![one_input()].into(),
        outputs: vec![one_output(-1)].into(),
        lock_time: 0,
    };
    match check_transaction(&tx) {
        Ok(ValidationResult::Invalid(message)) => {
            assert!(
                message.contains("Invalid output value: -1"),
                "negative output returned {message}"
            );
        }
        Ok(other) => panic!("negative output returned {other:?}"),
        Err(err) => panic!("negative output returned {err:?}"),
    }
}
