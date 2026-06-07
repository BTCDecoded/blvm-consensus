//! COV-C-02b: CLTV/CSV coverage via verify_script_with_context_full (BIP65/BIP112 paths).

#[path = "integration/helpers.rs"]
mod helpers;

use blvm_consensus::opcodes::{OP_1, OP_CHECKLOCKTIMEVERIFY, OP_CHECKSEQUENCEVERIFY};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY, SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
};
use blvm_consensus::script::{eval_script, verify_script_with_context, SigVersion};
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

#[test]
fn test_cltv_success_with_context() {
    let script_pubkey = helpers::push_locktime_script(400_000, OP_CHECKLOCKTIMEVERIFY);
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x11; 32],
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
        lock_time: 500_000,
    };
    let prevouts = vec![TransactionOutput {
        value: 1_000_000,
        script_pubkey: script_pubkey.clone().into(),
    }];
    assert!(verify_script_with_context(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        None,
        SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY,
        &tx,
        0,
        &prevouts,
        Some(500_000),
        blvm_consensus::types::Network::Mainnet,
    )
    .unwrap());
}

#[test]
fn test_cltv_fails_when_locktime_too_low() {
    let script_pubkey = helpers::push_locktime_script(600_000, OP_CHECKLOCKTIMEVERIFY);
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x12; 32],
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
        lock_time: 500_000,
    };
    let prevouts = vec![TransactionOutput {
        value: 1_000_000,
        script_pubkey: script_pubkey.clone().into(),
    }];
    assert!(!verify_script_with_context(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        None,
        SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY,
        &tx,
        0,
        &prevouts,
        Some(500_000),
        blvm_consensus::types::Network::Mainnet,
    )
    .unwrap());
}

#[test]
fn test_cltv_fails_with_final_sequence() {
    let script_pubkey = helpers::push_locktime_script(100, OP_CHECKLOCKTIMEVERIFY);
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x13; 32],
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
        lock_time: 200,
    };
    let prevouts = vec![TransactionOutput {
        value: 1_000_000,
        script_pubkey: script_pubkey.clone().into(),
    }];
    assert!(!verify_script_with_context(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        None,
        SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY,
        &tx,
        0,
        &prevouts,
        None,
        blvm_consensus::types::Network::Mainnet,
    )
    .unwrap());
}

#[test]
fn test_csv_success_with_context() {
    let script_pubkey = helpers::push_locktime_script(4, OP_CHECKSEQUENCEVERIFY);
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x21; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 5,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let prevouts = vec![TransactionOutput {
        value: 1_000_000,
        script_pubkey: script_pubkey.clone().into(),
    }];
    assert!(verify_script_with_context(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        None,
        SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
        &tx,
        0,
        &prevouts,
        None,
        blvm_consensus::types::Network::Mainnet,
    )
    .unwrap());
}

#[test]
fn test_csv_fails_when_sequence_too_low() {
    let script_pubkey = helpers::push_locktime_script(4, OP_CHECKSEQUENCEVERIFY);
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x22; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 3,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let prevouts = vec![TransactionOutput {
        value: 1_000_000,
        script_pubkey: script_pubkey.clone().into(),
    }];
    assert!(!verify_script_with_context(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        None,
        SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
        &tx,
        0,
        &prevouts,
        None,
        blvm_consensus::types::Network::Mainnet,
    )
    .unwrap());
}

#[test]
fn test_cltv_nop_without_flag() {
    let script = helpers::push_locktime_script(500_000, OP_CHECKLOCKTIMEVERIFY);
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_csv_nop_without_flag() {
    let script = helpers::push_locktime_script(4, OP_CHECKSEQUENCEVERIFY);
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}
