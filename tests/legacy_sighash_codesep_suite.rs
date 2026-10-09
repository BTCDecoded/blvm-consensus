//! Regression: legacy sighash must strip OP_CODESEPARATOR (0xab) opcodes from scriptCode
//! when serializing (Bitcoin Core SerializeScriptCode). Bytes inside push-data must be preserved.

use blvm_consensus::opcodes::{OP_CHECKSIG, OP_CODESEPARATOR, OP_DUP};
use blvm_consensus::transaction_hash::{
    SighashType, calculate_transaction_sighash_single_input,
    serialize_script_code_for_legacy_sighash,
};
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

fn sample_tx() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x55; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

#[test]
fn test_legacy_sighash_strips_opcode_codeseparator() {
    let tx = sample_tx();
    let with_codesep = vec![OP_DUP, OP_CODESEPARATOR, OP_CHECKSIG];
    let without_codesep = vec![OP_DUP, OP_CHECKSIG];

    let hash_with = calculate_transaction_sighash_single_input(
        &tx,
        0,
        &with_codesep,
        10_000,
        SighashType::ALL,
        #[cfg(feature = "production")]
        None,
    )
    .expect("with codesep");
    let hash_without = calculate_transaction_sighash_single_input(
        &tx,
        0,
        &without_codesep,
        10_000,
        SighashType::ALL,
        #[cfg(feature = "production")]
        None,
    )
    .expect("without codesep");

    assert_eq!(
        hash_with, hash_without,
        "legacy sighash must strip OP_CODESEPARATOR opcodes from scriptCode"
    );

    let serialized = serialize_script_code_for_legacy_sighash(&with_codesep);
    assert_eq!(serialized, without_codesep);
}

#[test]
fn test_legacy_sighash_does_not_strip_codesep_byte_inside_pushdata() {
    let tx = sample_tx();
    // PUSH_1 0xab OP_CHECKSIG — 0xab is push payload, not an opcode
    let script_with_push_ab = vec![0x01, 0xab, OP_CHECKSIG];
    let script_push_other = vec![0x01, 0xac, OP_CHECKSIG];

    let hash_ab = calculate_transaction_sighash_single_input(
        &tx,
        0,
        &script_with_push_ab,
        10_000,
        SighashType::ALL,
        #[cfg(feature = "production")]
        None,
    )
    .expect("push ab");
    let hash_ac = calculate_transaction_sighash_single_input(
        &tx,
        0,
        &script_push_other,
        10_000,
        SighashType::ALL,
        #[cfg(feature = "production")]
        None,
    )
    .expect("push ac");

    assert_ne!(
        hash_ab, hash_ac,
        "0xab inside push-data must not be stripped as OP_CODESEPARATOR"
    );
}

fn two_input_tx(n_out: usize) -> Transaction {
    let inputs: Vec<_> = (0..2)
        .map(|i| TransactionInput {
            prevout: OutPoint {
                hash: [0x55 + i as u8; 32],
                index: i as u32,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        })
        .collect();
    let outputs: Vec<_> = (0..n_out)
        .map(|i| TransactionOutput {
            value: 1_000 + i as i64,
            script_pubkey: vec![0x51].into(),
        })
        .collect();
    Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: outputs.into(),
        lock_time: 0,
    }
}

fn sighash_all(tx: &Transaction, input_index: usize, script: &[u8]) -> [u8; 32] {
    calculate_transaction_sighash_single_input(
        tx,
        input_index,
        script,
        10_000,
        SighashType::ALL,
        #[cfg(feature = "production")]
        None,
    )
    .expect("sighash")
}

#[test]
fn test_two_input_legacy_sighash_strips_opcode_codeseparator() {
    let with_codesep = vec![OP_DUP, OP_CODESEPARATOR, OP_CHECKSIG];
    let without_codesep = vec![OP_DUP, OP_CHECKSIG];
    for n_out in [1usize, 2] {
        let tx = two_input_tx(n_out);
        for input_index in [0usize, 1] {
            assert_eq!(
                sighash_all(&tx, input_index, &with_codesep),
                sighash_all(&tx, input_index, &without_codesep),
                "2-in-{n_out}-out input {input_index} must strip OP_CODESEPARATOR"
            );
        }
    }
}

#[test]
fn test_two_input_legacy_sighash_keeps_codesep_byte_inside_pushdata() {
    let tx = two_input_tx(2);
    let script_with_push_ab = vec![0x01, 0xab, OP_CHECKSIG];
    let script_push_other = vec![0x01, 0xac, OP_CHECKSIG];
    assert_ne!(
        sighash_all(&tx, 1, &script_with_push_ab),
        sighash_all(&tx, 1, &script_push_other),
        "0xab inside push-data must stay in a two-input sighash"
    );
}
