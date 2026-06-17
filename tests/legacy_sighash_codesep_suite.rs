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
