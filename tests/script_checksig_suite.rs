//! COV-C-02e: OP_CHECKSIG / OP_CHECKSIGVERIFY coverage (eval + Core vectors + tx context).

#[path = "core_test_vectors/script_tests.rs"]
mod core_script_tests;

use blvm_consensus::opcodes::{OP_0, OP_1, OP_CHECKSIG, OP_CHECKSIGVERIFY, OP_NOT};
use blvm_consensus::script::flags::{SCRIPT_VERIFY_DERSIG, SCRIPT_VERIFY_STRICTENC};
use blvm_consensus::script::{SigVersion, eval_script, verify_script, verify_script_with_context};
use blvm_consensus::types::Network;
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};
use core_script_tests::{load_default_script_vectors, parse_flag_string, score_core_script_tests};

fn sample_tx(script_sig: Vec<u8>, script_pubkey: Vec<u8>) -> (Transaction, Vec<TransactionOutput>) {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0xab; 32],
                index: 0,
            },
            script_sig: script_sig.into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: script_pubkey.clone().into(),
        }]
        .into(),
        lock_time: 0,
    };
    let prevouts = vec![TransactionOutput {
        value: 10_000,
        script_pubkey: script_pubkey.into(),
    }];
    (tx, prevouts)
}

#[test]
fn test_op_checksig_insufficient_stack_fails() {
    let mut stack = Vec::new();
    stack.push(vec![OP_1].into());
    let script = vec![OP_CHECKSIG];
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_err());
}

#[test]
fn test_op_checksig_invalid_signature_pushes_false() {
    let mut stack = Vec::new();
    stack.push(vec![OP_0].into());
    stack.push(vec![0x02, 0x01, 0x01].into());
    let script = vec![OP_CHECKSIG];
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert!(stack.last().unwrap().is_empty());
}

#[test]
fn test_op_checksigverify_invalid_signature_fails() {
    let mut stack = Vec::new();
    stack.push(vec![OP_0].into());
    stack.push(vec![0x02, 0x01, 0x01].into());
    let script = vec![OP_CHECKSIGVERIFY];
    assert!(!eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_checksig_nullfail_rejects_failed_nonempty_sig() {
    let mut stack = Vec::new();
    stack.push(vec![OP_0].into());
    stack.push(vec![0x02, 0x01, 0x01].into());
    let script = vec![OP_CHECKSIG];
    assert!(
        eval_script(
            &script,
            &mut stack,
            blvm_consensus::script::flags::SCRIPT_VERIFY_NULLFAIL,
            SigVersion::Base,
        )
        .is_err()
    );
}

#[test]
fn test_checksig_not_empty_sig_via_verify_script() {
    // Core: ["0", "0x21 <pubkey> CHECKSIG NOT", "STRICTENC", "OK"]
    let flags = SCRIPT_VERIFY_STRICTENC;
    let script_sig = vec![OP_0];
    let mut script_pubkey = vec![0x21];
    script_pubkey.extend_from_slice(
        &hex::decode("02865c40293a680cb9c020e7b1e106d8c1916d3cef99aa431a56d253e69256dac0").unwrap(),
    );
    script_pubkey.push(OP_CHECKSIG);
    script_pubkey.push(OP_NOT);
    assert!(verify_script(&script_sig, &script_pubkey, None, flags).unwrap());
}

#[test]
fn test_checksig_not_with_transaction_context() {
    let flags = SCRIPT_VERIFY_STRICTENC | SCRIPT_VERIFY_DERSIG;
    let script_sig = vec![OP_0];
    let mut script_pubkey = vec![0x21];
    script_pubkey.extend_from_slice(
        &hex::decode("02865c40293a680cb9c020e7b1e106d8c1916d3cef99aa431a56d253e69256dac0").unwrap(),
    );
    script_pubkey.push(OP_CHECKSIG);
    script_pubkey.push(OP_NOT);

    let (tx, prevouts) = sample_tx(script_sig.clone(), script_pubkey.clone());
    assert!(
        verify_script_with_context(
            &tx.inputs[0].script_sig,
            &script_pubkey,
            None,
            flags,
            &tx,
            0,
            &prevouts,
            Some(500_000),
            Network::Mainnet,
        )
        .unwrap()
    );
}

#[test]
fn test_checksig_stack_underflow_errors() {
    let flags = SCRIPT_VERIFY_STRICTENC;
    // Core: ["", "CHECKSIG NOT", "STRICTENC", "INVALID_STACK_OPERATION"]
    let script_pubkey = vec![OP_CHECKSIG, OP_NOT];
    let empty_sig: Vec<u8> = vec![];
    let result = verify_script(&empty_sig, &script_pubkey, None, flags);
    assert!(
        matches!(
            result,
            Err(blvm_consensus::error::ConsensusError::ScriptErrorWithCode {
                code: blvm_consensus::error::ScriptErrorCode::InvalidStackOperation,
                ..
            })
        ),
        "empty stack CHECKSIG must be InvalidStackOperation, got {result:?}"
    );

    // Core: ["0", "CHECKSIG NOT", "STRICTENC", "INVALID_STACK_OPERATION"] — one item, needs two
    let script_sig = vec![OP_0];
    let result = verify_script(&script_sig, &script_pubkey, None, flags);
    assert!(
        matches!(
            result,
            Err(blvm_consensus::error::ConsensusError::ScriptErrorWithCode {
                code: blvm_consensus::error::ScriptErrorCode::InvalidStackOperation,
                ..
            })
        ),
        "one-item CHECKSIG must be InvalidStackOperation, got {result:?}"
    );
}

#[test]
fn test_core_checksig_not_ok_vectors_if_present() {
    let vectors = load_default_script_vectors().expect("load script_tests.json");
    assert!(!vectors.is_empty(), "script_tests.json must load");

    let subset: Vec<_> = vectors
        .into_iter()
        .filter(|v| {
            v.expected_ok
                && v.script_pubkey_asm.contains("CHECKSIG")
                && v.script_pubkey_asm.contains("NOT")
                && !v.script_sig_asm.contains("0x")
        })
        .collect();
    assert!(
        !subset.is_empty(),
        "CHECKSIG NOT success rows with an empty signature must load"
    );

    let (passed, failed) = score_core_script_tests(&subset);
    assert_eq!(
        failed, 0,
        "expected all Core CHECKSIG NOT OK vectors to pass via verify_script, got {passed} passed / {failed} failed"
    );
}

#[test]
fn test_core_checksig_with_context_if_present() {
    let vectors = load_default_script_vectors().expect("load");
    assert!(!vectors.is_empty(), "script_tests.json must load");

    // An empty signature does not depend on the prevout those other rows were signed for.
    let subset: Vec<_> = vectors
        .into_iter()
        .filter(|v| {
            v.expected_ok
                && v.script_pubkey_asm.contains("CHECKSIG")
                && v.script_pubkey_asm.contains("NOT")
                && (v.script_sig_asm == "0" || v.script_sig_asm.is_empty())
        })
        .collect();
    assert!(
        !subset.is_empty(),
        "empty-signature CHECKSIG NOT rows must load"
    );

    let mut failed = 0usize;
    for vector in &subset {
        let (tx, prevouts) = sample_tx(vector.script_sig.clone(), vector.script_pubkey.clone());
        let result = verify_script_with_context(
            &tx.inputs[0].script_sig,
            &vector.script_pubkey,
            None,
            vector.flags,
            &tx,
            0,
            &prevouts,
            Some(500_000),
            Network::Mainnet,
        );
        if !matches!(result, Ok(true)) {
            failed += 1;
        }
    }
    assert_eq!(
        failed, 0,
        "empty-signature CHECKSIG NOT rows must succeed with a transaction"
    );
}

#[test]
fn test_core_checksig_strictenc_progress_if_present() {
    let vectors = load_default_script_vectors().expect("load");
    assert!(!vectors.is_empty(), "script_tests.json must load");

    let subset: Vec<_> = vectors
        .into_iter()
        .filter(|v| {
            v.expected_ok
                && v.flags == parse_flag_string("STRICTENC")
                && v.script_pubkey_asm.contains("CHECKSIG")
                && v.script_pubkey_asm.contains("NOT")
        })
        .collect();
    assert!(
        !subset.is_empty(),
        "STRICTENC CHECKSIG NOT success rows must load"
    );

    let (passed, failed) = score_core_script_tests(&subset);
    assert_eq!(
        failed,
        0,
        "STRICTENC CHECKSIG NOT vectors {passed}/{} matched",
        subset.len()
    );
}
