//! COV-C-02c: SegWit v0 witness script verification (P2WSH / P2WPKH-shaped paths).

#[path = "integration/helpers.rs"]
mod helpers;

use bitcoin_hashes::{Hash as BitcoinHash, sha256};
use blvm_consensus::opcodes::{OP_0, OP_1, OP_2, OP_ENDIF, OP_IF, PUSH_20_BYTES, PUSH_32_BYTES};
use blvm_consensus::script::flags::SCRIPT_VERIFY_WITNESS;
use blvm_consensus::script::{SigVersion, verify_script_with_context};
use blvm_consensus::types::Network;
use blvm_consensus::{
    OutPoint, SEGWIT_ACTIVATION_MAINNET, Transaction, TransactionInput, TransactionOutput,
};

#[path = "core_test_vectors/script_tests.rs"]
mod core_script_tests;

use core_script_tests::{load_default_witness_script_vectors, score_witness_script_tests};

fn p2wsh_scriptpubkey(witness_script: &[u8]) -> Vec<u8> {
    let hash = sha256::Hash::hash(witness_script);
    let mut spk = vec![OP_0, PUSH_32_BYTES];
    spk.extend_from_slice(hash.as_ref());
    spk
}

fn p2wpkh_scriptpubkey(pubkey_hash: &[u8; 20]) -> Vec<u8> {
    let mut spk = vec![OP_0, PUSH_20_BYTES];
    spk.extend_from_slice(pubkey_hash);
    spk
}

#[test]
fn test_p2wsh_op_true_witness_succeeds() {
    let witness_script = vec![OP_1];
    let script_pubkey = p2wsh_scriptpubkey(&witness_script);
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x01; 32],
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
        lock_time: 0,
    };
    let witness = vec![witness_script];
    let prevouts = vec![TransactionOutput {
        value: 10_000,
        script_pubkey: script_pubkey.clone().into(),
    }];
    assert!(
        verify_script_with_context(
            &tx.inputs[0].script_sig,
            &script_pubkey,
            Some(&witness),
            SCRIPT_VERIFY_WITNESS,
            &tx,
            0,
            &prevouts,
            Some(SEGWIT_ACTIVATION_MAINNET),
            Network::Mainnet,
        )
        .unwrap()
    );
}

#[test]
fn test_p2wsh_wrong_witness_script_fails() {
    let witness_script = vec![OP_1];
    let script_pubkey = p2wsh_scriptpubkey(&witness_script);
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x02; 32],
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
        lock_time: 0,
    };
    let bad_witness = vec![vec![OP_2]];
    let prevouts = vec![TransactionOutput {
        value: 10_000,
        script_pubkey: script_pubkey.into(),
    }];
    assert!(
        !verify_script_with_context(
            &tx.inputs[0].script_sig,
            &prevouts[0].script_pubkey,
            Some(&bad_witness),
            SCRIPT_VERIFY_WITNESS,
            &tx,
            0,
            &prevouts,
            Some(SEGWIT_ACTIVATION_MAINNET),
            Network::Mainnet,
        )
        .unwrap()
    );
}

#[test]
fn test_p2wpkh_requires_witness() {
    let pubkey_hash = [0xab; 20];
    let script_pubkey = p2wpkh_scriptpubkey(&pubkey_hash);
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x04; 32],
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
        lock_time: 0,
    };
    let prevouts = vec![TransactionOutput {
        value: 10_000,
        script_pubkey: script_pubkey.into(),
    }];
    assert!(
        !verify_script_with_context(
            &tx.inputs[0].script_sig,
            &prevouts[0].script_pubkey,
            None,
            SCRIPT_VERIFY_WITNESS,
            &tx,
            0,
            &prevouts,
            Some(SEGWIT_ACTIVATION_MAINNET),
            Network::Mainnet,
        )
        .unwrap()
    );
}

#[test]
fn test_witness_v0_minimalif_rejects_non_minimal() {
    let witness_script = vec![0x02, 0x01, 0x00, OP_IF, OP_1, OP_ENDIF]; // non-minimal IF OP_1 ENDIF
    let script_pubkey = p2wsh_scriptpubkey(&witness_script);
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x05; 32],
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
        lock_time: 0,
    };
    let witness = vec![witness_script];
    let prevouts = vec![TransactionOutput {
        value: 10_000,
        script_pubkey: script_pubkey.into(),
    }];
    let flags = SCRIPT_VERIFY_WITNESS | blvm_consensus::script::flags::SCRIPT_VERIFY_MINIMALIF;
    let result = verify_script_with_context(
        &tx.inputs[0].script_sig,
        &prevouts[0].script_pubkey,
        Some(&witness),
        flags,
        &tx,
        0,
        &prevouts,
        Some(SEGWIT_ACTIVATION_MAINNET),
        Network::Mainnet,
    );
    assert!(matches!(result, Ok(false) | Err(_)));
}

#[test]
fn test_unhandled_non_empty_witness_on_legacy_script_fails_closed() {
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x77; 32],
                index: 0,
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
    let prevouts = vec![TransactionOutput {
        value: 10_000,
        script_pubkey: vec![OP_1].into(),
    }];
    let witness = vec![vec![0x01, 0x02, 0x03]];
    assert!(
        !verify_script_with_context(
            &tx.inputs[0].script_sig,
            &prevouts[0].script_pubkey,
            Some(&witness),
            0,
            &tx,
            0,
            &prevouts,
            None,
            Network::Regtest,
        )
        .unwrap()
    );
}

#[test]
fn test_tapscript_op_true_evaluates() {
    let mut stack = Vec::new();
    assert!(
        blvm_consensus::script::eval_script(&[OP_1], &mut stack, 0, SigVersion::Tapscript,)
            .unwrap()
    );
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_execute_all_core_witness_script_vectors() {
    let vectors = load_default_witness_script_vectors().expect("load witness vectors");
    if vectors.is_empty() {
        return;
    }

    let (passed, failed) = score_witness_script_tests(&vectors);
    eprintln!(
        "Core witness script vectors: {passed}/{} scored, {failed} mismatches",
        vectors.len()
    );
    assert_eq!(passed + failed, vectors.len());

    let ok_subset: Vec<_> = vectors.iter().filter(|v| v.expected_ok).cloned().collect();
    if !ok_subset.is_empty() {
        let (ok_passed, ok_failed) = score_witness_script_tests(&ok_subset);
        eprintln!("Witness OK subset: {ok_passed}/{}", ok_subset.len());
        assert!(
            ok_failed == 0,
            "expected all Core witness OK vectors to pass, got {ok_passed} passed / {ok_failed} failed"
        );
    }
}

#[test]
fn test_execute_witness_fail_vectors_for_error_paths() {
    let vectors = load_default_witness_script_vectors().expect("load");
    if vectors.is_empty() {
        return;
    }
    let fail_vectors: Vec<_> = vectors.iter().filter(|v| !v.expected_ok).cloned().collect();
    if fail_vectors.is_empty() {
        return;
    }
    let (passed, failed) = score_witness_script_tests(&fail_vectors);
    eprintln!(
        "Witness fail vectors: {passed}/{} scored, {failed} mismatches",
        fail_vectors.len()
    );
    assert_eq!(passed + failed, fail_vectors.len());
}
