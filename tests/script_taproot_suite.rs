//! COV-C-06a: Taproot script verification paths (P2TR rejects, tapscript eval).

use blvm_consensus::opcodes::{OP_1, OP_2, OP_3, OP_ADD, PUSH_32_BYTES};
use blvm_consensus::script::flags::{SCRIPT_VERIFY_TAPROOT, SCRIPT_VERIFY_WITNESS};
use blvm_consensus::script::{eval_script, verify_script_with_context, SigVersion};
use blvm_consensus::taproot::{compute_script_merkle_root, TAPROOT_LEAF_VERSION_TAPSCRIPT};
use blvm_consensus::types::Network;
use blvm_consensus::{
    OutPoint, Transaction, TransactionInput, TransactionOutput, TAPROOT_ACTIVATION_MAINNET,
};

fn p2tr_scriptpubkey(key: &[u8; 32]) -> Vec<u8> {
    let mut spk = vec![OP_1, PUSH_32_BYTES];
    spk.extend_from_slice(key);
    spk
}

fn valid_internal_key() -> [u8; 32] {
    [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ]
}

fn taproot_script_path_witness() -> (Vec<u8>, Vec<Vec<u8>>) {
    let internal = valid_internal_key();
    let tapscript = vec![OP_1];
    let merkle_root =
        compute_script_merkle_root(&tapscript, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    let (output_key, parity) =
        blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(&internal, &merkle_root)
            .unwrap();
    let spk = p2tr_scriptpubkey(&output_key);
    let mut control_block = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control_block.extend_from_slice(&internal);
    let witness = vec![tapscript, control_block];
    (spk, witness)
}

#[test]
fn test_p2tr_rejects_nonempty_scriptsig() {
    let key = [0x11; 32];
    let script_pubkey = p2tr_scriptpubkey(&key);
    let tx = Transaction {
        version: 2,
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
    assert!(!verify_script_with_context(
        &tx.inputs[0].script_sig,
        &prevouts[0].script_pubkey,
        Some(&vec![vec![0x40; 64]]),
        SCRIPT_VERIFY_WITNESS,
        &tx,
        0,
        &prevouts,
        Some(TAPROOT_ACTIVATION_MAINNET),
        Network::Mainnet,
    )
    .unwrap());
}

#[test]
fn test_p2tr_rejects_missing_witness() {
    let key = [0x12; 32];
    let script_pubkey = p2tr_scriptpubkey(&key);
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
    let prevouts = vec![TransactionOutput {
        value: 10_000,
        script_pubkey: script_pubkey.into(),
    }];
    assert!(!verify_script_with_context(
        &tx.inputs[0].script_sig,
        &prevouts[0].script_pubkey,
        None,
        SCRIPT_VERIFY_WITNESS,
        &tx,
        0,
        &prevouts,
        Some(TAPROOT_ACTIVATION_MAINNET),
        Network::Mainnet,
    )
    .unwrap());
}

#[test]
fn test_tapscript_basic_eval() {
    let script = vec![OP_2, OP_3, OP_ADD];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Tapscript).unwrap());
    assert_eq!(stack[0].as_slice(), &[5]);
}

#[test]
fn test_p2tr_script_path_op_true_succeeds() {
    let (script_pubkey, witness) = taproot_script_path_witness();
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x20; 32],
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
    assert!(verify_script_with_context(
        &tx.inputs[0].script_sig,
        &prevouts[0].script_pubkey,
        Some(&witness),
        SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT,
        &tx,
        0,
        &prevouts,
        Some(TAPROOT_ACTIVATION_MAINNET),
        Network::Mainnet,
    )
    .unwrap());
}
