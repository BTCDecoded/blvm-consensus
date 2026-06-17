//! COV-C-06a: Taproot module coverage (merkle root, script path, witness parse, sighash).

use blvm_consensus::opcodes::{OP_1, OP_2};
use blvm_consensus::taproot::{
    TAPROOT_LEAF_VERSION_TAPSCRIPT, TAPROOT_SCRIPT_PREFIX, Witness, compute_script_merkle_root,
    compute_taproot_signature_hash, compute_taproot_tweak, compute_tapscript_signature_hash,
    extract_taproot_output_key, is_taproot_output, parse_taproot_script_path_witness,
    validate_taproot_key_aggregation, validate_taproot_script, validate_taproot_script_path,
    validate_taproot_script_path_with_leaf_version, validate_taproot_transaction,
};
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

fn p2tr_scriptpubkey(key: &[u8; 32]) -> Vec<u8> {
    let mut spk = vec![TAPROOT_SCRIPT_PREFIX];
    spk.extend_from_slice(key);
    spk.push(0x00);
    spk
}

fn sample_tx() -> Transaction {
    Transaction {
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
    }
}

#[test]
fn test_validate_taproot_script_rejects_wrong_length() {
    assert!(!validate_taproot_script(&vec![TAPROOT_SCRIPT_PREFIX]).unwrap());
    assert!(!validate_taproot_script(&vec![0x00; 34]).unwrap());
}

#[test]
fn test_extract_taproot_output_key_roundtrip() {
    let key = [0xab; 32];
    let spk = p2tr_scriptpubkey(&key);
    assert_eq!(extract_taproot_output_key(&spk).unwrap(), Some(key));
    assert!(is_taproot_output(&TransactionOutput {
        value: 0,
        script_pubkey: spk.into(),
    }));
}

#[test]
fn test_script_path_merkle_root_validates() {
    let script = vec![OP_1];
    let root =
        compute_script_merkle_root(&script, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).expect("root");
    assert!(validate_taproot_script_path(&script, &[], &root).unwrap());
    assert!(!validate_taproot_script_path(&script, &[], &[0xff; 32]).unwrap());
}

#[test]
fn test_script_path_two_leaf_tree() {
    let script_a = vec![OP_1];
    let script_b = vec![OP_2];
    let root_a =
        compute_script_merkle_root(&script_a, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    let root_b =
        compute_script_merkle_root(&script_b, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    let branch = compute_script_merkle_root(&script_a, &[root_b], TAPROOT_LEAF_VERSION_TAPSCRIPT)
        .expect("branch root");
    assert!(validate_taproot_script_path(&script_a, &[root_b], &branch).unwrap());
    let _ = root_a;
}

#[test]
fn test_parse_taproot_witness_rejects_short() {
    let key = [0x01; 32];
    let empty: Witness = vec![];
    assert!(
        parse_taproot_script_path_witness(&empty, &key)
            .unwrap()
            .is_none()
    );
    let short: Witness = vec![vec![OP_1]];
    assert!(
        parse_taproot_script_path_witness(&short, &key)
            .unwrap()
            .is_none()
    );
}

#[test]
fn test_compute_taproot_signature_hash_smoke() {
    let tx = sample_tx();
    let spk = p2tr_scriptpubkey(&[0x02; 32]);
    let pv = vec![10_000i64];
    let ps: Vec<&[u8]> = vec![spk.as_slice()];
    let h1 = compute_taproot_signature_hash(&tx, 0, &pv, &ps, 0x00, None).unwrap();
    let h2 = compute_taproot_signature_hash(&tx, 0, &pv, &ps, 0x01, None).unwrap();
    assert_ne!(h1, h2);
}

#[test]
fn test_compute_tapscript_signature_hash_smoke() {
    let tx = sample_tx();
    let script_code = vec![OP_1];
    let spk = p2tr_scriptpubkey(&[0x03; 32]);
    let pv = vec![10_000i64];
    let ps: Vec<&[u8]> = vec![spk.as_slice()];
    let hash = compute_tapscript_signature_hash(
        &tx,
        0,
        &pv,
        &ps,
        &script_code,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        0xffffffff,
        0x00,
        None,
    )
    .unwrap();
    assert_eq!(hash.len(), 32);
}

fn valid_internal_pubkey() -> [u8; 32] {
    [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ]
}

#[test]
fn test_compute_taproot_tweak_returns_32_bytes() {
    let internal = valid_internal_pubkey();
    let merkle_root = [0x05u8; 32];
    let tweaked = compute_taproot_tweak(&internal, &merkle_root).unwrap();
    assert_eq!(tweaked.len(), 32);
}

#[test]
fn test_validate_taproot_key_aggregation_smoke() {
    let internal = valid_internal_pubkey();
    let merkle_root = [0x00u8; 32];
    let (output, parity) =
        blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(&internal, &merkle_root)
            .unwrap();
    assert!(validate_taproot_key_aggregation(&internal, &merkle_root, &output, parity).unwrap());
}

#[test]
fn test_validate_taproot_script_path_with_leaf_version() {
    let script = vec![OP_1];
    let root = compute_script_merkle_root(&script, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    assert!(
        validate_taproot_script_path_with_leaf_version(
            &script,
            &[],
            &root,
            TAPROOT_LEAF_VERSION_TAPSCRIPT,
        )
        .unwrap()
    );
}

#[test]
fn test_validate_taproot_transaction_accepts_key_path_witness() {
    let key = [0x07u8; 32];
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x08; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: p2tr_scriptpubkey(&key).into(),
        }]
        .into(),
        lock_time: 0,
    };
    let witness: Witness = vec![vec![0x40; 64]];
    assert!(validate_taproot_transaction(&tx, Some(&witness)).unwrap());
}

#[test]
fn test_validate_taproot_transaction_rejects_invalid_script_path_witness() {
    let key = [0x09u8; 32];
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x0a; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: p2tr_scriptpubkey(&key).into(),
        }]
        .into(),
        lock_time: 0,
    };
    // Script path needs script + control block; single push is invalid structure.
    let witness: Witness = vec![vec![OP_1], vec![OP_1]];
    assert!(!validate_taproot_transaction(&tx, Some(&witness)).unwrap());
}

fn multi_input_tx() -> (Transaction, Vec<i64>, Vec<Vec<u8>>) {
    let tx = Transaction {
        version: 2,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x10; 32],
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xfffffffe,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x11; 32],
                    index: 1,
                },
                script_sig: vec![].into(),
                sequence: 0xfffffffd,
            },
        ]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: 2_000,
                script_pubkey: vec![OP_1].into(),
            },
            TransactionOutput {
                value: 3_000,
                script_pubkey: vec![OP_2].into(),
            },
        ]
        .into(),
        lock_time: 42,
    };
    let key_a = p2tr_scriptpubkey(&[0x20; 32]);
    let key_b = p2tr_scriptpubkey(&[0x21; 32]);
    let pv = vec![10_000i64, 20_000];
    let ps = vec![key_a, key_b];
    (tx, pv, ps)
}

#[test]
fn test_taproot_sighash_none_and_single_differ() {
    let (tx, pv, ps) = multi_input_tx();
    let ps_refs: Vec<&[u8]> = ps.iter().map(|s| s.as_slice()).collect();
    let all = compute_taproot_signature_hash(&tx, 0, &pv, &ps_refs, 0x01, None).unwrap();
    let none = compute_taproot_signature_hash(&tx, 0, &pv, &ps_refs, 0x02, None).unwrap();
    let single = compute_taproot_signature_hash(&tx, 0, &pv, &ps_refs, 0x03, None).unwrap();
    let single_in1 = compute_taproot_signature_hash(&tx, 1, &pv, &ps_refs, 0x03, None).unwrap();
    assert_ne!(all, none);
    assert_ne!(single, single_in1);
}

#[test]
fn test_taproot_sighash_anyonecanpay_differs() {
    let (tx, pv, ps) = multi_input_tx();
    let ps_refs: Vec<&[u8]> = ps.iter().map(|s| s.as_slice()).collect();
    let all = compute_taproot_signature_hash(&tx, 0, &pv, &ps_refs, 0x01, None).unwrap();
    let acp = compute_taproot_signature_hash(&tx, 1, &pv, &ps_refs, 0x81, None).unwrap();
    assert_ne!(all, acp);
}

#[test]
fn test_taproot_sighash_invalid_type_errors() {
    let tx = sample_tx();
    let spk = p2tr_scriptpubkey(&[0x02; 32]);
    let pv = vec![10_000i64];
    let ps: Vec<&[u8]> = vec![spk.as_slice()];
    assert!(compute_taproot_signature_hash(&tx, 0, &pv, &ps, 0x04, None).is_err());
}

#[test]
fn test_tapscript_sighash_differs_from_keypath() {
    let (tx, pv, ps) = multi_input_tx();
    let ps_refs: Vec<&[u8]> = ps.iter().map(|s| s.as_slice()).collect();
    let script_code = vec![OP_1, OP_2];
    let keypath = compute_taproot_signature_hash(&tx, 0, &pv, &ps_refs, 0x00, None).unwrap();
    let scriptpath = compute_tapscript_signature_hash(
        &tx,
        0,
        &pv,
        &ps_refs,
        &script_code,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        0xffffffff,
        0x00,
        None,
    )
    .unwrap();
    assert_ne!(keypath, scriptpath);
}

#[test]
fn test_parse_taproot_script_path_witness_success() {
    let internal = valid_internal_pubkey();
    let tapscript = vec![OP_1];
    let merkle_root =
        compute_script_merkle_root(&tapscript, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    let (output_key, parity) =
        blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(&internal, &merkle_root)
            .unwrap();
    let mut control_block = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control_block.extend_from_slice(&internal);
    let witness: Witness = vec![tapscript.clone(), control_block];
    let parsed = parse_taproot_script_path_witness(&witness, &output_key)
        .unwrap()
        .expect("valid script-path witness must parse");
    assert_eq!(parsed.0.as_slice(), tapscript.as_slice());
}

#[test]
fn test_validate_taproot_key_aggregation_rejects_wrong_output() {
    let internal = valid_internal_pubkey();
    let merkle_root = [0x00u8; 32];
    let (output, parity) =
        blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(&internal, &merkle_root)
            .unwrap();
    let mut wrong = output;
    wrong[0] ^= 0x01;
    assert!(!validate_taproot_key_aggregation(&internal, &merkle_root, &wrong, parity).unwrap());
}

#[test]
fn test_parse_taproot_script_path_rejects_malformed_control_block() {
    let key = [0x01; 32];
    let tapscript = vec![OP_1];
    let mut bad_control = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT];
    bad_control.extend_from_slice(&[0x02; 32]);
    bad_control.push(0x03);
    let witness: Witness = vec![tapscript, bad_control];
    assert!(
        parse_taproot_script_path_witness(&witness, &key)
            .unwrap()
            .is_none()
    );
}

#[test]
fn test_parse_taproot_script_path_with_annex() {
    let internal = valid_internal_pubkey();
    let tapscript = vec![OP_1];
    let merkle_root =
        compute_script_merkle_root(&tapscript, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    let (output_key, parity) =
        blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(&internal, &merkle_root)
            .unwrap();
    let mut control_block = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control_block.extend_from_slice(&internal);
    let annex = vec![0x50, 0x01];
    let witness: Witness = vec![vec![OP_1], tapscript.clone(), annex, control_block];
    let parsed = parse_taproot_script_path_witness(&witness, &output_key)
        .unwrap()
        .expect("annex-bearing witness must parse");
    assert_eq!(parsed.0.as_slice(), tapscript.as_slice());
    assert_eq!(parsed.1.len(), 1);
}

#[test]
fn test_tapscript_sighash_anyonecanpay_and_invalid_type() {
    let (tx, pv, ps) = multi_input_tx();
    let ps_refs: Vec<&[u8]> = ps.iter().map(|s| s.as_slice()).collect();
    let script_code = vec![OP_1];
    let all = compute_tapscript_signature_hash(
        &tx,
        0,
        &pv,
        &ps_refs,
        &script_code,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        0xffffffff,
        0x03,
        None,
    )
    .unwrap();
    let acp = compute_tapscript_signature_hash(
        &tx,
        1,
        &pv,
        &ps_refs,
        &script_code,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        0xffffffff,
        0x83,
        None,
    )
    .unwrap();
    assert_ne!(all, acp);
    assert!(
        compute_tapscript_signature_hash(
            &tx,
            0,
            &pv,
            &ps_refs,
            &script_code,
            TAPROOT_LEAF_VERSION_TAPSCRIPT,
            0xffffffff,
            0x04,
            None,
        )
        .is_err()
    );
}

#[test]
fn test_tapscript_sighash_anyonecanpay_rejects_out_of_range_input() {
    let (tx, pv, ps) = multi_input_tx();
    let ps_refs: Vec<&[u8]> = ps.iter().map(|s| s.as_slice()).collect();
    let script_code = vec![OP_1];
    assert!(
        compute_tapscript_signature_hash(
            &tx,
            99,
            &pv,
            &ps_refs,
            &script_code,
            TAPROOT_LEAF_VERSION_TAPSCRIPT,
            0xffffffff,
            0x83,
            None,
        )
        .is_err()
    );
}

#[test]
fn test_taproot_sighash_rejects_short_prevout_vectors() {
    let (tx, _, ps) = multi_input_tx();
    let ps_refs: Vec<&[u8]> = ps.iter().map(|s| s.as_slice()).collect();
    let short_values = vec![10_000i64];
    assert!(compute_taproot_signature_hash(&tx, 0, &short_values, &ps_refs, 0x01, None).is_err());
}
