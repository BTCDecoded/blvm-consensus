//! Script consensus divergences from Bitcoin Core block validation.

#![cfg(feature = "production")]

use blvm_consensus::TAPROOT_ACTIVATION_MAINNET;
use blvm_consensus::activation::ForkActivationTable;
use blvm_consensus::block::get_block_script_verify_flags_core;
use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_CHECKMULTISIG, OP_CHECKSIG, OP_DROP, OP_ELSE, OP_ENDIF, OP_IF, PUSH_32_BYTES,
};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_TAPROOT_VERSION, SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_TAPROOT,
    SCRIPT_VERIFY_WITNESS, SCRIPT_VERIFY_WITNESS_PUBKEYTYPE,
};
use blvm_consensus::script::{SigVersion, disable_fast_paths, verify_script_with_context_full};
use blvm_consensus::taproot::{
    TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root, witness_stack_serialize_size,
};
use blvm_consensus::transaction_hash::calculate_bip143_sighash;
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};
use ripemd::Ripemd160;
use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};
use sha2::{Digest, Sha256};

fn one_input_tx(script_sig: Vec<u8>) -> Transaction {
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x11; 32],
                index: 0,
            },
            script_sig: script_sig.into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 9_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

fn verify(
    tx: &Transaction,
    script_pubkey: &[u8],
    witness: Option<&blvm_consensus::witness::Witness>,
    flags: u32,
    prevout_value: i64,
    height: Option<u64>,
) -> bool {
    verify_at(
        tx,
        script_pubkey,
        witness,
        flags,
        prevout_value,
        height,
        true,
    )
}

fn verify_at(
    tx: &Transaction,
    script_pubkey: &[u8],
    witness: Option<&blvm_consensus::witness::Witness>,
    flags: u32,
    prevout_value: i64,
    height: Option<u64>,
    interpreter_only: bool,
) -> bool {
    if interpreter_only {
        disable_fast_paths(true);
    }
    let values = [prevout_value];
    let scripts = [script_pubkey];
    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        script_pubkey,
        witness,
        flags,
        tx,
        0,
        &values,
        &scripts,
        height,
        None,
        Network::Mainnet,
        SigVersion::Base,
        None,
        None,
        None,
        None,
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    );
    disable_fast_paths(false);
    matches!(result, Ok(true))
}

fn internal_key() -> [u8; 32] {
    [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ]
}

fn p2tr_spend(
    tapscript: Vec<u8>,
    stack: Vec<Vec<u8>>,
    leaf_version: u8,
) -> (Transaction, Vec<u8>, Vec<Vec<u8>>) {
    let internal = internal_key();
    let root = compute_script_merkle_root(&tapscript, &[], leaf_version).expect("leaf hash");
    let (output_key, parity) =
        blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(&internal, &root)
            .expect("tweak");
    let mut script_pubkey = vec![OP_1, PUSH_32_BYTES];
    script_pubkey.extend_from_slice(&output_key);
    let mut control = vec![leaf_version | parity];
    control.extend_from_slice(&internal);
    let mut witness = stack;
    witness.push(tapscript);
    witness.push(control);
    (one_input_tx(vec![]), script_pubkey, witness)
}

fn spend_tapscript(tapscript: Vec<u8>, stack: Vec<Vec<u8>>, flags: u32) -> bool {
    let (tx, spk, witness) = p2tr_spend(tapscript, stack, TAPROOT_LEAF_VERSION_TAPSCRIPT);
    verify(
        &tx,
        &spk,
        Some(&witness),
        flags,
        10_000,
        Some(TAPROOT_ACTIVATION_MAINNET),
    )
}

fn native_p2wpkh() -> (Transaction, Vec<u8>, Vec<Vec<u8>>) {
    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&[0x22; 32]).expect("key");
    let pubkey = PublicKey::from_secret_key(&secp, &secret).serialize();
    let hash160: [u8; 20] = Ripemd160::digest(Sha256::digest(pubkey)).into();
    let mut script_pubkey = vec![OP_0, 0x14];
    script_pubkey.extend_from_slice(&hash160);
    let tx = one_input_tx(vec![]);
    let mut scriptcode = vec![0x76, 0xa9, 0x14];
    scriptcode.extend_from_slice(&hash160);
    scriptcode.extend_from_slice(&[0x88, 0xac]);
    let sighash = calculate_bip143_sighash(&tx, 0, &scriptcode, 10_000, 0x01, None).unwrap();
    let msg = Message::from_digest_slice(&sighash).unwrap();
    let mut sig = secp.sign_ecdsa(&msg, &secret).serialize_der().to_vec();
    sig.push(0x01);
    (tx, script_pubkey, vec![sig, pubkey.to_vec()])
}

#[test]
fn native_p2wpkh_rejects_nonempty_script_sig() {
    let (_tx, spk, witness) = native_p2wpkh();
    let tx = one_input_tx(vec![OP_1]);
    assert!(!verify(
        &tx,
        &spk,
        Some(&witness),
        SCRIPT_VERIFY_WITNESS,
        10_000,
        Some(800_000),
    ));
}

#[test]
fn native_p2wpkh_rejects_bad_signature() {
    let (tx, spk, mut witness) = native_p2wpkh();
    witness[0][0] ^= 0xff;
    assert!(!verify(
        &tx,
        &spk,
        Some(&witness),
        SCRIPT_VERIFY_WITNESS,
        10_000,
        Some(800_000),
    ));
}

#[test]
fn native_p2wpkh_accepts_valid_witness() {
    let (tx, spk, witness) = native_p2wpkh();
    assert!(verify(
        &tx,
        &spk,
        Some(&witness),
        SCRIPT_VERIFY_WITNESS,
        10_000,
        Some(800_000),
    ));
}

#[test]
fn block_flags_keep_witness_without_pubkeytype() {
    let table = ForkActivationTable::from_network(Network::Mainnet);
    let hash = [0xab; 32];
    let flags = get_block_script_verify_flags_core(&hash, 800_000, &table, Network::Mainnet);
    assert_eq!(
        flags & (SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT),
        SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT
    );
    assert_eq!(flags & SCRIPT_VERIFY_WITNESS_PUBKEYTYPE, 0);
    assert_ne!(flags & 0x04, 0);
    assert_ne!(flags & 0x200, 0);
    assert_ne!(flags & 0x400, 0);
    assert_ne!(flags & 0x10, 0);
    let early = get_block_script_verify_flags_core(&hash, 100, &table, Network::Mainnet);
    assert_ne!(early & SCRIPT_VERIFY_WITNESS, 0);
}

#[test]
fn v0_program_without_witness_flag_is_bare_script() {
    let mut spk = vec![OP_0, 0x14];
    spk.extend_from_slice(&[0x11; 20]);
    let tx = one_input_tx(vec![]);
    assert!(verify(&tx, &spk, None, 0, 10_000, Some(100)));
}

#[test]
fn unknown_taproot_leaf_skips_script_unless_discouraged() {
    let tapscript = {
        let mut s = vec![PUSH_32_BYTES];
        s.extend_from_slice(&[0x22; 32]);
        s.push(OP_CHECKSIG);
        s
    };
    let (tx, spk, witness) = p2tr_spend(tapscript, vec![vec![0u8; 64]], 0xc2);
    let base = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    assert!(verify(
        &tx,
        &spk,
        Some(&witness),
        base,
        10_000,
        Some(TAPROOT_ACTIVATION_MAINNET),
    ));
    assert!(!verify(
        &tx,
        &spk,
        Some(&witness),
        base | SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_TAPROOT_VERSION,
        10_000,
        Some(TAPROOT_ACTIVATION_MAINNET),
    ));
}

#[test]
fn tapscript_checksig_invalid_32byte_key_fails_script() {
    let mut script = vec![0x40];
    script.extend_from_slice(&[0u8; 64]);
    script.push(PUSH_32_BYTES);
    script.extend_from_slice(&[0x33; 32]);
    script.push(OP_CHECKSIG);
    script.push(OP_DROP);
    script.push(OP_1);
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    assert!(!spend_tapscript(script, vec![], flags));
}

#[test]
fn tapscript_checksig_empty_pubkey_fails() {
    let script = vec![0x01, 0xaa, OP_0, OP_CHECKSIG, OP_1];
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    assert!(!spend_tapscript(script, vec![], flags));
}

#[test]
fn tapscript_validation_weight_rejects_too_many_nonempty_sigs() {
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    let one = vec![0x01, 0x11, 0x01, 0x22, OP_CHECKSIG];
    let (tx, spk, witness) = p2tr_spend(one.clone(), vec![], TAPROOT_LEAF_VERSION_TAPSCRIPT);
    let budget = (witness_stack_serialize_size(&witness) + 50) / 50;
    assert!(1 <= budget);
    assert!(verify(
        &tx,
        &spk,
        Some(&witness),
        flags,
        10_000,
        Some(TAPROOT_ACTIVATION_MAINNET),
    ));

    let mut many = Vec::new();
    let mut n = 0i64;
    loop {
        many.extend_from_slice(&[0x01, 0x11, 0x01, 0x22, OP_CHECKSIG]);
        n += 1;
        let (tx, spk, witness) = p2tr_spend(many.clone(), vec![], TAPROOT_LEAF_VERSION_TAPSCRIPT);
        let budget = (witness_stack_serialize_size(&witness) + 50) / 50;
        if n > budget {
            assert!(!verify(
                &tx,
                &spk,
                Some(&witness),
                flags,
                10_000,
                Some(TAPROOT_ACTIVATION_MAINNET),
            ));
            break;
        }
        assert!(
            n < 8,
            "a short tapscript must exhaust validation weight quickly"
        );
    }
}

#[test]
fn tapscript_minimal_if_is_empty_or_one() {
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    assert!(!spend_tapscript(
        vec![0x01, 0x02, OP_IF, OP_1, OP_ENDIF],
        vec![],
        flags
    ));
    assert!(!spend_tapscript(
        vec![0x01, 0x51, OP_IF, OP_1, OP_ENDIF],
        vec![],
        flags
    ));
    assert!(spend_tapscript(
        vec![0x01, 0x01, OP_IF, OP_1, OP_ENDIF],
        vec![],
        flags
    ));
    assert!(spend_tapscript(
        vec![OP_0, OP_IF, OP_0, OP_ELSE, OP_1, OP_ENDIF],
        vec![],
        flags
    ));
}

#[test]
fn tapscript_checkmultisig_fails() {
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    assert!(!spend_tapscript(vec![OP_CHECKMULTISIG], vec![], flags));
}

fn p2wsh(script: Vec<u8>, stack: Vec<Vec<u8>>) -> (Transaction, Vec<u8>, Vec<Vec<u8>>) {
    let hash = Sha256::digest(&script);
    let mut script_pubkey = vec![OP_0, PUSH_32_BYTES];
    script_pubkey.extend_from_slice(&hash);
    let mut witness = stack;
    witness.push(script);
    (one_input_tx(vec![]), script_pubkey, witness)
}

fn p2wsh_ok(script: Vec<u8>, interpreter_only: bool) -> bool {
    let (tx, spk, witness) = p2wsh(script, vec![]);
    verify_at(
        &tx,
        &spk,
        Some(&witness),
        SCRIPT_VERIFY_WITNESS,
        10_000,
        Some(800_000),
        interpreter_only,
    )
}

#[test]
fn p2wsh_requires_one_truthy_stack_item() {
    assert!(!p2wsh_ok(vec![OP_0], true));
    assert!(!p2wsh_ok(vec![OP_1, OP_1], true));
    assert!(p2wsh_ok(vec![OP_1], true));
}

#[test]
fn p2wsh_fast_path_requires_one_truthy_stack_item() {
    assert!(!p2wsh_ok(vec![OP_0], false));
    assert!(!p2wsh_ok(vec![OP_1, OP_1], false));
    assert!(p2wsh_ok(vec![OP_1], false));
}

#[test]
fn tapscript_requires_one_truthy_stack_item() {
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    assert!(!spend_tapscript(vec![OP_1, OP_1], vec![], flags));
    assert!(!spend_tapscript(vec![OP_0], vec![], flags));
    assert!(spend_tapscript(vec![OP_1], vec![], flags));
}

#[test]
fn tapscript_op_success_overrides_initial_stack_limits() {
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    let script = vec![0x50];
    let long = vec![vec![0x11; blvm_consensus::MAX_SCRIPT_ELEMENT_SIZE + 1]];
    assert!(spend_tapscript(script.clone(), long, flags));
    let many = vec![Vec::new(); blvm_consensus::MAX_STACK_SIZE + 1];
    assert!(spend_tapscript(script.clone(), many, flags));
    assert!(spend_tapscript(script, vec![vec![0x01]], flags));
}
