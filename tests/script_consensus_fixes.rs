//! Script consensus divergences from Bitcoin Core block validation.

#![cfg(feature = "production")]

use blvm_consensus::TAPROOT_ACTIVATION_MAINNET;
use blvm_consensus::activation::ForkActivationTable;
use blvm_consensus::block::get_block_script_verify_flags_core;
use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_ADD, OP_CAT, OP_CHECKLOCKTIMEVERIFY, OP_CHECKMULTISIG, OP_CHECKSEQUENCEVERIFY,
    OP_CHECKSIG, OP_CHECKSIGFROMSTACK, OP_DROP, OP_DUP, OP_ELSE, OP_ENDIF, OP_EQUAL, OP_HASH160,
    OP_IF, OP_NOP, OP_PUSHDATA2, OP_VER, OP_VERIF, PUSH_32_BYTES,
};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY, SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
    SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_TAPROOT_VERSION, SCRIPT_VERIFY_NULLDUMMY,
    SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_TAPROOT, SCRIPT_VERIFY_WITNESS,
    SCRIPT_VERIFY_WITNESS_PUBKEYTYPE,
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

fn verify_result(
    tx: &Transaction,
    script_pubkey: &[u8],
    witness: Option<&blvm_consensus::witness::Witness>,
    flags: u32,
    prevout_value: i64,
    height: Option<u64>,
) -> blvm_consensus::error::Result<bool> {
    disable_fast_paths(true);
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
    result
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

fn dead_branch(opcode: u8) -> Vec<u8> {
    vec![OP_0, OP_IF, opcode, OP_ENDIF, OP_1]
}

fn pushdata2(len: usize, byte: u8) -> Vec<u8> {
    let mut script = vec![OP_0, OP_IF, OP_PUSHDATA2];
    script.extend_from_slice(&(len as u16).to_le_bytes());
    script.extend(std::iter::repeat(byte).take(len));
    script.extend_from_slice(&[OP_ENDIF, OP_1]);
    script
}

#[test]
fn dead_branch_rejects_disabled_opcode_and_verif() {
    assert!(!p2wsh_ok(dead_branch(OP_CAT), true));
    assert!(p2wsh_ok(dead_branch(OP_NOP), true));
    assert!(p2wsh_ok(dead_branch(OP_ADD), true));
    assert!(!p2wsh_ok(dead_branch(OP_VERIF), true));
}

#[test]
fn legacy_dead_branch_keeps_ver_and_rejects_cat() {
    let cat = one_input_tx(vec![]);
    let cat_script = dead_branch(OP_CAT);
    assert!(!verify(&cat, &cat_script, None, 0, 10_000, Some(100)));
    let ver = one_input_tx(vec![]);
    let ver_script = dead_branch(OP_VER);
    assert!(verify(&ver, &ver_script, None, 0, 10_000, Some(100)));
}

#[test]
fn dead_branch_rejects_oversized_push() {
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    let too_big = pushdata2(blvm_consensus::MAX_SCRIPT_ELEMENT_SIZE + 1, 0x11);
    let small = pushdata2(1, 0x11);
    assert!(!p2wsh_ok(too_big.clone(), true));
    assert!(p2wsh_ok(small, true));
    assert!(!spend_tapscript(too_big, vec![], flags));
}

fn locktime_tx(lock_time: u32, sequence: u32) -> Transaction {
    let mut tx = one_input_tx(vec![]);
    tx.lock_time = lock_time.into();
    tx.inputs[0].sequence = sequence.into();
    tx
}

#[test]
fn cltv_and_csv_use_script_number_sign() {
    let height = blvm_consensus::BIP65_ACTIVATION_MAINNET;
    let neg = vec![0x01, 0x81, OP_CHECKLOCKTIMEVERIFY, OP_1];
    let tx = locktime_tx(200, 0);
    assert!(!verify(
        &tx,
        &neg,
        None,
        SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY,
        0,
        Some(height)
    ));
    let neg_zero = vec![0x01, 0x80, OP_CHECKLOCKTIMEVERIFY, OP_1];
    let tx = locktime_tx(100, 0);
    assert!(verify(
        &tx,
        &neg_zero,
        None,
        SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY,
        0,
        Some(height)
    ));
    let five = vec![
        0x05,
        0x01,
        0x00,
        0x00,
        0x00,
        0x80,
        OP_CHECKLOCKTIMEVERIFY,
        OP_1,
    ];
    let tx = locktime_tx(200, 0);
    assert!(!verify(
        &tx,
        &five,
        None,
        SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY,
        0,
        Some(height)
    ));
    let too_big = vec![
        0x05,
        0x00,
        0x00,
        0x00,
        0x00,
        0x01,
        OP_CHECKLOCKTIMEVERIFY,
        OP_1,
    ];
    assert!(!verify(
        &tx,
        &too_big,
        None,
        SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY,
        0,
        Some(height)
    ));

    let csv_neg = vec![0x01, 0x81, OP_CHECKSEQUENCEVERIFY, OP_1];
    let tx = locktime_tx(0, 200);
    assert!(!verify(
        &tx,
        &csv_neg,
        None,
        SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
        0,
        Some(height)
    ));
    let csv_zero = vec![0x01, 0x80, OP_CHECKSEQUENCEVERIFY, OP_1];
    assert!(verify(
        &tx,
        &csv_zero,
        None,
        SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
        0,
        Some(height)
    ));
}

#[test]
fn csv_requires_version_two_unless_disabled() {
    let height = blvm_consensus::BIP112_CSV_ACTIVATION_MAINNET;
    let script = vec![OP_0, OP_CHECKSEQUENCEVERIFY, OP_1];
    let mut v1 = locktime_tx(0, 0);
    v1.version = 1;
    assert!(!verify(
        &v1,
        &script,
        None,
        SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
        0,
        Some(height)
    ));
    let v2 = locktime_tx(0, 0);
    assert!(verify(
        &v2,
        &script,
        None,
        SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
        0,
        Some(height)
    ));
    // Disable bit on the stack is a no-op even when the version is 1.
    let disabled = vec![
        0x05,
        0x00,
        0x00,
        0x00,
        0x80,
        0x00,
        OP_CHECKSEQUENCEVERIFY,
        OP_1,
    ];
    assert!(verify(
        &v1,
        &disabled,
        None,
        SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
        0,
        Some(height)
    ));
}

#[test]
fn nulldummy_rejects_one_byte_zero() {
    let flags = SCRIPT_VERIFY_NULLDUMMY;
    let height = blvm_consensus::BIP147_ACTIVATION_MAINNET;
    let empty = vec![OP_0, OP_0, OP_0, OP_CHECKMULTISIG];
    let tx = one_input_tx(vec![]);
    assert!(verify(&tx, &empty, None, flags, 0, Some(height)));
    let one_byte = vec![0x01, 0x00, OP_0, OP_0, OP_CHECKMULTISIG];
    let tx = one_input_tx(vec![]);
    assert!(!verify(&tx, &one_byte, None, flags, 0, Some(height)));
    assert!(verify(&tx, &one_byte, None, flags, 0, Some(height - 1)));
}

#[test]
fn p2wsh_nulldummy_rejects_one_byte_zero() {
    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&[0x22; 32]).expect("key");
    let pubkey = PublicKey::from_secret_key(&secp, &secret).serialize();
    let mut script = vec![OP_1, 33];
    script.extend_from_slice(&pubkey);
    script.extend_from_slice(&[OP_1, OP_CHECKMULTISIG]);
    let tx = one_input_tx(vec![]);
    let sighash = calculate_bip143_sighash(&tx, 0, &script, 10_000, 0x01, None).unwrap();
    let msg = Message::from_digest_slice(&sighash).unwrap();
    let mut sig = secp.sign_ecdsa(&msg, &secret).serialize_der().to_vec();
    sig.push(0x01);
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_NULLDUMMY;
    let height = blvm_consensus::BIP147_ACTIVATION_MAINNET;
    let (tx, spk, witness) = p2wsh(script.clone(), vec![vec![], sig.clone()]);
    assert!(verify_at(
        &tx,
        &spk,
        Some(&witness),
        flags,
        10_000,
        Some(height),
        false,
    ));
    let (tx, spk, witness) = p2wsh(script, vec![vec![0x00], sig]);
    assert!(!verify_at(
        &tx,
        &spk,
        Some(&witness),
        flags,
        10_000,
        Some(height),
        false,
    ));
    assert!(verify_at(
        &tx,
        &spk,
        Some(&witness),
        flags,
        10_000,
        Some(height - 1),
        false,
    ));
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

#[test]
fn stack_limit_is_checked_after_each_opcode() {
    use blvm_consensus::MAX_STACK_SIZE;

    let exact = one_input_tx(vec![OP_1; MAX_STACK_SIZE]);
    assert!(verify(&exact, &[], None, 0, 0, None));

    let over = one_input_tx(vec![OP_1; MAX_STACK_SIZE + 1]);
    assert!(!verify(&over, &[], None, 0, 0, None));

    let mut duped = vec![OP_1; MAX_STACK_SIZE];
    duped.push(OP_DUP);
    assert!(!verify(&one_input_tx(duped), &[], None, 0, 0, None));

    let mut pushed = Vec::new();
    for _ in 0..=MAX_STACK_SIZE {
        pushed.extend_from_slice(&[0x01, 0x11]);
    }
    assert!(!verify(&one_input_tx(pushed), &[], None, 0, 0, None));

    // One pop brings 1001 items back to the limit. The spend still fails because
    // more than one item remains, but it is not a stack-limit error.
    let dropped = vec![vec![0x01]; MAX_STACK_SIZE + 1];
    let (tx, spk, witness) = p2wsh(vec![OP_DROP], dropped);
    assert!(matches!(
        verify_result(
            &tx,
            &spk,
            Some(&witness),
            SCRIPT_VERIFY_WITNESS,
            10_000,
            Some(800_000),
        ),
        Ok(false)
    ));

    let still_over = vec![vec![0x01]; MAX_STACK_SIZE + 1];
    let (tx, spk, witness) = p2wsh(vec![OP_NOP], still_over);
    assert!(matches!(
        verify_result(
            &tx,
            &spk,
            Some(&witness),
            SCRIPT_VERIFY_WITNESS,
            10_000,
            Some(800_000),
        ),
        Err(blvm_consensus::error::ConsensusError::ScriptErrorWithCode { .. })
    ));

    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    assert!(!spend_tapscript(
        vec![OP_DROP, OP_1],
        vec![Vec::new(); MAX_STACK_SIZE + 1],
        flags,
    ));
}

fn v0_program(len: usize) -> Vec<u8> {
    let mut script = vec![OP_0, len as u8];
    script.extend(std::iter::repeat_n(0x11, len));
    script
}

#[test]
fn witness_v0_wrong_program_length_fails() {
    let program = v0_program(2);
    let tx = one_input_tx(vec![]);
    assert!(!verify(
        &tx,
        &program,
        None,
        SCRIPT_VERIFY_WITNESS,
        0,
        Some(800_000),
    ));
    assert!(verify(&tx, &program, None, 0, 0, Some(800_000)));

    let future = vec![OP_1, 0x02, 0x4e, 0x73];
    assert!(verify(
        &one_input_tx(vec![]),
        &future,
        None,
        SCRIPT_VERIFY_WITNESS,
        0,
        Some(800_000),
    ));

    let redeem = v0_program(21);
    let mut script_sig = vec![redeem.len() as u8];
    script_sig.extend_from_slice(&redeem);
    let hash = Ripemd160::digest(Sha256::digest(&redeem));
    let mut spk = vec![OP_HASH160, 0x14];
    spk.extend_from_slice(&hash);
    spk.push(OP_EQUAL);
    let flags = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS;
    assert!(!verify(
        &one_input_tx(script_sig.clone()),
        &spk,
        None,
        flags,
        0,
        Some(800_000),
    ));
    assert!(verify(
        &one_input_tx(script_sig),
        &spk,
        None,
        SCRIPT_VERIFY_P2SH,
        0,
        Some(800_000),
    ));
}

fn p2sh_of(redeem: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut script_sig = vec![redeem.len() as u8];
    script_sig.extend_from_slice(redeem);
    let hash = Ripemd160::digest(Sha256::digest(redeem));
    let mut spk = vec![OP_HASH160, 0x14];
    spk.extend_from_slice(&hash);
    spk.push(OP_EQUAL);
    (script_sig, spk)
}

#[test]
fn nested_v0_without_witness_fails() {
    let flags = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS;
    let empty: Vec<Vec<u8>> = Vec::new();
    for len in [20usize, 32] {
        let (script_sig, spk) = p2sh_of(&v0_program(len));
        assert!(!verify(
            &one_input_tx(script_sig.clone()),
            &spk,
            None,
            flags,
            0,
            Some(800_000),
        ));
        assert!(!verify(
            &one_input_tx(script_sig.clone()),
            &spk,
            Some(&empty),
            flags,
            0,
            Some(800_000),
        ));
        assert!(verify(
            &one_input_tx(script_sig),
            &spk,
            None,
            SCRIPT_VERIFY_P2SH,
            0,
            Some(800_000),
        ));
    }
}

#[test]
fn taproot_control_block_rejects_more_than_128_nodes() {
    let flags = SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT;
    let internal = internal_key();
    let tapscript = vec![OP_1];
    for n in [128usize, 129] {
        let proof: Vec<[u8; 32]> = (0..n)
            .map(|i| {
                let mut node = [0u8; 32];
                node[0] = i as u8;
                node[1] = (i >> 8) as u8;
                node
            })
            .collect();
        let root = compute_script_merkle_root(&tapscript, &proof, TAPROOT_LEAF_VERSION_TAPSCRIPT)
            .expect("root");
        let (output_key, parity) =
            blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(&internal, &root)
                .expect("tweak");
        let mut spk = vec![OP_1, PUSH_32_BYTES];
        spk.extend_from_slice(&output_key);
        let mut control = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
        control.extend_from_slice(&internal);
        for node in &proof {
            control.extend_from_slice(node);
        }
        let witness = vec![tapscript.clone(), control];
        let ok = verify(
            &one_input_tx(vec![]),
            &spk,
            Some(&witness),
            flags,
            10_000,
            Some(TAPROOT_ACTIVATION_MAINNET),
        );
        if n == 128 {
            assert!(ok);
        } else {
            assert!(!ok);
        }
    }
}

#[test]
fn checksigfromstack_fails_outside_tapscript() {
    let script = vec![OP_1, OP_CHECKSIGFROMSTACK];
    assert!(!verify(
        &one_input_tx(vec![]),
        &script,
        None,
        0,
        0,
        Some(800_000),
    ));
    assert!(!p2wsh_ok(script, true));
    let skipped = vec![OP_0, OP_IF, OP_CHECKSIGFROMSTACK, OP_ENDIF, OP_1];
    assert!(verify(
        &one_input_tx(vec![]),
        &skipped,
        None,
        0,
        0,
        Some(800_000),
    ));
    assert!(spend_tapscript(
        vec![OP_CHECKSIGFROMSTACK],
        vec![],
        SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT,
    ));
}
