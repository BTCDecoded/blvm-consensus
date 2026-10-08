//! CHECKMULTISIG only verifies the top `m` signatures.
//!
//! A push above that window is the signature the script engine checks.
//! A witness stack with an extra item under the dummy fails the one-element
//! stack rule. Both verifiers have to agree.

#![cfg(feature = "production")]

use bitcoinconsensus::{
    Error, Utxo, VERIFY_DERSIG, VERIFY_NULLDUMMY, VERIFY_P2SH, VERIFY_WITNESS, verify_with_flags,
};
use blvm_consensus::crypto::OptimizedSha256;
use blvm_consensus::opcodes::{OP_0, OP_1, OP_2, OP_CHECKMULTISIG, PUSH_33_BYTES};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_DERSIG, SCRIPT_VERIFY_NULLDUMMY, SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_WITNESS,
};
use blvm_consensus::script::{SigVersion, verify_script_with_context_full};
use blvm_consensus::serialization::transaction::{
    serialize_transaction, serialize_transaction_with_witness,
};
use blvm_consensus::transaction_hash::{calculate_bip143_sighash, compute_legacy_sighash_nocache};
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};
use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

const FLAGS: u32 =
    SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_NULLDUMMY;
const VALUE: i64 = 100_000;
const HEIGHT: u64 = 500_000;

fn keys() -> (Secp256k1<secp256k1::All>, SecretKey, Vec<u8>, Vec<u8>) {
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x51; 32]).expect("secret");
    let pk1 = PublicKey::from_secret_key(&secp, &sk).serialize().to_vec();
    let pk2 =
        PublicKey::from_secret_key(&secp, &SecretKey::from_slice(&[0x52; 32]).expect("secret"))
            .serialize()
            .to_vec();
    (secp, sk, pk1, pk2)
}

fn redeem(pk1: &[u8], pk2: &[u8]) -> Vec<u8> {
    let mut script = vec![OP_1, PUSH_33_BYTES];
    script.extend_from_slice(pk1);
    script.push(PUSH_33_BYTES);
    script.extend_from_slice(pk2);
    script.push(OP_2);
    script.push(OP_CHECKMULTISIG);
    script
}

fn sign(secp: &Secp256k1<secp256k1::All>, sk: &SecretKey, sighash: &[u8; 32]) -> Vec<u8> {
    let mut sig = secp
        .sign_ecdsa(&Message::from_digest_slice(sighash).expect("digest"), sk)
        .serialize_der()
        .to_vec();
    sig.push(0x01);
    sig
}

fn push(buf: &mut Vec<u8>, data: &[u8]) {
    assert!(data.len() < 76);
    buf.push(data.len() as u8);
    buf.extend_from_slice(data);
}

fn tx_with_script(script_sig: Vec<u8>) -> Transaction {
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x44; 32],
                index: 0,
            },
            script_sig: script_sig.into(),
            sequence: 0xffff_ffff,
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

fn library_accepts(raw: &[u8], script_pubkey: &[u8]) -> bool {
    let spent = [Utxo {
        script_pubkey: script_pubkey.as_ptr(),
        script_pubkey_len: script_pubkey.len() as u32,
        value: VALUE,
    }];
    match verify_with_flags(script_pubkey, VALUE as u64, raw, Some(&spent), 0, FLAGS) {
        Ok(()) => true,
        Err(Error::ERR_SCRIPT) => false,
        Err(err) => panic!("script library api error: {err}"),
    }
}

fn blvm_accepts(tx: &Transaction, script_pubkey: &[u8], witness: Option<&Vec<Vec<u8>>>) -> bool {
    let prevout_values = [VALUE];
    let prevouts: [&[u8]; 1] = [script_pubkey];
    matches!(
        verify_script_with_context_full(
            &tx.inputs[0].script_sig,
            script_pubkey,
            witness,
            FLAGS,
            tx,
            0,
            &prevout_values,
            &prevouts,
            Some(HEIGHT),
            None,
            Network::Mainnet,
            SigVersion::Base,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        Ok(true)
    )
}

#[test]
fn multisig_window_matches_script_library() {
    assert_eq!(SCRIPT_VERIFY_P2SH, VERIFY_P2SH);
    assert_eq!(SCRIPT_VERIFY_WITNESS, VERIFY_WITNESS);
    assert_eq!(SCRIPT_VERIFY_DERSIG, VERIFY_DERSIG);
    assert_eq!(SCRIPT_VERIFY_NULLDUMMY, VERIFY_NULLDUMMY);

    let (secp, sk, pk1, pk2) = keys();
    let script = redeem(&pk1, &pk2);

    let bare = tx_with_script(Vec::new());
    let legacy = compute_legacy_sighash_nocache(&bare, 0, &script, 0x01);
    let sig = sign(&secp, &sk, &legacy);

    let mut exact_sig = vec![OP_0];
    push(&mut exact_sig, &sig);
    let exact = tx_with_script(exact_sig);
    let exact_raw = serialize_transaction(&exact);
    assert_eq!(
        blvm_accepts(&exact, &script, None),
        library_accepts(&exact_raw, &script),
        "bare 1-of-2"
    );
    assert!(library_accepts(&exact_raw, &script));

    let mut extra_sig = vec![OP_0];
    push(&mut extra_sig, &sig);
    push(&mut extra_sig, &[0x22; 9]);
    let extra = tx_with_script(extra_sig);
    let extra_raw = serialize_transaction(&extra);
    assert_eq!(
        blvm_accepts(&extra, &script, None),
        library_accepts(&extra_raw, &script),
        "push above the signature window"
    );
    assert!(!library_accepts(&extra_raw, &script));

    let program = OptimizedSha256::new().hash(&script);
    let mut program_script = vec![OP_0, 0x20];
    program_script.extend_from_slice(&program);
    let witness_tx = tx_with_script(Vec::new());
    let bip143 =
        calculate_bip143_sighash(&witness_tx, 0, &script, VALUE, 0x01, None).expect("bip143");
    let witness_sig = sign(&secp, &sk, &bip143);

    let exact_witness = vec![vec![], witness_sig.clone(), script.clone()];
    let exact_witness_raw =
        serialize_transaction_with_witness(&witness_tx, std::slice::from_ref(&exact_witness));
    assert_eq!(
        blvm_accepts(&witness_tx, &program_script, Some(&exact_witness)),
        library_accepts(&exact_witness_raw, &program_script),
        "p2wsh 1-of-2"
    );
    assert!(library_accepts(&exact_witness_raw, &program_script));

    let extra_witness = vec![vec![], vec![], witness_sig, script];
    let extra_witness_raw =
        serialize_transaction_with_witness(&witness_tx, std::slice::from_ref(&extra_witness));
    assert_eq!(
        blvm_accepts(&witness_tx, &program_script, Some(&extra_witness)),
        library_accepts(&extra_witness_raw, &program_script),
        "extra witness item under the dummy"
    );
    assert!(!library_accepts(&extra_witness_raw, &program_script));
}
