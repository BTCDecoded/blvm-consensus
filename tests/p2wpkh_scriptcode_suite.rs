//! Regression: BIP143 scriptCode for P2WPKH must be the P2PKH expansion (25 bytes),
//! not the witness program (22 bytes). Wrong scriptCode was the root cause of mass
//! sort-merge failures from block 481824 (native and nested P2WPKH).

use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_CHECKSIG, OP_DUP, OP_EQUALVERIFY, OP_HASH160, PUSH_20_BYTES,
};
use blvm_consensus::script::flags::SCRIPT_VERIFY_WITNESS;
use blvm_consensus::script::verify_script_with_context;
use blvm_consensus::transaction_hash::{
    calculate_bip143_sighash, derive_bip143_script_code_p2wpkh,
};
use blvm_consensus::types::Network;
use blvm_consensus::{
    OutPoint, SEGWIT_ACTIVATION_MAINNET, Transaction, TransactionInput, TransactionOutput,
};
use ripemd::Ripemd160;
use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};
use sha2::{Digest, Sha256};

fn push_bytes(script: &mut Vec<u8>, data: &[u8]) {
    let len = data.len();
    assert!(len <= 75);
    script.push(len as u8);
    script.extend_from_slice(data);
}

fn implicit_p2pkh_scriptcode(pubkey_hash: &[u8; 20]) -> [u8; 25] {
    let mut code = [0u8; 25];
    code[0] = OP_DUP;
    code[1] = OP_HASH160;
    code[2] = PUSH_20_BYTES;
    code[3..23].copy_from_slice(pubkey_hash);
    code[23] = OP_EQUALVERIFY;
    code[24] = OP_CHECKSIG;
    code
}

fn p2wpkh_script_pubkey(pubkey_hash: &[u8; 20]) -> Vec<u8> {
    let mut spk = vec![OP_0, PUSH_20_BYTES];
    spk.extend_from_slice(pubkey_hash);
    spk
}

fn pubkey_hash160(compressed_pubkey: &[u8]) -> [u8; 20] {
    let sha256_hash = Sha256::digest(compressed_pubkey);
    Ripemd160::digest(sha256_hash).into()
}

fn build_native_p2wpkh_spend() -> (
    Transaction,
    Vec<u8>,
    blvm_consensus::witness::Witness,
    Vec<TransactionOutput>,
) {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(&[0x22; 32]).expect("valid test key");
    let pubkey = PublicKey::from_secret_key(&secp, &secret_key);
    let compressed = pubkey.serialize();
    let pubkey_hash = pubkey_hash160(&compressed);
    let script_pubkey = p2wpkh_script_pubkey(&pubkey_hash);

    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x33; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let prevouts = vec![TransactionOutput {
        value: 10_000,
        script_pubkey: script_pubkey.clone().into(),
    }];

    let scriptcode = implicit_p2pkh_scriptcode(&pubkey_hash);
    let sighash = calculate_bip143_sighash(&tx, 0, &scriptcode, prevouts[0].value, 0x01, None)
        .expect("BIP143 sighash");
    let msg = Message::from_digest_slice(&sighash).expect("32-byte sighash");
    let sig = secp.sign_ecdsa(&msg, &secret_key);
    let mut sig_bytes = sig.serialize_der().to_vec();
    sig_bytes.push(0x01);

    let witness = vec![sig_bytes, compressed.to_vec()];
    (tx, script_pubkey, witness, prevouts)
}

#[test]
fn test_native_p2wpkh_valid_witness_succeeds() {
    let (tx, script_pubkey, witness, prevouts) = build_native_p2wpkh_spend();
    let flags = SCRIPT_VERIFY_WITNESS;

    assert!(
        verify_script_with_context(
            &tx.inputs[0].script_sig,
            &script_pubkey,
            Some(&witness),
            flags,
            &tx,
            0,
            &prevouts,
            Some(SEGWIT_ACTIVATION_MAINNET),
            Network::Mainnet,
        )
        .unwrap(),
        "native P2WPKH with correct BIP143 scriptCode must verify"
    );
}

#[test]
fn test_native_p2wpkh_wrong_scriptcode_signature_fails() {
    let (tx, script_pubkey, mut witness, prevouts) = build_native_p2wpkh_spend();

    // Re-sign using the 22-byte witness program as scriptCode (the pre-fix bug).
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(&[0x22; 32]).expect("valid test key");
    let wrong_scriptcode = script_pubkey.clone();
    let sighash =
        calculate_bip143_sighash(&tx, 0, &wrong_scriptcode, prevouts[0].value, 0x01, None)
            .expect("wrong sighash");
    let msg = Message::from_digest_slice(&sighash).expect("32-byte sighash");
    let sig = secp.sign_ecdsa(&msg, &secret_key);
    let mut sig_bytes = sig.serialize_der().to_vec();
    sig_bytes.push(0x01);
    witness[0] = sig_bytes;

    let flags = SCRIPT_VERIFY_WITNESS;
    assert!(
        !verify_script_with_context(
            &tx.inputs[0].script_sig,
            &script_pubkey,
            Some(&witness),
            flags,
            &tx,
            0,
            &prevouts,
            Some(SEGWIT_ACTIVATION_MAINNET),
            Network::Mainnet,
        )
        .unwrap(),
        "signature over witness-program scriptCode must not verify under fixed BLVM"
    );
}

#[test]
fn test_bip143_p2wpkh_scriptcode_length_is_25_not_22() {
    let pubkey_hash = [0xab; 20];
    let program = p2wpkh_script_pubkey(&pubkey_hash);
    let scriptcode = implicit_p2pkh_scriptcode(&pubkey_hash);
    assert_eq!(program.len(), 22, "witness program is 22 bytes");
    assert_eq!(scriptcode.len(), 25, "BIP143 scriptCode is 25 bytes");
    assert_ne!(program.as_slice(), scriptcode.as_slice());
    let derived = derive_bip143_script_code_p2wpkh(&pubkey_hash);
    assert_eq!(derived.as_slice(), scriptcode.as_slice());
}
