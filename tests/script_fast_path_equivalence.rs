//! P4.2: script fast paths ≡ full VerifyScript interpreter (production builds).

#![cfg(feature = "production")]

#[path = "core_test_vectors/script_asm.rs"]
mod script_asm;

use blvm_consensus::opcodes::PUSH_20_BYTES;
use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_CHECKSIG, OP_DUP, OP_ENDIF, OP_EQUAL, OP_EQUALVERIFY, OP_HASH160, OP_IF,
    PUSH_32_BYTES,
};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_TAPROOT, SCRIPT_VERIFY_WITNESS,
};
use blvm_consensus::script::{
    SigVersion, disable_fast_paths, verify_script, verify_script_with_context_full,
};
use blvm_consensus::serialization::deserialize_transaction_with_witness;
use blvm_consensus::transaction_hash::{
    SighashType, calculate_bip143_sighash, calculate_transaction_sighash,
    compute_legacy_sighash_nocache,
};
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};
use blvm_consensus::witness::is_witness_empty;
use blvm_consensus::{ByteString, SEGWIT_ACTIVATION_MAINNET};
use hex;
use ripemd::Ripemd160;
use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};
use sha2::{Digest, Sha256};

fn push_bytes(script: &mut Vec<u8>, data: &[u8]) {
    let len = data.len();
    assert!(len <= 75, "test helper only supports pushes up to 75 bytes");
    script.push(len as u8);
    script.extend_from_slice(data);
}

fn pubkey_hash160(compressed_pubkey: &[u8]) -> [u8; 20] {
    let sha256_hash = Sha256::digest(compressed_pubkey);
    Ripemd160::digest(sha256_hash).into()
}

fn implicit_p2pkh_scriptcode(pubkey_hash: &[u8; 20]) -> Vec<u8> {
    let mut scriptcode = Vec::with_capacity(25);
    scriptcode.push(OP_DUP);
    scriptcode.push(OP_HASH160);
    scriptcode.push(PUSH_20_BYTES);
    scriptcode.extend_from_slice(pubkey_hash);
    scriptcode.push(OP_EQUALVERIFY);
    scriptcode.push(OP_CHECKSIG);
    scriptcode
}

fn p2sh_scriptpubkey(redeem_script: &[u8]) -> Vec<u8> {
    use bitcoin_hashes::Hash as BitcoinHash;
    use bitcoin_hashes::hash160;
    let h = hash160::Hash::hash(redeem_script);
    let mut spk = vec![OP_HASH160, PUSH_20_BYTES];
    spk.extend_from_slice(h.as_ref());
    spk.push(OP_EQUAL);
    spk
}

fn assert_fast_path_equiv(
    script_sig: &ByteString,
    script_pubkey: &[u8],
    witness: Option<&blvm_consensus::witness::Witness>,
    flags: u32,
    tx: &Transaction,
    input_index: usize,
    prevout_values: &[i64],
    prevout_script_pubkeys: &[&[u8]],
    block_height: Option<u64>,
) {
    disable_fast_paths(false);
    let fast = verify_script_with_context_full(
        script_sig,
        script_pubkey,
        witness,
        flags,
        tx,
        input_index,
        prevout_values,
        prevout_script_pubkeys,
        block_height,
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
    )
    .expect("fast path verify");

    disable_fast_paths(true);
    let full = verify_script_with_context_full(
        script_sig,
        script_pubkey,
        witness,
        flags,
        tx,
        input_index,
        prevout_values,
        prevout_script_pubkeys,
        block_height,
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
    )
    .expect("interpreter verify");

    disable_fast_paths(false);
    assert_eq!(
        fast, full,
        "input {input_index}: fast path {fast:?} != interpreter {full:?}"
    );
    assert!(
        fast,
        "equivalence harness expects passing scripts (Ok(true))"
    );
}

fn build_valid_p2pkh_spend() -> (Transaction, Vec<u8>, i64) {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(&[0x33; 32]).expect("valid test secret key");
    let pubkey = PublicKey::from_secret_key(&secp, &secret_key);
    let compressed = pubkey.serialize();
    let pubkey_hash = pubkey_hash160(&compressed);

    let mut script_pubkey = vec![OP_DUP, OP_HASH160, PUSH_20_BYTES];
    script_pubkey.extend_from_slice(&pubkey_hash);
    script_pubkey.extend_from_slice(&[OP_EQUALVERIFY, OP_CHECKSIG]);

    let prevout_value = 10_000i64;
    let prevouts = vec![TransactionOutput {
        value: prevout_value,
        script_pubkey: script_pubkey.clone().into(),
    }];

    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x44; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 9_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let sighash =
        calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::ALL).expect("legacy sighash");
    let msg = Message::from_digest_slice(sighash.as_ref()).expect("32-byte sighash");
    let sig = secp.sign_ecdsa(&msg, &secret_key);
    let mut der_with_type = sig.serialize_der().to_vec();
    der_with_type.push(0x01);
    let mut script_sig = Vec::new();
    push_bytes(&mut script_sig, &der_with_type);
    push_bytes(&mut script_sig, &compressed);

    let mut signed = tx.clone();
    signed.inputs[0].script_sig = script_sig.clone().into();

    (signed, script_pubkey, prevout_value)
}

fn build_valid_p2pk_spend() -> (Transaction, Vec<u8>, i64) {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(&[0x34; 32]).expect("valid test secret key");
    let pubkey = PublicKey::from_secret_key(&secp, &secret_key);
    let compressed = pubkey.serialize();

    let mut script_pubkey = vec![0x21];
    script_pubkey.extend_from_slice(&compressed);
    script_pubkey.push(OP_CHECKSIG);

    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x46; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 8_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let sighash = compute_legacy_sighash_nocache(&tx, 0, &script_pubkey, 0x01);
    let msg = Message::from_digest_slice(&sighash).expect("32-byte sighash");
    let sig = secp.sign_ecdsa(&msg, &secret_key);
    let mut der_with_type = sig.serialize_der().to_vec();
    der_with_type.push(0x01);
    let mut script_sig = Vec::new();
    push_bytes(&mut script_sig, &der_with_type);

    let mut signed = tx.clone();
    signed.inputs[0].script_sig = script_sig.clone().into();

    (signed, script_pubkey, 9_000)
}

fn build_native_p2wpkh_spend() -> (Transaction, Vec<u8>, blvm_consensus::witness::Witness, i64) {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(&[0x22; 32]).expect("valid test key");
    let pubkey = PublicKey::from_secret_key(&secp, &secret_key);
    let compressed = pubkey.serialize();
    let pubkey_hash = pubkey_hash160(&compressed);

    let mut script_pubkey = vec![OP_0, PUSH_20_BYTES];
    script_pubkey.extend_from_slice(&pubkey_hash);

    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x55; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let prevout_value = 10_000i64;
    let scriptcode = implicit_p2pkh_scriptcode(&pubkey_hash);
    let sighash =
        calculate_bip143_sighash(&tx, 0, &scriptcode, prevout_value, 0x01, None).expect("sighash");
    let msg = Message::from_digest_slice(&sighash).expect("32-byte sighash");
    let sig = secp.sign_ecdsa(&msg, &secret_key);
    let mut sig_bytes = sig.serialize_der().to_vec();
    sig_bytes.push(0x01);
    let witness = vec![sig_bytes, compressed.to_vec()];

    (tx, script_pubkey, witness, prevout_value)
}

fn build_p2wpkh_in_p2sh_spend() -> (Transaction, Vec<u8>, blvm_consensus::witness::Witness, i64) {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(&[0x11; 32]).expect("valid test secret key");
    let pubkey = PublicKey::from_secret_key(&secp, &secret_key);
    let compressed = pubkey.serialize();
    let pubkey_hash = pubkey_hash160(&compressed);

    let mut redeem = vec![OP_0, PUSH_20_BYTES];
    redeem.extend_from_slice(&pubkey_hash);
    let script_pubkey = p2sh_scriptpubkey(&redeem);

    let mut script_sig = Vec::new();
    push_bytes(&mut script_sig, &redeem);

    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x48; 32],
                index: 0,
            },
            script_sig: script_sig.clone().into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 9_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let prevout_value = 10_000i64;
    let scriptcode = implicit_p2pkh_scriptcode(&pubkey_hash);
    let sighash =
        calculate_bip143_sighash(&tx, 0, &scriptcode, prevout_value, 0x01, None).expect("sighash");
    let msg = Message::from_digest_slice(&sighash).expect("32-byte sighash");
    let sig = secp.sign_ecdsa(&msg, &secret_key);
    let mut sig_bytes = sig.serialize_der().to_vec();
    sig_bytes.push(0x01);
    let witness = vec![sig_bytes, compressed.to_vec()];

    (tx, script_pubkey, witness, prevout_value)
}

#[test]
fn fast_path_equiv_simple_verify_script() {
    let script_sig: ByteString = vec![OP_1].into();
    let script_pubkey: ByteString = vec![OP_1].into();
    disable_fast_paths(false);
    let a = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
    disable_fast_paths(true);
    let b = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
    disable_fast_paths(false);
    assert_eq!(a, b);
}

#[test]
fn fast_path_equiv_invalid_p2pkh_sig_agrees_on_false() {
    let pubkey_hash = [0xab; 20];
    let mut script_pubkey = vec![OP_DUP, OP_HASH160, PUSH_20_BYTES];
    script_pubkey.extend_from_slice(&pubkey_hash);
    script_pubkey.extend_from_slice(&[OP_EQUALVERIFY, OP_CHECKSIG]);

    let script_sig: ByteString = vec![0x30, 0x06, 0x02, 0x01, 0x00, 0x02, 0x01, 0x00].into();

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: script_sig.clone(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: script_pubkey.clone().into(),
        }]
        .into(),
        lock_time: 0,
    };

    let pv = vec![1_000_000i64];
    let psp: Vec<ByteString> = vec![script_pubkey.into()];
    let psp_refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();

    disable_fast_paths(false);
    let fast = verify_script_with_context_full(
        &script_sig,
        psp_refs[0],
        None,
        0,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(500_000),
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
    )
    .unwrap();
    disable_fast_paths(true);
    let full = verify_script_with_context_full(
        &script_sig,
        psp_refs[0],
        None,
        0,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(500_000),
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
    )
    .unwrap();
    disable_fast_paths(false);
    assert_eq!(fast, full);
    assert!(!fast);
}

#[test]
fn legacy_sighash_sign_verify_sanity() {
    use blvm_consensus::secp256k1_backend::verify_ecdsa_direct;

    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(&[0x33; 32]).expect("valid test secret key");
    let compressed = PublicKey::from_secret_key(&secp, &secret_key).serialize();
    let pubkey_hash = pubkey_hash160(&compressed);

    let mut script_pubkey = vec![OP_DUP, OP_HASH160, PUSH_20_BYTES];
    script_pubkey.extend_from_slice(&pubkey_hash);
    script_pubkey.extend_from_slice(&[OP_EQUALVERIFY, OP_CHECKSIG]);

    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x44; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 9_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let prevouts = vec![TransactionOutput {
        value: 10_000,
        script_pubkey: script_pubkey.clone().into(),
    }];
    let h_calc = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::ALL).unwrap();
    let h_nocache = compute_legacy_sighash_nocache(&tx, 0, &script_pubkey, 0x01);
    assert_eq!(h_nocache.as_ref(), h_calc.as_ref(), "sighash mismatch");

    let msg = Message::from_digest_slice(h_calc.as_ref()).expect("32-byte sighash");
    let sig = secp.sign_ecdsa(&msg, &secret_key);
    let der = sig.serialize_der();

    assert!(
        verify_ecdsa_direct(
            &der,
            &compressed,
            h_calc.as_ref().try_into().unwrap(),
            false,
            false
        )
        .unwrap(),
        "ECDSA must verify with blvm-secp256k1 backend"
    );
}

#[test]
fn fast_path_equiv_valid_p2pkh_succeeds() {
    let (tx, script_pubkey, prevout_value) = build_valid_p2pkh_spend();
    let pv = vec![prevout_value];
    let psp_refs: Vec<&[u8]> = vec![script_pubkey.as_slice()];

    assert_fast_path_equiv(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        None,
        0,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(200_000),
    );
}

#[test]
fn fast_path_equiv_valid_p2pk_succeeds() {
    let (tx, script_pubkey, prevout_value) = build_valid_p2pk_spend();
    let pv = vec![prevout_value];
    let psp_refs: Vec<&[u8]> = vec![script_pubkey.as_slice()];

    assert_fast_path_equiv(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        None,
        0,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(200_000),
    );
}

#[test]
fn fast_path_equiv_native_p2wpkh_succeeds() {
    let (tx, script_pubkey, witness, prevout_value) = build_native_p2wpkh_spend();
    let flags = SCRIPT_VERIFY_WITNESS;
    let pv = vec![prevout_value];
    let psp_refs: Vec<&[u8]> = vec![script_pubkey.as_slice()];

    assert_fast_path_equiv(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        Some(&witness),
        flags,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(SEGWIT_ACTIVATION_MAINNET),
    );
}

#[test]
fn fast_path_equiv_p2wpkh_in_p2sh_succeeds() {
    let (tx, script_pubkey, witness, prevout_value) = build_p2wpkh_in_p2sh_spend();
    let flags = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS;
    let pv = vec![prevout_value];
    let psp_refs: Vec<&[u8]> = vec![script_pubkey.as_slice()];

    assert_fast_path_equiv(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        Some(&witness),
        flags,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(SEGWIT_ACTIVATION_MAINNET),
    );
}

/// Core `tx_valid.json` BIP143 P2SH-P2WSH 6-of-6 multisig (exercises P2WPKH/P2WSH fast-path family).
#[test]
fn fast_path_equiv_bip143_core_p2sh_p2wsh_multisig() {
    const TX_HEX: &str = "0100000000010136641869ca081e70f394c6948e8af409e18b619df2ed74aa106c1ca29787b96e0100000023220020a16b5755f7f6f96dbd65f5f0d6ab9418b89af4b1f14a1bb8a09062c35f0dcb54ffffffff0200e9a435000000001976a914389ffce9cd9ae88dcc0631e88a821ffdbe9bfe2688acc0832f05000000001976a9147480a33f950689af511e6e84c138dbbd3c3ee41588ac080047304402206ac44d672dac41f9b00e28f4df20c52eeb087207e8d758d76d92c6fab3b73e2b0220367750dbbe19290069cba53d096f44530e4f98acaa594810388cf7409a1870ce01473044022068c7946a43232757cbdf9176f009a928e1cd9a1a8c212f15c1e11ac9f2925d9002205b75f937ff2f9f3c1246e547e54f62e027f64eefa2695578cc6432cdabce271502473044022059ebf56d98010a932cf8ecfec54c48e6139ed6adb0728c09cbe1e4fa0915302e022007cd986c8fa870ff5d2b3a89139c9fe7e499259875357e20fcbb15571c76795403483045022100fbefd94bd0a488d50b79102b5dad4ab6ced30c4069f1eaa69a4b5a763414067e02203156c6a5c9cf88f91265f5a942e96213afae16d83321c8b31bb342142a14d16381483045022100a5263ea0553ba89221984bd7f0b13613db16e7a70c549a86de0cc0444141a407022005c360ef0ae5a5d4f9f2f87a56c1546cc8268cab08c73501d6b3be2e1e1a8a08824730440220525406a1482936d5a21888260dc165497a90a15669636d8edca6b9fe490d309c022032af0c646a34a44d1f4576bf6a4a74b67940f8faa84c7df9abe12a01a11e2b4783cf56210307b8ae49ac90a048e9b53357a2354b3334e9c8bee813ecb98e99a7e07e8c3ba32103b28f0c28bfab54554ae8c658ac5c3e0ce6e79ad336331f78c428dd43eea8449b21034b8113d703413d57761b8b9781957b8c0ac1dfe69f492580ca4195f50376ba4a21033400f6afecb833092a9a21cfdf1ed1376e58c5d1f47de74683123987e967a8f42103a6d48b1131e94ba04d9737d61acdaa1322008af9602b3b14862c07a1789aac162102d8b661b0b3302ee2f162b09e07a55ad5dfbe673a9f01d9f0c19617681024306b56ae00000000";
    const PREVOUT_SCRIPT_ASM: &str =
        "HASH160 0x14 0x9993a429037b5d912407a71c252019287b8d27a5 EQUAL";
    const PREVOUT_VALUE: i64 = 987_654_321;

    let tx_bytes = hex::decode(TX_HEX).expect("valid tx hex");
    let (tx, witnesses, _) =
        deserialize_transaction_with_witness(&tx_bytes).expect("segwit tx deserialize");
    let prevout_script =
        script_asm::parse_core_script_asm(PREVOUT_SCRIPT_ASM).expect("parse prevout scriptPubKey");
    let pv = vec![PREVOUT_VALUE];
    let psp_refs: Vec<&[u8]> = vec![prevout_script.as_slice()];
    let flags = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS;

    assert_fast_path_equiv(
        &tx.inputs[0].script_sig,
        &prevout_script,
        Some(&witnesses[0]),
        flags,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(SEGWIT_ACTIVATION_MAINNET),
    );
}

/// Block 709635 P2TR key-path (65-byte witness) — regression surface for Taproot fast path.
#[test]
fn fast_path_equiv_block709635_p2tr_keypath_all_inputs() {
    const RAW_TX_HEX: &str = "\
020000000001041ee2529c53a3c05c1e35fa853b9209cbc1a17be31aae9f4e7ea42d13f24c65890000000000\
ffffffff1ee2529c53a3c05c1e35fa853b9209cbc1a17be31aae9f4e7ea42d13f24c65890100000000\
ffffffff1ee2529c53a3c05c1e35fa853b9209cbc1a17be31aae9f4e7ea42d13f24c65890200000000\
ffffffff1ee2529c53a3c05c1e35fa853b9209cbc1a17be31aae9f4e7ea42d13f24c65890300000000\
ffffffff01007ea60000000000225120a457d0c0399b499ed2df571d612ba549ae7f199387edceac175999210f6aa39d\
0141b23008b3e044d16078fc93ae4f342b6e5ba44241c598503f80269fd66e7ce484e926b2ff58ac5633be79857951b\
3dc778082fd38a9e06a1139e6eea41a8680c7010141be98ba2a47fce6fbe4f7456e5fe0c2381f38ed3ae3b89d0748\
fdbfc6936b68019e01ff60343abbea025138e58aed2544dc8d3c0b2ccb35e2073fa2f9feeff5ed010141466d525b977\
33d4733220694bf747fd6e9d4b0b96ea3b2fb06b7486b4b8e864df0057481a01cf10f7ea06849fb4717d62b902fe580\
7a1cba03a46bf3a7087e940101418dbfbdd2c164005eceb0de04c317b9cae62b0c97ed33da9dcec6301fa0517939b\
9024eba99e22098a5b0d86eb7218957883ea9fc13b737e1146ae2b95185fcf90100000000";

    const PREVOUT_SCRIPTS_HEX: [&str; 4] = [
        "51205f4237bd79e8fe440d102a5e0c20a75160e96d42a8b19825ac90f73f1f667768",
        "5120e914be846f7afb29f5c3b24e5f630886ed5cbcc79a28888d91009be90924508d",
        "5120d9390cafa11bdeb19de21e0a2bbd541f4d0979473999503408d40814399b7f91",
        "5120e8d645f42be8700595c7cbb278602fb51471d5bb24ccd27668321b7affd167bf",
    ];
    const PREVOUT_VALUES: [i64; 4] = [3_400_000, 3_410_000, 3_420_000, 709_632];
    const BLOCK_HEIGHT: u64 = 709_635;

    let raw: Vec<u8> = (0..RAW_TX_HEX.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&RAW_TX_HEX[i..i + 2], 16).unwrap())
        .collect();
    let (tx, witnesses, _) =
        deserialize_transaction_with_witness(&raw).expect("deserialize segwit tx");

    let prevout_scripts_owned: Vec<Vec<u8>> = PREVOUT_SCRIPTS_HEX
        .iter()
        .map(|s| {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect()
        })
        .collect();
    let flags = 0x01 | 0x04 | 0x10 | 0x200 | 0x400 | 0x800 | 0x8000 | 0x20000;
    let prevout_values: Vec<i64> = PREVOUT_VALUES.to_vec();
    let prevout_scripts: Vec<&[u8]> = prevout_scripts_owned.iter().map(|s| s.as_slice()).collect();

    for input_idx in 0..4 {
        let witness_stack = witnesses.get(input_idx).filter(|w| !is_witness_empty(w));
        assert!(
            witness_stack.is_some(),
            "input {input_idx} must have witness"
        );

        assert_fast_path_equiv(
            &tx.inputs[input_idx].script_sig,
            &prevout_scripts_owned[input_idx],
            witness_stack,
            flags,
            &tx,
            input_idx,
            &prevout_values,
            &prevout_scripts,
            Some(BLOCK_HEIGHT),
        );
    }
}

fn valid_taproot_internal_key() -> [u8; 32] {
    [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ]
}

fn envelope_body() -> Vec<u8> {
    vec![0x03, b'o', b'r', b'd']
}

fn envelope(body: &[u8]) -> Vec<u8> {
    let mut s = vec![OP_0, OP_IF];
    s.extend_from_slice(body);
    s.push(OP_ENDIF);
    s
}

fn push32_checksig(pk: &[u8; 32]) -> Vec<u8> {
    let mut s = vec![PUSH_32_BYTES];
    s.extend_from_slice(pk);
    s.push(OP_CHECKSIG);
    s
}

/// Build a signed P2TR script-path spend. `stack_items == 1` (the sig).
fn build_unexec_if_spend(
    tapscript: Vec<u8>,
    annex: Option<Vec<u8>>,
) -> (Transaction, Vec<u8>, blvm_consensus::witness::Witness, i64) {
    use blvm_consensus::taproot::{
        TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root,
        compute_tapscript_signature_hash,
    };
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(&[0x42; 32]).expect("leaf secret");
    let keypair = Keypair::from_secret_key(&secp, &secret_key);
    let (xonly, _) = keypair.x_only_public_key();
    let leaf_pk = xonly.serialize();
    assert!(
        tapscript.windows(32).any(|w| w == leaf_pk),
        "tapscript must contain the leaf x-only pubkey"
    );

    let internal = valid_taproot_internal_key();
    let merkle_root =
        compute_script_merkle_root(&tapscript, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).expect("root");
    let (output_key, parity) =
        blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(&internal, &merkle_root)
            .expect("tweak");
    let mut script_pubkey = vec![OP_1, PUSH_32_BYTES];
    script_pubkey.extend_from_slice(&output_key);

    let mut control_block = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control_block.extend_from_slice(&internal);

    let prevout_value = 10_000i64;
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x71; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 9_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let mut placeholder = vec![vec![0u8; 64], tapscript.clone(), control_block.clone()];
    if let Some(ref a) = annex {
        placeholder.push(a.clone());
    }
    let (_, annex_hash) = blvm_consensus::taproot::strip_taproot_annex(&placeholder);

    let sighash = compute_tapscript_signature_hash(
        &tx,
        0,
        &[prevout_value],
        &[script_pubkey.as_slice()],
        &tapscript,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        0xffff_ffff,
        0x00,
        annex_hash.as_ref(),
    )
    .expect("tapscript sighash");
    let msg = Message::from_digest_slice(&sighash).expect("32-byte sighash");
    let sig = secp.sign_schnorr_no_aux_rand(&msg, &keypair);
    let sig_bytes = sig.as_ref().to_vec();

    let mut witness = vec![sig_bytes, tapscript, control_block];
    if let Some(a) = annex {
        witness.push(a);
    }

    (tx, script_pubkey, witness, prevout_value)
}

fn taproot_equiv_flags() -> u32 {
    SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT
}

fn unexec_if_equiv(tapscript: Vec<u8>, annex: Option<Vec<u8>>) {
    use blvm_consensus::TAPROOT_ACTIVATION_MAINNET;

    let (tx, script_pubkey, witness, prevout_value) = build_unexec_if_spend(tapscript, annex);
    let pv = vec![prevout_value];
    let psp_refs: Vec<&[u8]> = vec![script_pubkey.as_slice()];

    assert_fast_path_equiv(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        Some(&witness),
        taproot_equiv_flags(),
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(TAPROOT_ACTIVATION_MAINNET),
    );
}

#[test]
fn fast_path_equiv_unexec_if_suffix_checksig() {
    use secp256k1::{Keypair, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x42; 32]).unwrap();
    let kp = Keypair::from_secret_key(&secp, &sk);
    let pk = kp.x_only_public_key().0.serialize();
    let mut tapscript = envelope(&envelope_body());
    tapscript.extend(push32_checksig(&pk));
    unexec_if_equiv(tapscript, None);
}

#[test]
fn fast_path_equiv_unexec_if_prefix_checksig() {
    use secp256k1::{Keypair, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x42; 32]).unwrap();
    let kp = Keypair::from_secret_key(&secp, &sk);
    let pk = kp.x_only_public_key().0.serialize();
    let mut tapscript = push32_checksig(&pk);
    tapscript.extend(envelope(&envelope_body()));
    unexec_if_equiv(tapscript, None);
}

#[test]
fn fast_path_equiv_unexec_if_annex() {
    use secp256k1::{Keypair, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x42; 32]).unwrap();
    let kp = Keypair::from_secret_key(&secp, &sk);
    let pk = kp.x_only_public_key().0.serialize();
    let mut tapscript = envelope(&envelope_body());
    tapscript.extend(push32_checksig(&pk));
    unexec_if_equiv(tapscript, Some(vec![0x50, 0xaa]));
}

#[test]
fn fast_path_equiv_two_envelopes_no_unexec_if() {
    use secp256k1::{Keypair, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x42; 32]).unwrap();
    let kp = Keypair::from_secret_key(&secp, &sk);
    let pk = kp.x_only_public_key().0.serialize();
    let mut tapscript = envelope(&[0x01, 0xaa]);
    tapscript.extend(envelope(&[0x01, 0xbb]));
    tapscript.extend(push32_checksig(&pk));
    unexec_if_equiv(tapscript, None);
}

#[test]
fn fast_path_equiv_nested_if_no_unexec_if() {
    use secp256k1::{Keypair, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x42; 32]).unwrap();
    let kp = Keypair::from_secret_key(&secp, &sk);
    let pk = kp.x_only_public_key().0.serialize();
    let mut tapscript = vec![OP_0, OP_IF, OP_0, OP_IF, OP_ENDIF, OP_ENDIF];
    tapscript.extend(push32_checksig(&pk));
    unexec_if_equiv(tapscript, None);
}

#[test]
fn fast_path_equiv_leftover_opcode_no_unexec_if() {
    use secp256k1::{Keypair, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x42; 32]).unwrap();
    let kp = Keypair::from_secret_key(&secp, &sk);
    let pk = kp.x_only_public_key().0.serialize();
    let mut tapscript = envelope(&[0x01, 0xaa]);
    tapscript.push(OP_DUP);
    tapscript.extend(push32_checksig(&pk));
    // Interpreter may fail (extra stack item); still no UnexecIf shortcut.
    let (tx, script_pubkey, witness, prevout_value) = build_unexec_if_spend(tapscript, None);
    let pv = vec![prevout_value];
    let psp_refs: Vec<&[u8]> = vec![script_pubkey.as_slice()];
    disable_fast_paths(false);
    let fast = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        Some(&witness),
        taproot_equiv_flags(),
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(blvm_consensus::TAPROOT_ACTIVATION_MAINNET),
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
    )
    .expect("fast path verify");
    disable_fast_paths(true);
    let full = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        Some(&witness),
        taproot_equiv_flags(),
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(blvm_consensus::TAPROOT_ACTIVATION_MAINNET),
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
    )
    .expect("interpreter verify");
    disable_fast_paths(false);
    assert_eq!(fast, full);
}

#[test]
fn fast_path_equiv_op_success_no_unexec_if() {
    use secp256k1::{Keypair, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x42; 32]).unwrap();
    let kp = Keypair::from_secret_key(&secp, &sk);
    let pk = kp.x_only_public_key().0.serialize();
    let mut tapscript = vec![OP_0, OP_IF, 0xba, OP_ENDIF];
    tapscript.extend(push32_checksig(&pk));
    let (tx, script_pubkey, witness, prevout_value) = build_unexec_if_spend(tapscript, None);
    let pv = vec![prevout_value];
    let psp_refs: Vec<&[u8]> = vec![script_pubkey.as_slice()];
    disable_fast_paths(false);
    let fast = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        Some(&witness),
        taproot_equiv_flags(),
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(blvm_consensus::TAPROOT_ACTIVATION_MAINNET),
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
    disable_fast_paths(true);
    let full = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        Some(&witness),
        taproot_equiv_flags(),
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(blvm_consensus::TAPROOT_ACTIVATION_MAINNET),
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
    assert_eq!(fast.is_ok(), full.is_ok());
    if let (Ok(a), Ok(b)) = (fast, full) {
        assert_eq!(a, b);
    }
}

/// Counter is process-global; keep this test self-contained and sequential.
#[test]
fn unexec_if_fast_path_counter() {
    use secp256k1::{Keypair, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x42; 32]).unwrap();
    let kp = Keypair::from_secret_key(&secp, &sk);
    let pk = kp.x_only_public_key().0.serialize();

    let mut hit_script = envelope(&envelope_body());
    hit_script.extend(push32_checksig(&pk));
    let (tx, script_pubkey, witness, prevout_value) = build_unexec_if_spend(hit_script, None);
    let pv = vec![prevout_value];
    let psp_refs: Vec<&[u8]> = vec![script_pubkey.as_slice()];
    let _ = blvm_consensus::script::take_fast_path_unexec_if();
    disable_fast_paths(false);
    let ok = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        Some(&witness),
        taproot_equiv_flags(),
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(blvm_consensus::TAPROOT_ACTIVATION_MAINNET),
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
    )
    .expect("verify");
    assert!(ok);
    assert!(
        blvm_consensus::script::take_fast_path_unexec_if() >= 1,
        "suffix UnexecIf must increment FAST_PATH_UNEXEC_IF"
    );

    let mut miss = envelope(&[0x01, 0xaa]);
    miss.extend(envelope(&[0x01, 0xbb]));
    miss.extend(push32_checksig(&pk));
    let (tx2, spk2, wit2, val2) = build_unexec_if_spend(miss, None);
    let pv2 = vec![val2];
    let psp2: Vec<&[u8]> = vec![spk2.as_slice()];
    let _ = blvm_consensus::script::take_fast_path_unexec_if();
    let _ = verify_script_with_context_full(
        &tx2.inputs[0].script_sig,
        &spk2,
        Some(&wit2),
        taproot_equiv_flags(),
        &tx2,
        0,
        &pv2,
        &psp2,
        Some(blvm_consensus::TAPROOT_ACTIVATION_MAINNET),
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
    assert_eq!(
        blvm_consensus::script::take_fast_path_unexec_if(),
        0,
        "two envelopes must not take UnexecIf"
    );
}
