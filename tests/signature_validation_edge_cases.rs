//! Script signature validation edge cases
//!
//! Tests for signature validation edge cases:
//! - DER signature encoding validation
//! - Low S requirement (SCRIPT_VERIFY_LOW_S)
//! - High S rejection
//! - Null dummy enforcement (SCRIPT_VERIFY_NULLDUMMY)
//! - P2SH scriptSig push-only enforcement
//! - Signature strict encoding (SCRIPT_VERIFY_STRICTENC)
//!
//! Consensus-critical: Signature validation bugs can allow invalid transactions.

use blvm_consensus::error::{ConsensusError, ScriptErrorCode};
use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_CHECKMULTISIG, OP_CHECKSIG, OP_DUP, OP_EQUAL, OP_HASH160, PUSH_20_BYTES,
};
use blvm_consensus::script::{
    SigVersion, disable_fast_paths, p2sh_push_only_check, verify_script,
    verify_script_with_context_full,
};
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};

#[path = "core_test_vectors/tx_loader.rs"]
mod tx_loader;

/// Script verification flags for signature validation
pub const SCRIPT_VERIFY_DERSIG: u32 = 0x04;
pub const SCRIPT_VERIFY_LOW_S: u32 = 0x08;
pub const SCRIPT_VERIFY_NULLDUMMY: u32 = 0x10;
pub const SCRIPT_VERIFY_STRICTENC: u32 = 0x02;

/// Signature over the all-zero message, the hash `verify_script` feeds CHECKSIG.
/// Secret is 32 bytes of `0x11`. Low S verifies. High S is `n − s`.
fn zero_hash_pubkey() -> Vec<u8> {
    hex::decode("034f355bdcb7cc0af728ef3cceb9615d90684bb5b2ca5f859ab0f0b704075871aa").unwrap()
}

fn low_s_der() -> Vec<u8> {
    hex::decode("3045022100e0a180fdd0fe38037cc878c03832861b40a29d32bd7b40b10c9e1efc8c1468a002205ae06d1624896d0d29f4b31e32772ea3cb1b4d7ed4e077e5da28dcc33c0e7812").unwrap()
}

fn high_s_der() -> Vec<u8> {
    hex::decode("3046022100e0a180fdd0fe38037cc878c03832861b40a29d32bd7b40b10c9e1efc8c1468a0022100a51f92e9db7692f2d60b4ce1cd88d15aef938f67da682855e5a981c99427c92f").unwrap()
}

fn push_data(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + data.len());
    out.push(data.len() as u8);
    out.extend_from_slice(data);
    out
}

fn verify_signed(der: &[u8], hashtype: u8, flags: u32) -> blvm_consensus::error::Result<bool> {
    let mut signature = der.to_vec();
    signature.push(hashtype);
    let script_sig = push_data(&signature);
    let mut script_pubkey = push_data(&zero_hash_pubkey());
    script_pubkey.push(OP_CHECKSIG);
    verify_script(&script_sig, &script_pubkey, None, flags)
}

/// Test DER signature encoding validation
///
/// A signature over the zero hash is strict DER. DERSIG must accept it.
#[test]
fn test_der_signature_encoding() {
    assert_eq!(
        verify_signed(&low_s_der(), 0x01, SCRIPT_VERIFY_DERSIG).unwrap(),
        true
    );
}

/// Test Low S requirement
///
/// SCRIPT_VERIFY_LOW_S: S value must be <= secp256k1 order / 2
#[test]
fn test_low_s_requirement() {
    assert_eq!(
        verify_signed(&low_s_der(), 0x01, SCRIPT_VERIFY_LOW_S).unwrap(),
        true
    );
    assert_eq!(
        verify_signed(&high_s_der(), 0x01, SCRIPT_VERIFY_LOW_S).unwrap(),
        false
    );
}

/// Test High S rejection
///
/// If S value > secp256k1 order / 2, signature should be rejected
/// (when LOW_S flag is enabled)
#[test]
fn test_high_s_rejection() {
    assert_eq!(
        verify_signed(&high_s_der(), 0x01, SCRIPT_VERIFY_LOW_S).unwrap(),
        false
    );
    assert_eq!(verify_signed(&high_s_der(), 0x01, 0).unwrap(), true);
}

/// Test NULLDUMMY enforcement
///
/// SCRIPT_VERIFY_NULLDUMMY: OP_CHECKMULTISIG dummy argument must be empty
#[test]
fn test_null_dummy_enforcement() {
    let height = blvm_consensus::BIP147_ACTIVATION_MAINNET;
    let empty_dummy = vec![OP_0, OP_0, OP_0, OP_CHECKMULTISIG];
    let one_byte_dummy = vec![0x01, 0x00, OP_0, OP_0, OP_CHECKMULTISIG];
    assert_eq!(
        nulldummy_at(&empty_dummy, height).expect("empty dummy"),
        true
    );
    match nulldummy_at(&one_byte_dummy, height) {
        Err(ConsensusError::ScriptErrorWithCode { code, .. }) => {
            assert_eq!(code, ScriptErrorCode::SigNullDummy);
        }
        other => panic!("one-byte dummy must be SigNullDummy, got {other:?}"),
    }
    assert_eq!(
        nulldummy_at(&one_byte_dummy, height - 1).expect("before activation"),
        true
    );
}

fn nulldummy_at(script_pubkey: &[u8], height: u64) -> blvm_consensus::error::Result<bool> {
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x11; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };
    disable_fast_paths(true);
    let values = [0i64];
    let scripts = [script_pubkey];
    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        script_pubkey,
        None,
        SCRIPT_VERIFY_NULLDUMMY,
        &tx,
        0,
        &values,
        &scripts,
        Some(height),
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

/// P2SH scriptSig must be push-only when SCRIPT_VERIFY_P2SH is set.
#[test]
fn test_p2sh_scriptsig_push_only() {
    let redeem = vec![OP_1];
    assert!(p2sh_push_only_check(&redeem));
    assert!(!p2sh_push_only_check(&[OP_DUP]));

    let mut script_pubkey = vec![OP_HASH160, PUSH_20_BYTES];
    script_pubkey.extend_from_slice(&hash160(&redeem));
    script_pubkey.push(OP_EQUAL);

    let pushed = {
        let mut script = vec![redeem.len() as u8];
        script.extend_from_slice(&redeem);
        script
    };
    assert_eq!(
        verify_script(&pushed, &script_pubkey, None, 0x01).unwrap(),
        true
    );
    assert_eq!(
        verify_script(&vec![OP_DUP], &script_pubkey, None, 0x01).unwrap(),
        false
    );
}

fn hash160(data: &[u8]) -> [u8; 20] {
    use ripemd::Ripemd160;
    use sha2::{Digest, Sha256};
    let sha = Sha256::digest(data);
    let ripe = Ripemd160::digest(sha);
    let mut out = [0u8; 20];
    out.copy_from_slice(&ripe);
    out
}

/// Test STRICTENC enforcement
///
/// SCRIPT_VERIFY_STRICTENC: Public keys must be valid compressed or uncompressed
#[test]
fn test_strictenc_enforcement() {
    // STRICTENC requires public keys to be:
    // - Compressed: 0x02 or 0x03 + 32 bytes
    // - Uncompressed: 0x04 + 64 bytes

    assert_eq!(
        verify_signed(&low_s_der(), 0x01, SCRIPT_VERIFY_STRICTENC).unwrap(),
        true
    );
    assert_eq!(
        verify_signed(&low_s_der(), 0x00, SCRIPT_VERIFY_STRICTENC).unwrap(),
        false
    );
    assert_eq!(verify_signed(&low_s_der(), 0x00, 0).unwrap(), true);
}

/// Test combined signature validation flags
#[test]
fn test_combined_signature_flags() {
    // Test combinations of signature validation flags
    let flag_combinations = vec![
        SCRIPT_VERIFY_DERSIG,
        SCRIPT_VERIFY_LOW_S,
        SCRIPT_VERIFY_NULLDUMMY,
        SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_LOW_S,
        SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_STRICTENC,
        SCRIPT_VERIFY_LOW_S | SCRIPT_VERIFY_NULLDUMMY,
        SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_LOW_S | SCRIPT_VERIFY_STRICTENC,
    ];

    let low = low_s_der();
    let high = high_s_der();
    assert_eq!(
        verify_signed(
            &low,
            0x01,
            SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_LOW_S | SCRIPT_VERIFY_STRICTENC
        )
        .unwrap(),
        true
    );
    assert_eq!(
        verify_signed(&high, 0x01, SCRIPT_VERIFY_LOW_S | SCRIPT_VERIFY_STRICTENC).unwrap(),
        false
    );
    assert_eq!(
        verify_signed(&low, 0x00, SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_STRICTENC).unwrap(),
        false
    );

    let mut script_pubkey = push_data(&zero_hash_pubkey());
    script_pubkey.push(OP_CHECKSIG);
    let bad_der = vec![0x03, 0x30, 0x01, 0x00];
    for flags in flag_combinations {
        if flags & SCRIPT_VERIFY_DERSIG == 0 {
            continue;
        }
        match verify_script(&bad_der, &script_pubkey, None, flags) {
            Err(ConsensusError::ScriptErrorWithCode {
                code: ScriptErrorCode::SigDer,
                ..
            }) => {}
            other => panic!("non-DER under {flags:#x} must be SigDer, got {other:?}"),
        }
    }
}

/// Test signature validation with invalid DER encoding
#[test]
fn test_invalid_der_encoding() {
    // Invalid DER signatures should be rejected with DERSIG flag
    // Invalid formats include:
    // - Missing ASN.1 sequence marker
    // - Incorrect length fields
    // - Leading zeros in integers
    // - Negative integers

    let mut script_pubkey = vec![0x21];
    script_pubkey.extend_from_slice(
        &hex::decode("02865c40293a680cb9c020e7b1e106d8c1916d3cef99aa431a56d253e69256dac0").unwrap(),
    );
    script_pubkey.push(OP_CHECKSIG);
    // A 3-byte push of `30 01 00` is not a DER signature.
    let script_sig = vec![0x03, 0x30, 0x01, 0x00];

    match verify_script(&script_sig, &script_pubkey, None, SCRIPT_VERIFY_DERSIG) {
        Err(ConsensusError::ScriptErrorWithCode {
            code: ScriptErrorCode::SigDer,
            ..
        }) => {}
        other => panic!("non-DER signature under DERSIG must be SigDer, got {other:?}"),
    }
}

/// Test signature validation edge cases from reference test vectors
///
/// The specification's tx_valid.json includes specific signature edge cases:
/// - Invalidly-encoded signatures that OpenSSL accepts
/// - Signatures with negative ASN.1 integers
/// - Signatures before/after BIP66 activation
#[test]
fn test_core_signature_edge_cases() {
    use blvm_consensus::ValidationResult;

    let vectors =
        tx_loader::load_transaction_test_vectors("tests/test_data/core_vectors/transactions")
            .expect("load tx vectors");
    assert!(
        !vectors.is_empty(),
        "tx_valid.json and tx_invalid.json must load"
    );
    assert!(
        vectors.iter().any(|v| v.structural_failure),
        "BADTX rows must load"
    );
    for vector in &vectors {
        match tx_loader::structural_validation(&vector.transaction) {
            Ok(ValidationResult::Invalid(_)) if vector.structural_failure => {}
            Ok(ValidationResult::Valid) if !vector.structural_failure => {}
            other => panic!(
                "structural verdict badtx={} got {other:?} ({})",
                vector.structural_failure, vector.description
            ),
        }
    }
}
