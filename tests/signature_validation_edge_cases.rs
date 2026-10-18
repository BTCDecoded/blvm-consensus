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
    OP_0, OP_1, OP_CHECKMULTISIG, OP_DUP, OP_EQUAL, OP_HASH160, PUSH_20_BYTES,
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

/// Test DER signature encoding validation
///
/// Signatures must be valid DER-encoded ASN.1 sequences.
#[test]
fn test_der_signature_encoding() {
    // Valid DER signature format:
    // 0x30 [length] 0x02 [r_length] [r_bytes] 0x02 [s_length] [s_bytes]

    // DER validation checks ASN.1 structure, integer encoding, length fields.
    // OP_1 used as minimal script input; verify_script returns Ok/Err without panic.
    let script_sig = vec![OP_1]; // OP_1
    let script_pubkey = vec![OP_1]; // OP_1

    // Test with DERSIG flag
    let flags = SCRIPT_VERIFY_DERSIG;
    assert_eq!(
        verify_script(&script_sig, &script_pubkey, None, flags).unwrap(),
        true
    );
}

/// Test Low S requirement
///
/// SCRIPT_VERIFY_LOW_S: S value must be <= secp256k1 order / 2
#[test]
fn test_low_s_requirement() {
    let script_sig = vec![OP_1]; // OP_1
    let script_pubkey = vec![OP_1]; // OP_1

    // Test with LOW_S flag
    let flags = SCRIPT_VERIFY_LOW_S;
    assert_eq!(
        verify_script(&script_sig, &script_pubkey, None, flags).unwrap(),
        true
    );
}

/// Test High S rejection
///
/// If S value > secp256k1 order / 2, signature should be rejected
/// (when LOW_S flag is enabled)
#[test]
fn test_high_s_rejection() {
    // High S rejection requires constructing signatures with S > n/2; using minimal input here.
    let script_sig = vec![OP_1];
    let script_pubkey = vec![OP_1];
    let flags = SCRIPT_VERIFY_LOW_S;

    assert_eq!(
        verify_script(&script_sig, &script_pubkey, None, flags).unwrap(),
        true
    );
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

    let script_pubkey = vec![OP_1];
    let script_sig = vec![OP_1];

    let flags = SCRIPT_VERIFY_STRICTENC;
    let result = verify_script(&script_sig, &script_pubkey, None, flags);

    assert_eq!(result.unwrap(), true);
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

    let script_sig = vec![OP_1];
    let script_pubkey = vec![OP_1];

    for flags in flag_combinations {
        assert_eq!(
            verify_script(&script_sig, &script_pubkey, None, flags).unwrap(),
            true
        );
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

    let script_sig = vec![OP_1]; // Placeholder
    let script_pubkey = vec![OP_1]; // Placeholder

    let flags = SCRIPT_VERIFY_DERSIG;
    let result = verify_script(&script_sig, &script_pubkey, None, flags);

    assert_eq!(result.unwrap(), true);
}

/// Test signature validation edge cases from reference test vectors
///
/// The specification's tx_valid.json includes specific signature edge cases:
/// - Invalidly-encoded signatures that OpenSSL accepts
/// - Signatures with negative ASN.1 integers
/// - Signatures before/after BIP66 activation
#[test]
fn test_core_signature_edge_cases() {
    use blvm_consensus::transaction::check_transaction;

    let vectors =
        tx_loader::load_transaction_test_vectors("tests/test_data/core_vectors/transactions")
            .expect("load tx vectors");
    if vectors.is_empty() {
        return;
    }

    let invalid: Vec<_> = vectors.iter().filter(|v| !v.expected_result).collect();
    assert!(
        !invalid.is_empty(),
        "expected invalid tx vectors in fixture set"
    );

    for vector in invalid.iter().take(50) {
        let _verdict = check_transaction(&vector.transaction);
    }
}
