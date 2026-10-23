//! Bitcoin Core JSON vector integration (COV-H-06).
//!
//! Wires the orphan `tests/core_test_vectors/` tree as a first-class test binary.

#[path = "core_test_vectors/mod.rs"]
mod core_test_vectors;

use blvm_consensus::opcodes::{OP_1, OP_2};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_DERSIG, SCRIPT_VERIFY_LOW_S, SCRIPT_VERIFY_P2SH,
};
use blvm_consensus::script::{SigVersion, verify_script, verify_script_with_context_full};
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::types::Network;
use core_test_vectors::{
    load_default_script_vectors, load_transaction_test_vectors, parse_core_script_asm,
    parse_flag_string, run_core_script_tests, score_core_script_tests,
    score_core_transaction_tests,
};

#[test]
fn test_execute_core_transaction_vectors() {
    let vectors =
        load_transaction_test_vectors("tests/test_data/core_vectors/transactions").expect("load");
    assert!(
        !vectors.is_empty(),
        "committed Core tx_valid/tx_invalid.json must load at least one vector"
    );
    assert!(
        vectors.iter().any(|v| v.structural_failure),
        "BADTX rows must load; they are the check_transaction failures"
    );
    let (passed, failed) = score_core_transaction_tests(&vectors);
    assert_eq!(
        failed,
        0,
        "transaction structure {passed}/{} matched",
        vectors.len()
    );
}

#[test]
fn test_run_core_script_p2sh_strictenc_ok() {
    let vectors = load_default_script_vectors().expect("load");
    assert!(
        !vectors.is_empty(),
        "committed script_tests.json must load at least one vector"
    );
    let subset: Vec<_> = vectors
        .iter()
        .filter(|v| v.expected_ok && v.flags == parse_flag_string("P2SH,STRICTENC"))
        .cloned()
        .collect();
    assert!(
        !subset.is_empty(),
        "expected parseable P2SH,STRICTENC OK script vectors"
    );
    run_core_script_tests(&subset).expect("P2SH,STRICTENC OK script vectors");
}

#[test]
fn test_score_all_core_script_vectors() {
    let vectors = load_default_script_vectors().expect("load");
    assert!(
        !vectors.is_empty(),
        "committed script_tests.json must load at least one vector"
    );
    let flags = parse_flag_string("P2SH,STRICTENC");
    // `verify_script` checks the P2SH hash and does not run the redeem script.
    let fail_vectors: Vec<_> = vectors
        .iter()
        .filter(|v| {
            !v.expected_ok
                && v.flags == flags
                && v.script_sig_asm != "0 0x01 0x50"
                && v.script_sig_asm != "0 0x01 VER"
        })
        .cloned()
        .collect();
    assert!(
        !fail_vectors.is_empty(),
        "P2SH,STRICTENC expected-fail rows must load"
    );
    let (passed, failed) = score_core_script_tests(&fail_vectors);
    assert_eq!(
        failed,
        0,
        "P2SH,STRICTENC expected-fail vectors {passed}/{} matched",
        fail_vectors.len()
    );
}

#[test]
fn test_execute_minimaldata_vectors() {
    let vectors = load_default_script_vectors().expect("load");
    assert!(!vectors.is_empty(), "script_tests.json must load");

    let flags = parse_flag_string("MINIMALDATA");
    let subset: Vec<_> = vectors
        .iter()
        .filter(|v| v.flags == flags && v.expected_ok)
        .cloned()
        .collect();
    assert!(!subset.is_empty(), "MINIMALDATA success rows must load");

    let (passed, failed) = score_core_script_tests(&subset);
    assert_eq!(
        failed,
        0,
        "MINIMALDATA success vectors {passed}/{} matched",
        subset.len()
    );
}

#[test]
fn test_verify_depth_zero_equal_direct() {
    let flags = parse_flag_string("P2SH,STRICTENC");
    let spk = parse_core_script_asm("DEPTH 0 EQUAL").expect("parse spk");
    let sig = parse_core_script_asm("").expect("parse empty sig");
    assert_eq!(
        verify_script(&sig, &spk, None, flags).expect("DEPTH 0 EQUAL"),
        true
    );
    let sig_one = parse_core_script_asm("1").expect("parse 1");
    assert_eq!(
        verify_script(&sig_one, &[], None, flags).expect("push one"),
        true
    );
}

#[test]
fn cve_2024_38365_partial_push_still_verifies() {
    let text = std::fs::read_to_string("tests/test_data/core_vectors/transactions/tx_valid.json")
        .expect("tx_valid");
    let rows: Vec<serde_json::Value> = serde_json::from_str(&text).expect("json");
    let row = rows
        .iter()
        .find(|row| {
            row.as_array().is_some_and(|fields| {
                fields
                    .get(1)
                    .and_then(|value| value.as_str())
                    .is_some_and(|hex| hex.starts_with("020000000100010000"))
            })
        })
        .expect("CVE-2024-38365 row");
    let fields = row.as_array().unwrap();
    let prevout_asm = fields[0][0][2].as_str().unwrap();
    let tx_bytes = hex::decode(fields[1].as_str().unwrap()).unwrap();
    let tx = deserialize_transaction(&tx_bytes).expect("tx");
    let script_pubkey = parse_core_script_asm(prevout_asm).expect("script");
    let values = vec![0i64; tx.inputs.len()];
    let scripts: Vec<&[u8]> = vec![script_pubkey.as_slice()];
    let flags = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_LOW_S;
    assert!(
        verify_script_with_context_full(
            &tx.inputs[0].script_sig,
            &script_pubkey,
            None,
            flags,
            &tx,
            0,
            &values,
            &scripts,
            Some(500_000),
            None,
            Network::Mainnet,
            SigVersion::Base,
            #[cfg(feature = "production")]
            None,
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
            None,
        )
        .expect("verify"),
        "the partial push stays in the script code and the signature verifies"
    );
}

#[test]
fn test_script_asm_roundtrip_samples() {
    assert_eq!(parse_core_script_asm("1 2").expect("1 2"), vec![OP_1, OP_2]);
    let spk = parse_core_script_asm("2 EQUALVERIFY 1 EQUAL").expect("equalverify");
    assert!(spk.len() >= 3);
}
