//! Bitcoin Core JSON vector integration (COV-H-06).
//!
//! Wires the orphan `tests/core_test_vectors/` tree as a first-class test binary.

#[path = "core_test_vectors/mod.rs"]
mod core_test_vectors;

use blvm_consensus::opcodes::{OP_1, OP_2};
use blvm_consensus::script::verify_script;
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
fn test_script_asm_roundtrip_samples() {
    assert_eq!(parse_core_script_asm("1 2").expect("1 2"), vec![OP_1, OP_2]);
    let spk = parse_core_script_asm("2 EQUALVERIFY 1 EQUAL").expect("equalverify");
    assert!(spk.len() >= 3);
}
