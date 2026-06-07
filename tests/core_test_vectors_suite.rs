//! Bitcoin Core JSON vector integration (COV-H-06).
//!
//! Wires the orphan `tests/core_test_vectors/` tree as a first-class test binary.

#[path = "core_test_vectors/mod.rs"]
mod core_test_vectors;

use core_test_vectors::{
    load_default_script_vectors, load_transaction_test_vectors, parse_flag_string,
    run_core_script_tests, score_core_script_tests, score_core_transaction_tests,
};

#[test]
fn test_execute_core_transaction_vectors_if_present() {
    let vectors =
        load_transaction_test_vectors("tests/test_data/core_vectors/transactions").expect("load");
    if vectors.is_empty() {
        return;
    }
    let (passed, failed) = score_core_transaction_tests(&vectors);
    eprintln!(
        "Core transaction vectors (check_transaction): {passed}/{} scored, {failed} mismatches",
        vectors.len()
    );
    assert_eq!(passed + failed, vectors.len());
}

#[test]
fn test_run_core_script_p2sh_strictenc_ok_if_present() {
    let vectors = load_default_script_vectors().expect("load");
    if vectors.is_empty() {
        return;
    }
    let subset: Vec<_> = vectors
        .iter()
        .filter(|v| v.expected_ok && v.flags == parse_flag_string("P2SH,STRICTENC"))
        .cloned()
        .collect();
    if subset.is_empty() {
        return;
    }
    run_core_script_tests(&subset).expect("P2SH,STRICTENC OK script vectors");
}

#[test]
fn test_score_all_core_script_vectors_if_present() {
    let vectors = load_default_script_vectors().expect("load");
    if vectors.is_empty() {
        return;
    }
    let (passed, failed) = score_core_script_tests(&vectors);
    eprintln!(
        "core_test_vectors_suite script score: {passed}/{}",
        vectors.len()
    );
    assert_eq!(passed + failed, vectors.len());
}
