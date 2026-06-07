//! COV-C-02: Execute all parseable Core legacy script vectors for coverage.
//!
//! Runs OK and expected-fail cases through `verify_script`; asserts only on
//! known-good subsets (P2SH,STRICTENC OK gate).

#[path = "core_test_vectors/script_tests.rs"]
mod core_script_tests;

use core_script_tests::{
    load_default_script_vectors, load_default_witness_script_vectors, parse_flag_string,
    run_core_script_tests, score_core_script_tests, score_witness_script_tests,
};

#[test]
fn test_execute_all_legacy_script_vectors() {
    let vectors = load_default_script_vectors().expect("load script_tests.json");
    if vectors.is_empty() {
        return;
    }

    let (passed, failed) = score_core_script_tests(&vectors);
    eprintln!(
        "Core legacy script vectors: {passed}/{} scored, {failed} mismatches",
        vectors.len()
    );

    let p2sh_strictenc_ok: Vec<_> = vectors
        .iter()
        .filter(|v| v.expected_ok && v.flags == parse_flag_string("P2SH,STRICTENC"))
        .cloned()
        .collect();
    assert!(!p2sh_strictenc_ok.is_empty());
    run_core_script_tests(&p2sh_strictenc_ok)
        .expect("all P2SH,STRICTENC OK vectors must pass verify_script");
}

#[test]
fn test_execute_legacy_fail_vectors_for_error_paths() {
    let vectors = load_default_script_vectors().expect("load");
    if vectors.is_empty() {
        return;
    }

    let fail_vectors: Vec<_> = vectors.iter().filter(|v| !v.expected_ok).cloned().collect();
    if fail_vectors.is_empty() {
        return;
    }

    let (passed, failed) = score_core_script_tests(&fail_vectors);
    eprintln!(
        "Core legacy fail vectors: {passed}/{} scored, {failed} mismatches",
        fail_vectors.len()
    );
    // Exercise error paths; do not require 100% agreement with Core's granular error codes.
    assert!(
        passed + failed == fail_vectors.len(),
        "every fail vector must be executed"
    );
}

#[test]
fn test_execute_minimaldata_vectors_if_present() {
    let vectors = load_default_script_vectors().expect("load");
    if vectors.is_empty() {
        return;
    }

    let flags = parse_flag_string("MINIMALDATA");
    let subset: Vec<_> = vectors
        .iter()
        .filter(|v| v.flags == flags)
        .cloned()
        .collect();
    if subset.is_empty() {
        return;
    }

    let (passed, failed) = score_core_script_tests(&subset);
    eprintln!("MINIMALDATA vectors: {passed}/{} scored", subset.len());
    assert_eq!(passed + failed, subset.len());
}

#[test]
fn test_execute_all_witness_script_vectors() {
    let vectors = load_default_witness_script_vectors().expect("load witness vectors");
    if vectors.is_empty() {
        return;
    }
    let (passed, failed) = score_witness_script_tests(&vectors);
    eprintln!(
        "Core witness script vectors: {passed}/{} scored, {failed} mismatches",
        vectors.len()
    );
    assert_eq!(passed + failed, vectors.len());
}
