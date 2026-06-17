//! COV-C-06d: BIP348 Schnorr collector, batch verify, and tapscript Schnorr paths.

#![cfg(feature = "production")]

use blvm_consensus::bip348::{
    SchnorrSignatureCollector, batch_verify_signatures_from_stack, verify_signature_from_stack,
    verify_tapscript_schnorr_signature,
};

fn hex32(s: &str) -> [u8; 32] {
    let v = hex::decode(s).expect("hex32");
    v.try_into().expect("32 bytes")
}

fn hex64(s: &str) -> [u8; 64] {
    let v = hex::decode(s).expect("hex64");
    v.try_into().expect("64 bytes")
}

/// BIP340 vector (32-byte message, direct digest in batch/tapscript paths).
fn bip340_vector() -> ([u8; 32], [u8; 32], [u8; 64]) {
    (
        hex32("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659"),
        hex32("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89"),
        hex64(
            "6896BD60EEAE296DB48A229FF71DFE071BDE413E6D43F917DC8DCF8C78DE33418906D11AC976ABCCB20B091292BFF4EA897EFCB639EA871CFA95F6DE339E4B0A",
        ),
    )
}

#[test]
fn test_batch_verify_signatures_from_stack_empty() {
    assert!(batch_verify_signatures_from_stack(&[]).unwrap().is_empty());
}

#[test]
fn test_batch_verify_unknown_pubkey_type_succeeds() {
    let tasks = [(
        b"arbitrary message".as_slice(),
        [1u8; 33].as_slice(),
        [0u8; 64].as_slice(),
    )];
    assert_eq!(
        batch_verify_signatures_from_stack(&tasks).unwrap(),
        vec![true]
    );
}

#[test]
fn test_batch_verify_empty_pubkey_fails() {
    let tasks = [(b"msg".as_slice(), [].as_slice(), [0u8; 64].as_slice())];
    assert_eq!(
        batch_verify_signatures_from_stack(&tasks).unwrap(),
        vec![false]
    );
}

#[test]
fn test_batch_verify_invalid_signature_length_fails() {
    let tasks = [(
        b"msg".as_slice(),
        [1u8; 32].as_slice(),
        [0u8; 63].as_slice(),
    )];
    assert_eq!(
        batch_verify_signatures_from_stack(&tasks).unwrap(),
        vec![false]
    );
}

#[test]
fn test_batch_verify_valid_bip340_vector() {
    let (pk, msg, sig) = bip340_vector();
    let tasks = [(msg.as_slice(), pk.as_slice(), sig.as_slice())];
    assert_eq!(
        batch_verify_signatures_from_stack(&tasks).unwrap(),
        vec![true]
    );
}

#[test]
fn test_batch_verify_mixed_tasks() {
    let (pk, msg, sig) = bip340_vector();
    let tasks = [
        (msg.as_slice(), pk.as_slice(), sig.as_slice()),
        (
            b"other".as_slice(),
            [2u8; 33].as_slice(),
            [0u8; 64].as_slice(),
        ),
        (b"x".as_slice(), [].as_slice(), [0u8; 64].as_slice()),
    ];
    let results = batch_verify_signatures_from_stack(&tasks).unwrap();
    assert_eq!(results.len(), 3);
    assert!(results[0]);
    assert!(results[1]);
    assert!(!results[2]);
}

#[test]
fn test_batch_verify_hashes_non_32_byte_messages() {
    let (pk, msg, sig) = bip340_vector();
    // CSFS path: non-32-byte message is SHA256-hashed before verify — invalid sig expected.
    let tasks = [(
        b"not-thirty-two-bytes".as_slice(),
        pk.as_slice(),
        sig.as_slice(),
    )];
    let results = batch_verify_signatures_from_stack(&tasks).unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0]);
    // Sanity: 32-byte message still verifies.
    let ok = [(msg.as_slice(), pk.as_slice(), sig.as_slice())];
    assert!(batch_verify_signatures_from_stack(&ok).unwrap()[0]);
}

#[test]
fn test_verify_tapscript_schnorr_immediate_valid() {
    let (pk, msg, sig) = bip340_vector();
    assert!(verify_tapscript_schnorr_signature(&msg, pk.as_slice(), sig.as_slice(), None).unwrap());
}

#[test]
fn test_verify_tapscript_schnorr_wrong_pubkey_length() {
    let (_, msg, sig) = bip340_vector();
    assert!(!verify_tapscript_schnorr_signature(&msg, &[1u8; 33], sig.as_slice(), None).unwrap());
}

#[test]
fn test_verify_tapscript_schnorr_wrong_sig_length() {
    let (pk, msg, _) = bip340_vector();
    assert!(!verify_tapscript_schnorr_signature(&msg, pk.as_slice(), &[0u8; 63], None).unwrap());
}

#[test]
fn test_verify_tapscript_schnorr_collector_deferred_batch() {
    let (pk, msg, sig) = bip340_vector();
    let collector = SchnorrSignatureCollector::new();
    assert!(collector.is_empty());
    assert!(!collector.uses_soa());
    assert!(
        verify_tapscript_schnorr_signature(&msg, pk.as_slice(), sig.as_slice(), Some(&collector),)
            .unwrap()
    );
    let results = collector.verify_batch().unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0]);
}

#[test]
fn test_collector_soa_capacity_and_verify_batch() {
    let collector = SchnorrSignatureCollector::new_with_capacity(8);
    assert!(collector.uses_soa());
    let (pk, msg, sig) = bip340_vector();
    collector.collect_with_index(2, msg.as_slice(), pk.as_slice(), sig.as_slice());
    collector.collect_with_index(0, msg.as_slice(), pk.as_slice(), sig.as_slice());
    let results = collector.verify_batch().unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|&v| v));
    collector.clear();
    assert!(collector.is_empty());
}

#[test]
fn test_collector_soa_skips_invalid_lengths() {
    let collector = SchnorrSignatureCollector::new_with_capacity(4);
    collector.collect_with_index(0, b"short", &[1u8; 32], &[0u8; 64]);
    // Invalid SoA dimensions fall through to SegQueue; batch verify fails closed.
    let results = collector.verify_batch().unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0]);
}

#[test]
fn test_collector_segqueue_invalid_signature() {
    let collector = SchnorrSignatureCollector::new();
    collector.collect(b"hello", &[2u8; 32], &[0u8; 64]);
    let results = collector.verify_batch().unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0]);
}

#[test]
fn test_verify_signature_from_stack_with_collector_defers() {
    let collector = SchnorrSignatureCollector::new();
    assert!(
        verify_signature_from_stack(b"hello", &[4u8; 32], &[0u8; 64], Some(&collector),).unwrap()
    );
    let results = collector.verify_batch().unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0]);
}

#[test]
fn test_verify_signature_from_stack_immediate_invalid_sig() {
    assert!(!verify_signature_from_stack(b"hello", &[5u8; 32], &[0u8; 64], None).unwrap());
}

#[cfg(feature = "rayon")]
#[test]
fn test_collector_extend_from_merges_tasks() {
    let a = SchnorrSignatureCollector::new();
    let b = SchnorrSignatureCollector::new();
    b.collect(b"x", &[6u8; 32], &[0u8; 64]);
    a.extend_from(&b);
    let results = a.verify_batch().unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0]);
}

#[test]
fn test_collector_default_impl() {
    let c = SchnorrSignatureCollector::default();
    assert!(c.is_empty());
}

#[cfg(feature = "rayon")]
#[test]
fn test_collector_try_verify_chunk_streaming_merge() {
    let (pk, msg, sig) = bip340_vector();
    let collector = SchnorrSignatureCollector::new();
    collector.collect(msg.as_slice(), pk.as_slice(), sig.as_slice());
    collector.try_verify_chunk(4);
    let results = collector.verify_batch().unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0]);
}

#[cfg(feature = "rayon")]
#[test]
fn test_collector_try_verify_chunk_empty_is_noop() {
    let collector = SchnorrSignatureCollector::new();
    collector.try_verify_chunk(0);
    assert!(collector.verify_batch().unwrap().is_empty());
}

#[test]
fn test_verify_tapscript_schnorr_invalid_pubkey_bytes() {
    let (_, msg, sig) = bip340_vector();
    // Invalid x-only point still exercises parse/verify failure path.
    assert!(!verify_tapscript_schnorr_signature(&msg, &[0xff; 32], sig.as_slice(), None).unwrap());
}
