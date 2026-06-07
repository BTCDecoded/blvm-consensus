//! COV-C-02f: Production signature verification paths (`script/signature.rs`).

use blvm_consensus::script::{batch_verify_signatures, SigVersion};
use blvm_consensus::types::Network;

#[test]
fn test_batch_verify_signatures_empty() {
    let results = batch_verify_signatures(&[], 0, 0, Network::Mainnet, SigVersion::Base).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_batch_verify_signatures_invalid_der_pushes_false() {
    let pubkey = [0x02u8; 33];
    let bad_sig = [0x30u8, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01];
    let sighash = [0xab; 32];
    let tasks = [(&pubkey[..], &bad_sig[..], sighash)];
    let results =
        batch_verify_signatures(&tasks, 0, 500_000, Network::Mainnet, SigVersion::Base).unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0]);
}

#[cfg(feature = "production")]
#[test]
fn test_verify_pre_extracted_ecdsa_invalid_signature() {
    use blvm_consensus::script::verify_pre_extracted_ecdsa;

    let pubkey = [0x02u8; 33];
    let bad_sig = [0x30u8, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01];
    let sighash = [0xcd; 32];
    assert!(
        !verify_pre_extracted_ecdsa(&pubkey, &bad_sig, &sighash, 0, 500_000, Network::Mainnet)
            .unwrap()
    );
}
