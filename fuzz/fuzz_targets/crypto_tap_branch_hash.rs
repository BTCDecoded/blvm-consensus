#![no_main]
//! Taproot branch hash (BIP 341).
use blvm_consensus::secp256k1_backend::tap_branch_hash;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 64 {
        return;
    }
    let mut left = [0u8; 32];
    left.copy_from_slice(&data[0..32]);
    let mut right = [0u8; 32];
    right.copy_from_slice(&data[32..64]);
    let _ = tap_branch_hash(&left, &right);
});
