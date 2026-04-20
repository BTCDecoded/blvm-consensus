#![no_main]
//! Taproot internal-key tweak (BIP 341).
use blvm_consensus::taproot::compute_taproot_tweak;
use blvm_consensus::types::Hash;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 64 {
        return;
    }
    let mut internal = [0u8; 32];
    internal.copy_from_slice(&data[0..32]);
    let mut merkle: Hash = [0u8; 32];
    merkle.copy_from_slice(&data[32..64]);
    let _ = compute_taproot_tweak(&internal, &merkle);
});
