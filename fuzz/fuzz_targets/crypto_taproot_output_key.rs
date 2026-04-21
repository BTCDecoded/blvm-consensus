#![no_main]
//! Taproot output key tweak from internal key + merkle root.
use blvm_consensus::secp256k1_backend::taproot_output_key;
use blvm_consensus::types::Hash;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 + 32 {
        return;
    }
    let mut internal = [0u8; 32];
    internal.copy_from_slice(&data[0..32]);
    let mut merkle: Hash = [0u8; 32];
    merkle.copy_from_slice(&data[32..64]);
    let _ = taproot_output_key(&internal, &merkle);
});
