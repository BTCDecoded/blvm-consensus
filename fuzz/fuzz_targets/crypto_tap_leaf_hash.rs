#![no_main]
//! Taproot leaf hash (BIP 341).
use blvm_consensus::secp256k1_backend::tap_leaf_hash;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let leaf_version = data[0];
    let script = &data[1..];
    let _ = tap_leaf_hash(leaf_version, script);
});
