#![no_main]
//! Taproot sighash tagged hash helper.
use blvm_consensus::secp256k1_backend::tap_sighash_hash;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = tap_sighash_hash(data);
});
