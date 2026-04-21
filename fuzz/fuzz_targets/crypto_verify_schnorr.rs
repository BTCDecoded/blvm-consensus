#![no_main]
//! BIP340 Schnorr signature verify.
use blvm_consensus::secp256k1_backend::verify_schnorr;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 64 + 32 + 32 {
        return;
    }
    let mut sig = [0u8; 64];
    sig.copy_from_slice(&data[0..64]);
    let msg = &data[64..64 + 32];
    let mut pk = [0u8; 32];
    pk.copy_from_slice(&data[96..128]);
    let _ = verify_schnorr(&sig, msg, &pk);
});
