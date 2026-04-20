#![no_main]
//! ECDSA verify: compact signature + compressed pubkey.
use blvm_consensus::secp256k1_backend::verify_ecdsa;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 + 64 + 33 {
        return;
    }
    let mut msg = [0u8; 32];
    msg.copy_from_slice(&data[0..32]);
    let mut sig = [0u8; 64];
    sig.copy_from_slice(&data[32..96]);
    let mut pk = [0u8; 33];
    pk.copy_from_slice(&data[96..129]);
    let _ = verify_ecdsa(&msg, &sig, &pk);
});
