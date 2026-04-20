#![no_main]
//! ECDSA verify from DER encoding + pubkey bytes (`verify_ecdsa_direct`).
use blvm_consensus::secp256k1_backend::verify_ecdsa_direct;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 + 4 {
        return;
    }
    let mut msg = [0u8; 32];
    msg.copy_from_slice(&data[0..32]);
    let rest = &data[32..];
    let der_end = (rest.first().copied().unwrap_or(8) as usize % rest.len().max(1)).max(1).min(rest.len());
    let der = &rest[..der_end];
    let pubkey = &rest[der_end..];
    let strict_der = data.get(1).map(|b| (b & 1) != 0).unwrap_or(true);
    let low_s = data.get(2).map(|b| (b & 1) != 0).unwrap_or(true);
    let _ = verify_ecdsa_direct(der, pubkey, &msg, strict_der, low_s);
});
