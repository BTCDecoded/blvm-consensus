#![no_main]
//! Schnorr batch verify.
use blvm_consensus::secp256k1_backend::verify_schnorr_batch;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    let n = (data[0] as usize % 8) + 1;
    let mut sigs = Vec::with_capacity(n);
    let mut msgs: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut pks = Vec::with_capacity(n);
    let mut off = 1usize;
    for _ in 0..n {
        if off + 64 + 32 + 32 > data.len() {
            break;
        }
        let mut s = [0u8; 64];
        s.copy_from_slice(&data[off..off + 64]);
        sigs.push(s);
        msgs.push(data[off + 64..off + 64 + 32].to_vec());
        let mut pk = [0u8; 32];
        pk.copy_from_slice(&data[off + 64 + 32..off + 64 + 32 + 32]);
        pks.push(pk);
        off += 64 + 32 + 32;
    }
    if sigs.is_empty() {
        return;
    }
    let msg_refs: Vec<&[u8]> = msgs.iter().map(|m| m.as_slice()).collect();
    let _ = verify_schnorr_batch(&sigs, &msg_refs, &pks);
});
