#![no_main]
//! Taproot key-path sighash (BIP 341 SigMsg).
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::taproot::compute_taproot_signature_hash;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let Ok(tx) = deserialize_transaction(data) else {
        return;
    };
    if tx.inputs.is_empty() {
        return;
    }
    let n = tx.inputs.len();
    let input_index = (data.get(0).copied().unwrap_or(0) as usize) % n;
    let sighash_type = data.get(1).copied().unwrap_or(1);
    let mut prevout_values = vec![0i64; n];
    let mut prevout_scripts: Vec<Vec<u8>> = vec![vec![]; n];
    for i in 0..n {
        let off = 2 + i * 16;
        if data.len() >= off + 8 {
            prevout_values[i] = i64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        } else {
            prevout_values[i] = 1000;
        }
        let slen = data
            .get(off + 8)
            .copied()
            .unwrap_or(0) as usize
            % 256;
        let base = off + 9;
        if base + slen <= data.len() {
            prevout_scripts[i] = data[base..base + slen].to_vec();
        }
    }
    let refs: Vec<&[u8]> = prevout_scripts.iter().map(|v| v.as_slice()).collect();
    let _ = compute_taproot_signature_hash(
        &tx,
        input_index,
        &prevout_values,
        &refs,
        sighash_type,
    );
});
