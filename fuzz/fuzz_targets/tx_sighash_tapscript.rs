#![no_main]
//! Tapscript sighash (BIP 342).
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::taproot::compute_tapscript_signature_hash;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
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
    let leaf_version = data.get(2).copied().unwrap_or(0xc0);
    let codesep_pos = u32::from_le_bytes(
        data.get(3..7)
            .and_then(|s| s.try_into().ok())
            .map(|a: [u8; 4]| a)
            .unwrap_or([0; 4]),
    );
    let tap_len = (data.get(7).copied().unwrap_or(0) as usize) % (data.len().saturating_sub(8).min(4096) + 1);
    let tapscript = &data[8..(8 + tap_len).min(data.len())];

    let mut prevout_values = vec![0i64; n];
    let mut prevout_scripts: Vec<Vec<u8>> = vec![vec![]; n];
    for i in 0..n {
        let off = (8 + tap_len + i * 12).min(data.len().saturating_sub(8));
        if data.len() >= off + 8 {
            prevout_values[i] = i64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        } else {
            prevout_values[i] = 1000;
        }
        let slen = data.get(off + 8).copied().unwrap_or(0) as usize % 128;
        if off + 9 + slen <= data.len() {
            prevout_scripts[i] = data[off + 9..off + 9 + slen].to_vec();
        }
    }
    let refs: Vec<&[u8]> = prevout_scripts.iter().map(|v| v.as_slice()).collect();
    let _ = compute_tapscript_signature_hash(
        &tx,
        input_index,
        &prevout_values,
        &refs,
        tapscript,
        leaf_version,
        codesep_pos,
        sighash_type,
    );
});
