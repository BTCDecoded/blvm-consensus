#![no_main]
//! Batch sighash (`batch_compute_sighashes`, `batch_compute_legacy_sighashes`).
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::transaction_hash::{
    batch_compute_legacy_sighashes, batch_compute_sighashes, SighashType,
};
use blvm_consensus::types::{Transaction, TransactionOutput};
use libfuzzer_sys::fuzz_target;

fn synth_prevouts(tx: &Transaction, data: &[u8]) -> Vec<TransactionOutput> {
    let n = tx.inputs.len();
    (0..n)
        .map(|i| {
            let off = i * 24;
            let value = if data.len() >= off + 8 {
                i64::from_le_bytes(data[off..off + 8].try_into().unwrap())
            } else {
                1000 + i as i64
            };
            let slen = data
                .get(off + 8)
                .copied()
                .unwrap_or(0) as usize
                % 128;
            let base = off + 9;
            let script = if base + slen <= data.len() {
                data[base..base + slen].to_vec()
            } else {
                vec![0x51]
            };
            TransactionOutput {
                value,
                script_pubkey: script.into(),
            }
        })
        .collect()
}

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
    let prevouts = synth_prevouts(&tx, data);
    let st = SighashType::from_byte(data.get(0).copied().unwrap_or(1));
    let _ = batch_compute_sighashes(&tx, &prevouts, st);

    let prevout_values: Vec<i64> = prevouts.iter().map(|o| o.value).collect();
    let prevout_script_pubkeys: Vec<&[u8]> = prevouts
        .iter()
        .map(|o| o.script_pubkey.as_slice())
        .collect();
    let n_specs = (data.get(1).copied().unwrap_or(1) as usize % n).max(1).min(n);
    let mut specs: Vec<(usize, u8, &[u8])> = Vec::with_capacity(n_specs);
    for k in 0..n_specs {
        let input_index = (data.get(2 + k).copied().unwrap_or(0) as usize) % n;
        let sighash_byte = data.get(10 + k).copied().unwrap_or(1);
        let script = prevout_script_pubkeys[input_index];
        specs.push((input_index, sighash_byte, script));
    }
    let _ = batch_compute_legacy_sighashes(
        &tx,
        &prevout_values,
        &prevout_script_pubkeys,
        &specs,
    );
});
