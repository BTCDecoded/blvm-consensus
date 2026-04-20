#![no_main]
//! Legacy sighash: `compute_legacy_sighash_nocache` vs buffered (invariant: equal digest).
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::transaction_hash::{compute_legacy_sighash_buffered, compute_legacy_sighash_nocache};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
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
    let sighash_byte = data.get(1).copied().unwrap_or(1);
    let max_script = data.len().saturating_sub(4).min(10_000);
    let script_len = (data.get(2).copied().unwrap_or(0) as usize) % (max_script + 1);
    let script_start = 3usize.min(data.len());
    let script_end = (script_start + script_len).min(data.len());
    let script_code = &data[script_start..script_end];

    let a = compute_legacy_sighash_nocache(&tx, input_index, script_code, sighash_byte);
    let b = compute_legacy_sighash_buffered(&tx, input_index, script_code, sighash_byte);
    assert_eq!(a, b, "legacy sighash nocache vs buffered mismatch");
});
