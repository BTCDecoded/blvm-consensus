#![no_main]
//! BIP143 (SegWit v0) sighash.
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::transaction_hash::calculate_bip143_sighash;
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
    let amount = i64::from_le_bytes(
        data.get(2..10)
            .and_then(|s| s.try_into().ok())
            .map(|a: [u8; 8]| a)
            .unwrap_or([0x00, 0xe1, 0xf5, 0x05, 0, 0, 0, 0]),
    );
    let max_script = data.len().saturating_sub(12).min(10_000);
    let script_len = (data.get(10).copied().unwrap_or(0) as usize) % (max_script + 1);
    let script_start = 11usize.min(data.len());
    let script_end = (script_start + script_len).min(data.len());
    let script_code = &data[script_start..script_end];

    let _ = calculate_bip143_sighash(&tx, input_index, script_code, amount, sighash_type, None);
});
