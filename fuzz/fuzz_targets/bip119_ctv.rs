#![no_main]
//! BIP119 template hash and script helpers (`ctv` feature).
use blvm_consensus::bip119::{
    calculate_template_hash, extract_template_hash_from_script, is_ctv_script,
    validate_template_hash,
};
use blvm_consensus::serialization::transaction::deserialize_transaction;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = is_ctv_script(data);
    let _ = extract_template_hash_from_script(data);

    if let Ok(tx) = deserialize_transaction(data) {
        if tx.inputs.is_empty() || tx.outputs.is_empty() {
            return;
        }

        let idx = (data.first().copied().unwrap_or(0) as usize) % tx.inputs.len();
        let _ = calculate_template_hash(&tx, idx);
        let _ = calculate_template_hash(&tx, 0);

        let _ = validate_template_hash(&tx, idx, data);
        if data.len() >= 32 {
            let _ = validate_template_hash(&tx, idx, &data[..32]);
        }
    }

    if data.len() > 8 {
        if let Ok(tx) = deserialize_transaction(&data[4..]) {
            if !tx.inputs.is_empty() && !tx.outputs.is_empty() {
                let idx = (data[0] as usize) % tx.inputs.len();
                let _ = calculate_template_hash(&tx, idx);
            }
        }
    }
});
