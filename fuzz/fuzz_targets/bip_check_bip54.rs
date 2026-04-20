#![no_main]
//! BIP54 coinbase locktime/sequence rules.
use blvm_consensus::bip_validation::check_bip54_coinbase;
use blvm_consensus::serialization::transaction::deserialize_transaction;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(tx) = deserialize_transaction(data) else {
        return;
    };
    let height = u64::from_le_bytes(
        data.get(0..8)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([100, 0, 0, 0, 0, 0, 0, 0]),
    );
    let _ = check_bip54_coinbase(&tx, height);
});
