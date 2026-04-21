#![no_main]
//! Per-transaction legacy sigop totals.
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::sigop::{get_legacy_sigop_count, get_legacy_sigop_count_accurate};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(tx) = deserialize_transaction(data) else {
        return;
    };
    let _ = get_legacy_sigop_count(&tx);
    let _ = get_legacy_sigop_count_accurate(&tx);
});
