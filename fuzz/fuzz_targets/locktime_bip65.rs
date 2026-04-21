#![no_main]
//! BIP65 CLTV locktime comparison helpers.
use blvm_consensus::locktime::check_bip65;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let tx_lt = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let stack_lt = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let _ = check_bip65(tx_lt, stack_lt);
});
