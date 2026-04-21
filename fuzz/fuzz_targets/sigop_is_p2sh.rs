#![no_main]
//! P2SH pattern detection.
use blvm_consensus::sigop::is_pay_to_script_hash;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = is_pay_to_script_hash(data);
});
