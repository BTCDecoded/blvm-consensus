#![no_main]
//! Sigop counting in legacy script bytecode.
use blvm_consensus::sigop::count_sigops_in_script;
use blvm_consensus::types::ByteString;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let script: ByteString = data.to_vec().into();
    let accurate = data.first().map(|b| (b & 1) != 0).unwrap_or(false);
    let _ = count_sigops_in_script(&script, accurate);
});
