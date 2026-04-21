#![no_main]
//! Tapscript sigop counting.
use blvm_consensus::sigop::count_tapscript_sigops;
use blvm_consensus::types::ByteString;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let script: ByteString = data.to_vec().into();
    let _ = count_tapscript_sigops(&script);
});
