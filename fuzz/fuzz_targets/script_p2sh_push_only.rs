#![no_main]
//! P2SH scriptSig push-only predicate.
use blvm_consensus::script::p2sh_push_only_check;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = p2sh_push_only_check(data);
});
