#![no_main]
//! Locktime stack encoding: decode, encode round-trip, and type classification.
use blvm_consensus::locktime::{
    decode_locktime_value, encode_locktime_value, get_locktime_type, locktime_types_match,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let slice = &data[..data.len().min(5)];
    if let Some(v) = decode_locktime_value(slice) {
        if (0..=u32::MAX as i64).contains(&v) {
            let enc = encode_locktime_value(v as u32);
            let _ = decode_locktime_value(enc.as_slice());
        }
    }
    if data.len() >= 8 {
        let a = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let b = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let _ = locktime_types_match(a, b);
        let _ = get_locktime_type(a);
    }
});
