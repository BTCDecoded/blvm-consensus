#![no_main]
//! `verify_script` with optional witness stack (covers legacy + witness script path).
use blvm_consensus::script::verify_script;
use blvm_consensus::types::ByteString;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let flags = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let rest = &data[4..];
    if rest.is_empty() {
        return;
    }
    let t = rest.len();
    let a = t / 3;
    let b = t / 3;
    let end_sig = a;
    let end_pk = (a + b).min(t);
    let script_sig: ByteString = rest[..end_sig].to_vec().into();
    let script_pubkey = rest[end_sig..end_pk].to_vec();
    let wit_slice = &rest[end_pk..];
    let witness = if wit_slice.is_empty() {
        None
    } else {
        Some(ByteString::from(wit_slice.to_vec()))
    };
    let _ = verify_script(&script_sig, &script_pubkey, witness.as_ref(), flags);
});
