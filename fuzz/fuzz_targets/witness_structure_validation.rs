#![no_main]
//! Witness stack shape checks (SegWit vs Taproot script path).
use blvm_consensus::types::ByteString;
use blvm_consensus::witness::{validate_segwit_witness_structure, validate_taproot_witness_structure};
use libfuzzer_sys::fuzz_target;

fn build_witness(data: &[u8]) -> Vec<ByteString> {
    let mut w = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        let len = (data[i] as usize) % 520;
        i += 1;
        let end = (i + len).min(data.len());
        w.push(data[i..end].to_vec().into());
        i = end;
        if w.len() > 64 {
            break;
        }
    }
    w
}

fuzz_target!(|data: &[u8]| {
    let w = build_witness(data);
    let _ = validate_segwit_witness_structure(&w);
    let script_path = data.first().map(|b| (b & 1) != 0).unwrap_or(false);
    let _ = validate_taproot_witness_structure(&w, script_path);
});
