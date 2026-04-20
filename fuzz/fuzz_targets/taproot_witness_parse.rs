#![no_main]
//! Taproot script-path witness parsing (control block + tapscript).
use blvm_consensus::taproot::parse_taproot_script_path_witness;
use blvm_consensus::types::ByteString;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }
    let mut out_key = [0u8; 32];
    out_key.copy_from_slice(&data[0..32]);
    let mut witness: Vec<ByteString> = Vec::new();
    let mut rest = &data[32..];
    while !rest.is_empty() && witness.len() < 40 {
        let take = (rest[0] as usize) % 256;
        rest = &rest[1..];
        let end = take.min(rest.len());
        witness.push(rest[..end].to_vec().into());
        rest = &rest[end..];
    }
    let _ = parse_taproot_script_path_witness(&witness, &out_key);
});
