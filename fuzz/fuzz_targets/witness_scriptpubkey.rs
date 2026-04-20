#![no_main]
//! Witness program extraction from scriptPubKey bytes.
use blvm_consensus::types::ByteString;
use blvm_consensus::witness::{
    extract_witness_program, extract_witness_version, validate_witness_program_length,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let script: ByteString = data.to_vec().into();
    if let Some(ver) = extract_witness_version(&script) {
        if let Some(prog) = extract_witness_program(&script, ver) {
            let _ = validate_witness_program_length(&prog, ver);
        }
    }
});
