#![no_main]
//! BIP68 sequence field parsers and BIP113 median time (small header slices).
use blvm_consensus::bip113::{get_median_time_past, get_median_time_past_reversed};
use blvm_consensus::locktime::{
    extract_sequence_locktime_value, extract_sequence_type_flag, is_sequence_disabled,
};
use blvm_consensus::serialization::block::deserialize_block_header;
use blvm_consensus::types::BlockHeader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() >= 4 {
        let seq = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let _ = is_sequence_disabled(seq);
        let _ = extract_sequence_type_flag(seq);
        let _ = extract_sequence_locktime_value(seq);
    }
    let mut headers: Vec<BlockHeader> = Vec::new();
    for off in (0..data.len().saturating_sub(79)).take(64) {
        if let Ok(h) = deserialize_block_header(&data[off..off + 80]) {
            headers.push(h);
        }
    }
    if !headers.is_empty() {
        let _ = get_median_time_past(&headers);
        let _ = get_median_time_past_reversed(&headers);
    }
});
