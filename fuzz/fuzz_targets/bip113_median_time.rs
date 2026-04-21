#![no_main]
//! BIP113 median time-past over fuzz-built header chains.
use blvm_consensus::bip113::{get_median_time_past, get_median_time_past_reversed};
use blvm_consensus::types::BlockHeader;
use libfuzzer_sys::fuzz_target;

fn header_from_chunk(chunk: &[u8]) -> BlockHeader {
    if chunk.len() < 80 {
        return BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            bits: 0,
            nonce: 0,
        };
    }
    BlockHeader {
        version: i64::from_le_bytes([
            chunk[0], chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
            chunk.get(3).copied().unwrap_or(0),
            chunk.get(4).copied().unwrap_or(0),
            chunk.get(5).copied().unwrap_or(0),
            chunk.get(6).copied().unwrap_or(0),
            chunk.get(7).copied().unwrap_or(0),
        ]),
        prev_block_hash: chunk
            .get(8..40)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0; 32]),
        merkle_root: chunk
            .get(40..72)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0; 32]),
        timestamp: u64::from_le_bytes([
            chunk[72], chunk[73], chunk[74], chunk[75], chunk[76], chunk[77], chunk[78], chunk[79],
        ]),
        bits: u32::from_le_bytes([
            chunk.get(80).copied().unwrap_or(0),
            chunk.get(81).copied().unwrap_or(0),
            chunk.get(82).copied().unwrap_or(0),
            chunk.get(83).copied().unwrap_or(0),
        ]) as u64,
        nonce: u32::from_le_bytes([
            chunk.get(84).copied().unwrap_or(0),
            chunk.get(85).copied().unwrap_or(0),
            chunk.get(86).copied().unwrap_or(0),
            chunk.get(87).copied().unwrap_or(0),
        ]) as u64,
    }
}

fuzz_target!(|data: &[u8]| {
    const CHUNK: usize = 88;
    let mut headers: Vec<BlockHeader> = Vec::new();
    for chunk in data.chunks(CHUNK).take(32) {
        headers.push(header_from_chunk(chunk));
    }
    if headers.is_empty() {
        return;
    }

    let _ = get_median_time_past(&headers);

    let newest_first: Vec<BlockHeader> = headers.iter().rev().cloned().collect();
    let _ = get_median_time_past_reversed(&newest_first);
});
