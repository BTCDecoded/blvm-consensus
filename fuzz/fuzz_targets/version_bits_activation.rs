#![no_main]
//! BIP9-style version-bits activation height from a 2016-header window.
use blvm_consensus::serialization::block::deserialize_block_header;
use blvm_consensus::types::BlockHeader;
use blvm_consensus::version_bits::{activation_height_from_headers, bip54_deployment_mainnet};
use libfuzzer_sys::fuzz_target;

fn synthetic_header(seed: &[u8]) -> BlockHeader {
    BlockHeader {
        version: i64::from_le_bytes(seed.get(0..8).unwrap_or(&[0; 8]).try_into().unwrap())
            | (1i64 << 15),
        prev_block_hash: seed
            .get(8..40)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0u8; 32]),
        merkle_root: seed
            .get(40..72)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0u8; 32]),
        timestamp: u64::from_le_bytes(
            seed.get(72..80)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0x00, 0x94, 0x35, 0x5f, 0, 0, 0, 0]),
        ),
        bits: 0x1d00ffff,
        nonce: 0,
    }
}

fuzz_target!(|data: &[u8]| {
    let dep = bip54_deployment_mainnet();
    let mut headers: Vec<BlockHeader> = Vec::new();
    for chunk in data.chunks(80) {
        if chunk.len() == 80 {
            if let Ok(h) = deserialize_block_header(chunk) {
                headers.push(h);
            }
        }
    }
    let pad_seed: Vec<u8> = if data.len() >= 80 {
        data[..80].to_vec()
    } else {
        let mut v = data.to_vec();
        v.resize(80, 0);
        v
    };
    while headers.len() < 2016 {
        headers.push(synthetic_header(&pad_seed));
    }
    let window: Vec<BlockHeader> = headers.into_iter().take(2016).collect();
    let current_height = u64::from_le_bytes(
        data.get(0..8)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0x00, 0x80, 0x25, 0, 0, 0, 0, 0]),
    )
    .saturating_add(400_000);
    let current_time = u64::from_le_bytes(
        data.get(8..16)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0x00, 0x94, 0x35, 0x5f, 0, 0, 0, 0]),
    );
    let _ = activation_height_from_headers(&window, current_height, current_time, &dep);
});
