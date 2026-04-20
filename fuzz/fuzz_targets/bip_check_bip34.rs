#![no_main]
//! BIP34 coinbase height in block.
use blvm_consensus::bip_validation::check_bip34_network;
use blvm_consensus::serialization::block::deserialize_block_with_witnesses;
use blvm_consensus::types::Network;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok((block, _)) = deserialize_block_with_witnesses(data) else {
        return;
    };
    let height = u64::from_le_bytes(
        data.get(0..8)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0x00, 0x80, 0x25, 0, 0, 0, 0, 0]),
    );
    let _ = check_bip34_network(&block, height, Network::Mainnet);
});
