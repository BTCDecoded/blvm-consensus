#![no_main]
//! BIP90 buried deployments: minimum block version vs height.
use blvm_consensus::bip_validation::check_bip90_network;
use blvm_consensus::types::Network;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    let ver = i64::from_le_bytes(data[0..8].try_into().unwrap());
    let height = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let _ = check_bip90_network(ver, height, Network::Mainnet);
});
