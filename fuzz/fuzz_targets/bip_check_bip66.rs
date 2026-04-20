#![no_main]
//! BIP66 strict DER signature encoding.
use blvm_consensus::bip_validation::check_bip66_network;
use blvm_consensus::types::Network;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let height = u64::from(data.first().copied().unwrap_or(0)) + 400_000;
    let _ = check_bip66_network(data, height, Network::Mainnet);
});
