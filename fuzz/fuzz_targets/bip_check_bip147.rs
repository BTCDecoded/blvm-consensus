#![no_main]
//! BIP147 NULLDUMMY for OP_CHECKMULTISIG.
use blvm_consensus::bip_validation::{check_bip147_network, Bip147Network};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let mid = data.len() / 2;
    let script_sig = &data[..mid];
    let script_pubkey = &data[mid..];
    let height = u64::from(data[0]) + 500_000;
    let _ = check_bip147_network(script_sig, script_pubkey, height, Bip147Network::Mainnet);
});
