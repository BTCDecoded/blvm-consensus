//! Integration checks: `ForkActivationTable` testnet3 heights match `blvm-primitives` / Core.
//!
//! Mainnet coverage lives in other tests; this file locks testnet CSV / SegWit / Taproot / BIP65.

use blvm_consensus::activation::{taproot_activation_height, ForkActivationTable, IsForkActive};
use blvm_consensus::types::{ForkId, Network};
use blvm_consensus::{
    BIP112_CSV_ACTIVATION_TESTNET, BIP147_ACTIVATION_TESTNET, BIP65_ACTIVATION_TESTNET,
    SEGWIT_ACTIVATION_TESTNET, TAPROOT_ACTIVATION_TESTNET,
};

#[test]
fn testnet_fork_table_matches_primitives_constants() {
    let t = ForkActivationTable::from_network(Network::Testnet);
    assert_eq!(t.bip65, BIP65_ACTIVATION_TESTNET);
    assert_eq!(t.bip112, BIP112_CSV_ACTIVATION_TESTNET);
    assert_eq!(t.bip147, BIP147_ACTIVATION_TESTNET);
    assert_eq!(t.segwit, SEGWIT_ACTIVATION_TESTNET);
    assert_eq!(t.taproot, TAPROOT_ACTIVATION_TESTNET);
}

#[test]
fn testnet_segwit_same_height_as_bip147_on_testnet3() {
    assert_eq!(
        SEGWIT_ACTIVATION_TESTNET, BIP147_ACTIVATION_TESTNET,
        "Core ties NULLDUMMY (BIP147) to SegWit deployment on testnet3"
    );
}

#[test]
fn testnet_taproot_activation_helper_matches_table() {
    assert_eq!(
        taproot_activation_height(Network::Testnet),
        TAPROOT_ACTIVATION_TESTNET
    );
    let t = ForkActivationTable::from_network(Network::Testnet);
    assert!(t.is_fork_active(ForkId::Taproot, TAPROOT_ACTIVATION_TESTNET));
    assert!(!t.is_fork_active(ForkId::Taproot, TAPROOT_ACTIVATION_TESTNET - 1));
}

#[test]
fn testnet_csv_after_mainnet_csv_height_still_pre_segwit() {
    // Mainnet CSV is 419328; testnet3 CSV is 770112 — do not treat mainnet height as testnet-active.
    let t = ForkActivationTable::from_network(Network::Testnet);
    let mainnet_csv = blvm_consensus::BIP112_CSV_ACTIVATION_MAINNET;
    assert!(
        mainnet_csv < BIP112_CSV_ACTIVATION_TESTNET,
        "testnet CSV must be after mainnet CSV for this invariant"
    );
    assert!(
        !t.is_fork_active(ForkId::Bip112, mainnet_csv),
        "testnet must not enable CSV at mainnet CSV height"
    );
}

#[test]
fn testnet_segwit_after_mainnet_segwit_height_still_pre_segwit() {
    let t = ForkActivationTable::from_network(Network::Testnet);
    let mainnet_segwit = blvm_consensus::SEGWIT_ACTIVATION_MAINNET;
    assert!(
        mainnet_segwit < SEGWIT_ACTIVATION_TESTNET,
        "testnet SegWit is later than mainnet"
    );
    assert!(!t.is_fork_active(ForkId::SegWit, mainnet_segwit));
}
