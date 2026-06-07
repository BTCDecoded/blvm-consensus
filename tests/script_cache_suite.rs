//! COV-C-01d: Block script flag calculation and exception table coverage.

use blvm_consensus::activation::ForkActivationTable;
use blvm_consensus::block::{
    calculate_base_script_flags_for_block_network, calculate_script_flags_for_block_network,
    get_block_script_flags, get_block_script_verify_flags_core, script_flag_exceptions_lookup,
};
use blvm_consensus::constants::{
    BIP112_CSV_ACTIVATION_MAINNET, BIP16_P2SH_ACTIVATION_MAINNET, BIP66_ACTIVATION_MAINNET,
    SEGWIT_ACTIVATION_MAINNET,
};
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::script::flags::{SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_WITNESS};
use blvm_consensus::types::Network;
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

fn rpc_block_hash(hex_str: &str) -> [u8; 32] {
    let mut bytes = hex::decode(hex_str).expect("valid block hash hex");
    assert_eq!(bytes.len(), 32);
    bytes.reverse();
    bytes.try_into().expect("32-byte hash")
}

fn sample_tx() -> Transaction {
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x01; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

#[test]
fn test_script_flag_exception_mainnet_bip16_block() {
    let hash = rpc_block_hash("00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22");
    assert_eq!(
        script_flag_exceptions_lookup(&hash, Network::Mainnet),
        Some(0)
    );
}

#[test]
fn test_script_flag_exception_mainnet_taproot_block() {
    let hash = rpc_block_hash("0000000000000000000f14c35b2d841e986ab5441de8c585d5ffe55ea1e395ad");
    assert_eq!(
        script_flag_exceptions_lookup(&hash, Network::Mainnet),
        Some(SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS)
    );
}

#[test]
fn test_script_flag_exception_testnet_bip16_block() {
    let hash = rpc_block_hash("00000000dd30457c001f4095d208cc1296b0eed002427aa599874af7a432b105");
    assert_eq!(
        script_flag_exceptions_lookup(&hash, Network::Testnet),
        Some(0)
    );
}

#[test]
fn test_script_flag_exception_unknown_hash_returns_none() {
    assert!(script_flag_exceptions_lookup(&[0xab; 32], Network::Mainnet).is_none());
    assert!(script_flag_exceptions_lookup(&[0xcd; 32], Network::Regtest).is_none());
}

#[test]
fn test_base_script_flags_enable_p2sh_after_bip16() {
    let pre = calculate_base_script_flags_for_block_network(
        BIP16_P2SH_ACTIVATION_MAINNET.saturating_sub(1),
        Network::Mainnet,
    );
    let post = calculate_base_script_flags_for_block_network(
        BIP16_P2SH_ACTIVATION_MAINNET,
        Network::Mainnet,
    );
    assert_eq!(pre & SCRIPT_VERIFY_P2SH, 0);
    assert_ne!(post & SCRIPT_VERIFY_P2SH, 0);
}

#[test]
fn test_base_script_flags_enable_dersig_after_bip66() {
    let pre = calculate_base_script_flags_for_block_network(
        BIP66_ACTIVATION_MAINNET.saturating_sub(1),
        Network::Mainnet,
    );
    let post =
        calculate_base_script_flags_for_block_network(BIP66_ACTIVATION_MAINNET, Network::Mainnet);
    assert_eq!(pre & 0x04, 0);
    assert_ne!(post & 0x04, 0);
}

#[test]
fn test_base_script_flags_enable_csv_after_bip112() {
    let pre = calculate_base_script_flags_for_block_network(
        BIP112_CSV_ACTIVATION_MAINNET.saturating_sub(1),
        Network::Mainnet,
    );
    let post = calculate_base_script_flags_for_block_network(
        BIP112_CSV_ACTIVATION_MAINNET,
        Network::Mainnet,
    );
    assert_eq!(pre & 0x400, 0);
    assert_ne!(post & 0x400, 0);
}

#[test]
fn test_get_block_script_flags_uses_exception_table() {
    let hash = rpc_block_hash("00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22");
    let tx = sample_tx();
    let flags = get_block_script_flags(&hash, &tx, false, 500_000, Network::Mainnet);
    assert_eq!(flags, 0, "exception block must override per-tx flags");
}

#[test]
fn test_get_block_script_flags_without_exception_uses_height() {
    let tx = sample_tx();
    let flags = get_block_script_flags(
        &[0x99; 32],
        &tx,
        false,
        BIP66_ACTIVATION_MAINNET,
        Network::Mainnet,
    );
    assert_ne!(flags & 0x04, 0);
}

#[test]
fn test_per_tx_script_flags_add_witness_at_segwit_height() {
    let tx = sample_tx();
    let without_witness = calculate_script_flags_for_block_network(
        &tx,
        false,
        SEGWIT_ACTIVATION_MAINNET.saturating_sub(1),
        Network::Mainnet,
    );
    let with_witness = calculate_script_flags_for_block_network(
        &tx,
        true,
        SEGWIT_ACTIVATION_MAINNET,
        Network::Mainnet,
    );
    assert_eq!(without_witness & SCRIPT_VERIFY_WITNESS, 0);
    assert_ne!(with_witness & SCRIPT_VERIFY_WITNESS, 0);
}

#[test]
fn test_get_block_script_verify_flags_core_regtest_baseline() {
    let table = ForkActivationTable::from_network(Network::Regtest);
    let flags = get_block_script_verify_flags_core(&[0x01; 32], 100, &table, Network::Regtest);
    assert_ne!(flags, 0);
}

#[test]
fn test_get_block_script_flags_regtest_without_exception() {
    let tx = sample_tx();
    let flags = get_block_script_flags(&[0x42; 32], &tx, false, 100, Network::Regtest);
    assert_ne!(flags, 0);
}

#[test]
fn test_calculate_script_flags_regtest_enables_witness_when_requested() {
    let tx = sample_tx();
    let flags = calculate_script_flags_for_block_network(&tx, true, 500, Network::Regtest);
    assert_ne!(flags & SCRIPT_VERIFY_WITNESS, 0);
}

#[test]
fn test_get_block_script_verify_flags_core_taproot_exception() {
    let hash = rpc_block_hash("0000000000000000000f14c35b2d841e986ab5441de8c585d5ffe55ea1e395ad");
    let table = ForkActivationTable::from_network(Network::Mainnet);
    let flags = get_block_script_verify_flags_core(&hash, 800_000, &table, Network::Mainnet);
    assert_ne!(flags & SCRIPT_VERIFY_P2SH, 0);
    assert_ne!(flags & SCRIPT_VERIFY_WITNESS, 0);
}

#[test]
fn test_base_script_flags_enable_nulldummy_at_segwit() {
    let pre = calculate_base_script_flags_for_block_network(
        SEGWIT_ACTIVATION_MAINNET.saturating_sub(1),
        Network::Mainnet,
    );
    let post =
        calculate_base_script_flags_for_block_network(SEGWIT_ACTIVATION_MAINNET, Network::Mainnet);
    assert_eq!(pre & 0x10, 0);
    assert_ne!(post & 0x10, 0);
}

#[test]
fn test_script_flags_include_taproot_output_type_at_activation() {
    use blvm_consensus::constants::TAPROOT_ACTIVATION_MAINNET;
    use blvm_consensus::opcodes::PUSH_32_BYTES;
    let mut spk = vec![OP_1, PUSH_32_BYTES];
    spk.extend_from_slice(&[0x11; 32]);
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x01; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: spk.into(),
        }]
        .into(),
        lock_time: 0,
    };
    let flags = calculate_script_flags_for_block_network(
        &tx,
        false,
        TAPROOT_ACTIVATION_MAINNET,
        Network::Mainnet,
    );
    assert_ne!(flags & 0x8000, 0);
}
