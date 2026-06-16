//! Regression: per-tx SCRIPT_VERIFY_WITNESS gating must match Orange Paper §5.2.5.
//! Empty witness stacks in witness-serialized blocks must not alone enable witness rules.

use blvm_consensus::block::{
    calculate_script_flags_for_block_network, tx_has_nonempty_input_witness,
    tx_requires_witness_script_flags,
};
use blvm_consensus::opcodes::{OP_0, OP_1, PUSH_20_BYTES};
use blvm_consensus::script::flags::SCRIPT_VERIFY_WITNESS;
use blvm_consensus::types::Network;
use blvm_consensus::witness::Witness;
use blvm_consensus::{
    OutPoint, Transaction, TransactionInput, TransactionOutput, SEGWIT_ACTIVATION_MAINNET,
};

fn legacy_tx() -> Transaction {
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

fn p2wpkh_output_tx() -> Transaction {
    let mut spk = vec![OP_0, PUSH_20_BYTES];
    spk.extend_from_slice(&[0xcd; 20]);
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x02; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: spk.into(),
        }]
        .into(),
        lock_time: 0,
    }
}

#[test]
fn test_empty_witness_stacks_do_not_count_as_has_witness() {
    let empty_per_input: Vec<Witness> = vec![vec![], vec![]];
    assert!(!tx_has_nonempty_input_witness(Some(&empty_per_input)));
}

#[test]
fn test_nonempty_witness_stack_counts_as_has_witness() {
    let stacks: Vec<Witness> = vec![vec![], vec![vec![0x30, 0x01]]];
    assert!(tx_has_nonempty_input_witness(Some(&stacks)));
}

#[test]
fn test_legacy_tx_post_segwit_without_witness_does_not_require_witness_flag() {
    let tx = legacy_tx();
    let empty: Vec<Witness> = vec![vec![]];
    assert!(!tx_requires_witness_script_flags(
        &tx,
        tx_has_nonempty_input_witness(Some(&empty))
    ));

    let flags = calculate_script_flags_for_block_network(
        &tx,
        false,
        SEGWIT_ACTIVATION_MAINNET,
        Network::Mainnet,
    );
    assert_eq!(flags & SCRIPT_VERIFY_WITNESS, 0);
}

#[test]
fn test_segwit_output_tx_gets_witness_flag_without_input_witness() {
    let tx = p2wpkh_output_tx();
    assert!(tx_requires_witness_script_flags(
        &tx,
        tx_has_nonempty_input_witness(None)
    ));

    let flags = calculate_script_flags_for_block_network(
        &tx,
        false,
        SEGWIT_ACTIVATION_MAINNET,
        Network::Mainnet,
    );
    assert_ne!(flags & SCRIPT_VERIFY_WITNESS, 0);
}

#[test]
fn test_sort_merge_flag_parity_with_consensus_helper() {
    let tx = legacy_tx();
    let empty: Vec<Witness> = vec![vec![]];
    let has_real = tx_has_nonempty_input_witness(Some(&empty));
    let height_has_segwit = true;

    let step6_witness = height_has_segwit && tx_requires_witness_script_flags(&tx, has_real);
    let consensus_flags = calculate_script_flags_for_block_network(
        &tx,
        has_real,
        SEGWIT_ACTIVATION_MAINNET,
        Network::Mainnet,
    );
    let consensus_witness = consensus_flags & SCRIPT_VERIFY_WITNESS != 0;

    assert_eq!(
        step6_witness, consensus_witness,
        "sort-merge step6 witness flag must match calculate_script_flags_for_block_network"
    );
}
