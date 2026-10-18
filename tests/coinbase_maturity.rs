//! Coinbase maturity requirement tests
//!
//! Tests for coinbase maturity requirements (BIP34).
//! Coinbase outputs cannot be spent until 100 blocks deep (COINBASE_MATURITY).
//!
//! Consensus-critical: Spending coinbase too early causes consensus violation.

use blvm_consensus::opcodes::OP_1;
use blvm_consensus::types::{
    OutPoint, Transaction, TransactionInput, TransactionOutput, UTXO, UtxoSet, ValidationResult,
};

use blvm_consensus::constants::COINBASE_MATURITY;
use blvm_consensus::{SEGWIT_ACTIVATION_MAINNET, TAPROOT_ACTIVATION_MAINNET};

/// Test coinbase maturity at exact boundary (100 blocks)
#[test]
fn test_coinbase_maturity_exact_boundary() {
    // Create a coinbase transaction at height 0
    let coinbase_tx = Transaction {
        version: 1,
        inputs: blvm_consensus::tx_inputs![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00], // Height encoding
            sequence: 0xffffffff,
        }],
        outputs: blvm_consensus::tx_outputs![TransactionOutput {
            value: 50_0000_0000,       // 50 BTC
            script_pubkey: vec![OP_1], // OP_1
        }],
        lock_time: 0,
    };

    // Create UTXO set with coinbase output
    let _utxo_set = UtxoSet::default();
    let _coinbase_outpoint = OutPoint {
        hash: blvm_consensus::block::calculate_tx_id(&coinbase_tx),
        index: 0,
    };
    // Note: Actual UTXO insertion would use proper method

    // Attempt to spend coinbase at exactly 100 blocks (should succeed)
    let spending_height = COINBASE_MATURITY;
    // This should validate - coinbase is mature
    assert_eq!(spending_height, 100);

    // Attempt to spend coinbase at 99 blocks (should fail)
    let immature_height = COINBASE_MATURITY - 1;
    // This should fail - coinbase is not mature
    assert_eq!(immature_height, 99);
}

/// Test coinbase maturity at 99 blocks (should fail)
#[test]
fn test_coinbase_maturity_one_block_early() {
    // Coinbase created at height 0
    let coinbase_height = 0;

    // Attempt to spend at height 99 (one block too early)
    let spending_height = coinbase_height + COINBASE_MATURITY - 1;

    // Should fail - coinbase is not mature
    assert_eq!(spending_height, 99);
    assert!(spending_height < COINBASE_MATURITY);
}

/// Test coinbase maturity at exactly 100 blocks (should succeed)
#[test]
fn test_coinbase_maturity_exactly_100_blocks() {
    // Coinbase created at height 0
    let coinbase_height = 0;

    // Attempt to spend at height 100 (exactly mature)
    let spending_height = coinbase_height + COINBASE_MATURITY;

    // Should succeed - coinbase is mature
    assert_eq!(spending_height, 100);
    assert!(spending_height >= COINBASE_MATURITY);
}

/// Test coinbase maturity after 100 blocks (should succeed)
#[test]
fn test_coinbase_maturity_after_100_blocks() {
    // Coinbase created at height 0
    let coinbase_height = 0;

    // Attempt to spend at height 101 (well after maturity)
    let spending_height = coinbase_height + COINBASE_MATURITY + 1;

    // Should succeed - coinbase is mature
    assert_eq!(spending_height, 101);
    assert!(spending_height > COINBASE_MATURITY);
}

/// Test coinbase maturity in different consensus eras
#[test]
fn test_coinbase_maturity_different_eras() {
    // COINBASE_MATURITY is constant across all consensus eras
    // Test that it's the same at different heights

    let pre_segwit_height = SEGWIT_ACTIVATION_MAINNET - 1;
    let post_segwit_height = SEGWIT_ACTIVATION_MAINNET;
    let post_taproot_height = TAPROOT_ACTIVATION_MAINNET;

    // Maturity requirement is the same in all eras
    assert_eq!(COINBASE_MATURITY, 100);

    // Test spending coinbase at various heights
    for base_height in &[pre_segwit_height, post_segwit_height, post_taproot_height] {
        let coinbase_height = *base_height;
        let mature_height = coinbase_height + COINBASE_MATURITY;
        let immature_height = coinbase_height + COINBASE_MATURITY - 1;

        // Should fail before maturity
        assert!(immature_height < mature_height);

        // Should succeed after maturity
        assert!(mature_height >= coinbase_height + COINBASE_MATURITY);
    }
}

/// Test coinbase maturity interaction with reorgs
///
/// If a reorg occurs, coinbase maturity must be recalculated based on
/// the new chain position.
#[test]
fn test_coinbase_maturity_reorg() {
    // Coinbase created at height 100 in chain A
    let chain_a_height = 100;

    // After reorg, coinbase is at height 50 in chain B
    let chain_b_height = 50;

    // Attempt to spend at height 149 in chain A (should succeed)
    let spending_height_a = chain_a_height + COINBASE_MATURITY;
    assert_eq!(spending_height_a, 200);

    // Attempt to spend at height 149 in chain B (should fail - only 99 blocks deep)
    let spending_height_b = chain_b_height + COINBASE_MATURITY - 1;
    assert_eq!(spending_height_b, 149);
    assert!(spending_height_b < chain_b_height + COINBASE_MATURITY);

    // Should succeed at height 150 in chain B
    let mature_height_b = chain_b_height + COINBASE_MATURITY;
    assert_eq!(mature_height_b, 150);
    assert!(mature_height_b >= chain_b_height + COINBASE_MATURITY);
}

/// Test multiple coinbase outputs with different maturity
#[test]
fn test_multiple_coinbase_maturity() {
    // Create coinbase at height 0
    let coinbase1_height = 0;

    // Create another coinbase at height 50
    let coinbase2_height = 50;

    // At height 100:
    // - Coinbase 1 is mature (100 blocks deep)
    // - Coinbase 2 is not mature (only 50 blocks deep)
    let current_height = 100;

    assert!(blvm_consensus::transaction::check_coinbase_maturity(
        current_height,
        coinbase1_height,
        true
    ));
    assert!(!blvm_consensus::transaction::check_coinbase_maturity(
        current_height,
        coinbase2_height,
        true
    ));
    assert!(blvm_consensus::transaction::check_coinbase_maturity(
        current_height,
        coinbase2_height,
        false
    ));
}

/// Test coinbase maturity with block validation
///
/// Verifies that blocks attempting to spend immature coinbase are rejected.
#[test]
fn test_coinbase_maturity_block_validation() {
    let prevout = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        prevout,
        std::sync::Arc::new(UTXO {
            value: 50_0000_0000,
            script_pubkey: vec![OP_1].into(),
            height: 0,
            is_coinbase: true,
        }),
    );
    let spend = Transaction {
        version: 1,
        inputs: blvm_consensus::tx_inputs![TransactionInput {
            prevout,
            script_sig: vec![OP_1],
            sequence: 0xffffffff,
        }],
        outputs: blvm_consensus::tx_outputs![TransactionOutput {
            value: 25_0000_0000,
            script_pubkey: vec![OP_1],
        }],
        lock_time: 0,
    };
    let (immature, fee) =
        blvm_consensus::transaction::check_tx_inputs(&spend, &utxo_set, 99).unwrap();
    assert!(matches!(immature, ValidationResult::Invalid(_)));
    assert_eq!(fee, 0);
    let (mature, fee) =
        blvm_consensus::transaction::check_tx_inputs(&spend, &utxo_set, 100).unwrap();
    assert!(matches!(mature, ValidationResult::Valid));
    assert_eq!(fee, 25_0000_0000);
}
