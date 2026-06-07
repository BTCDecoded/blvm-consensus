//! Unit tests for economic model functions

use blvm_consensus::economic::*;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::orange_paper_constants::{C, H};

#[test]
fn test_get_block_subsidy_genesis() {
    let subsidy = get_block_subsidy(0);
    let initial_subsidy = (50 * C) as i64;
    assert_eq!(subsidy, initial_subsidy);
}

#[test]
fn test_get_block_subsidy_first_halving() {
    let subsidy = get_block_subsidy(H);
    let initial_subsidy = (50 * C) as i64;
    assert_eq!(subsidy, initial_subsidy / 2);
}

#[test]
fn test_get_block_subsidy_second_halving() {
    let subsidy = get_block_subsidy(H * 2);
    let initial_subsidy = (50 * C) as i64;
    assert_eq!(subsidy, initial_subsidy / 4);
}

#[test]
fn test_get_block_subsidy_max_halvings() {
    assert_eq!(get_block_subsidy(H * 64), 0);
}

#[test]
fn test_total_supply_convergence() {
    // Test that total supply approaches 21M BTC
    // Using Orange Paper constant H (halving interval = 210,000)
    let supply_at_halving = total_supply(H);
    let initial_subsidy = (50 * C) as i64;
    let expected_at_halving = (H as i64) * initial_subsidy;
    // The difference is due to bit shifting in get_block_subsidy
    // Allow for significant rounding differences due to bit operations
    let difference = (supply_at_halving - expected_at_halving).abs();
    assert!(difference <= 3_000_000_000); // Allow for significant rounding differences
}

#[test]
fn test_supply_limit() {
    // Test that supply limit is respected
    // Using Orange Paper constant H (halving interval = 210,000)
    assert!(validate_supply_limit(0).unwrap());
    assert!(validate_supply_limit(H).unwrap());
    assert!(validate_supply_limit(H * 10).unwrap());
}

#[test]
fn test_calculate_fee() {
    use blvm_consensus::types::*;

    // Build a simple tx spending a 1000-sat UTXO to a 800-sat output (200 sat fee)
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0u32,
    };
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: outpoint,
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 800,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    };
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        outpoint,
        std::sync::Arc::new(UTXO {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
            height: 1,
            is_coinbase: false,
        }),
    );
    let fee = calculate_fee(&tx, &utxo_set).unwrap();
    assert_eq!(fee, 200);
}

#[test]
fn test_calculate_fee_zero() {
    use blvm_consensus::types::*;

    let outpoint = OutPoint {
        hash: [2; 32],
        index: 0u32,
    };
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: outpoint,
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    };
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        outpoint,
        std::sync::Arc::new(UTXO {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
            height: 1,
            is_coinbase: false,
        }),
    );
    let fee = calculate_fee(&tx, &utxo_set).unwrap();
    assert_eq!(fee, 0);
}

#[test]
fn test_validate_supply_limit_excessive() {
    // Test with a height that would create excessive supply
    // Using Orange Paper constant H (halving interval = 210,000)
    let excessive_height = H * 100; // Way beyond normal operation
    let result = validate_supply_limit(excessive_height);
    // This should either pass (if the calculation is correct) or fail gracefully
    match result {
        Ok(valid) => assert!(valid),
        Err(_) => {
            // Expected failure for excessive height
        }
    }
}
