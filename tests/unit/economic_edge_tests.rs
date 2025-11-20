use bllvm_consensus::economic;
use bllvm_consensus::{UtxoSet, OutPoint};

#[path = "../test_helpers.rs"]
mod test_helpers;
use test_helpers::{create_tx_with_value, create_coinbase_tx, create_test_utxo};

#[test]
fn test_calculate_fee_coinbase() {
    let coinbase_tx = create_coinbase_tx(50_000_000_000);
    
    let utxo = UtxoSet::new();
    let fee = economic::calculate_fee(&coinbase_tx, &utxo);
    
    // Coinbase transactions should have zero fee
    assert!(fee.is_ok());
    assert_eq!(fee.unwrap(), 0);
}

#[test]
fn test_calculate_fee_negative() {
    let tx = create_tx_with_value(1000);
    let mut utxo = UtxoSet::new();
    
    // Create UTXO with less value than transaction output
    utxo.insert(
        OutPoint { hash: [1; 32], index: 0 },
        bllvm_consensus::UTXO { value: 500, script_pubkey: vec![0x51], height: 1 }
    );
    
    let fee = economic::calculate_fee(&tx, &utxo);
    
    // Should fail due to negative fee (outputs > inputs)
    assert!(fee.is_err());
}

// Note: High halving tests are covered by:
// - economic_tests.rs: test_get_block_subsidy_max_halvings (tests HALVING_INTERVAL * 64)
// - consensus_property_tests.rs: prop_block_subsidy_halving_schedule (property test up to 10 halvings)
// This test was redundant and removed.

#[test]
fn test_total_supply_convergence() {
    // Test that total supply converges to 21M BTC
    let max_height = 2_100_000;
    let total_supply = economic::total_supply(max_height);
    
    // Should be close to 21M BTC (2,100,000,000,000 satoshis)
    assert!(total_supply <= 2_100_000_000_000);
}

#[test]
fn test_supply_limit() {
    // Test supply limit constant
    let limit = bllvm_consensus::MAX_MONEY;
    
    // Should be 21M BTC in satoshis
    assert_eq!(limit, 2_100_000_000_000);
}

























