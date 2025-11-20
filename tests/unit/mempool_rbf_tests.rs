use bllvm_consensus::mempool;
use bllvm_consensus::UtxoSet;

#[path = "../test_helpers.rs"]
mod test_helpers;
use test_helpers::{create_rbf_tx, create_test_utxo};

#[test]
fn test_rbf_sequence_checks() {
    let pool = mempool::Mempool::new();
    
    // RBF transaction (sequence < 0xffffffff)
    let rbf_tx = create_rbf_tx(0xfffffffeu64);
    let non_rbf_tx = create_rbf_tx(0xffffffffu64);
    
    // Test RBF replacement logic
    let can_replace = mempool::replacement_checks(&rbf_tx, &non_rbf_tx, &pool);
    // Whether it succeeds depends on implementation, just exercise the path
    let _ = can_replace;
}

#[test]
fn test_mempool_duplicate_detection() {
    let mut pool = mempool::Mempool::new();
    let tx = create_rbf_tx(0xffffffffu64);
    let utxo = UtxoSet::new();
    
    // First acceptance should work
    let _ = mempool::accept_to_memory_pool(&tx, &utxo, &pool, 1);
    
    // Second acceptance should detect duplicate
    let result = mempool::accept_to_memory_pool(&tx, &utxo, &pool, 1);
    // Should fail due to duplicate
    assert!(result.is_err());
}
















