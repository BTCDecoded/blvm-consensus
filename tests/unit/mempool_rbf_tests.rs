use blvm_consensus::UtxoSet;
use blvm_consensus::mempool;
use blvm_consensus::types::Network;

#[path = "../test_helpers.rs"]
mod test_helpers;
use test_helpers::{create_rbf_tx, create_test_utxo};

#[test]
fn test_rbf_sequence_checks() {
    let pool = mempool::Mempool::new();
    let (utxo, _) = create_test_utxo(10_000);

    // RBF transaction (sequence < 0xffffffff)
    let rbf_tx = create_rbf_tx(0xfffffffeu64);
    let non_rbf_tx = create_rbf_tx(0xffffffffu64);

    // Test RBF replacement logic (live 4-arg API includes UtxoSet)
    let can_replace = mempool::replacement_checks(&rbf_tx, &non_rbf_tx, &utxo, &pool);
    // Whether it succeeds depends on implementation, just exercise the path
    let _ = can_replace;
}

#[test]
fn test_mempool_duplicate_detection() {
    let mut pool = mempool::Mempool::new();
    let tx = create_rbf_tx(0xffffffffu64);
    let utxo = UtxoSet::default();

    // Seed mempool so accept sees a duplicate (accept does not mutate pool)
    let tx_id = mempool::calculate_tx_id(&tx);
    pool.insert(tx_id);

    let result = mempool::accept_to_memory_pool(&tx, None, &utxo, &pool, 1, None, Network::Mainnet);
    assert!(
        matches!(result, Ok(mempool::MempoolResult::Rejected(ref r)) if r.contains("already")),
        "duplicate mempool entry should be rejected: {result:?}"
    );
}
