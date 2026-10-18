use blvm_consensus::UtxoSet;
use blvm_consensus::constants::MIN_RELAY_FEE;
use blvm_consensus::mempool;
use blvm_consensus::types::Network;

#[path = "../test_helpers.rs"]
mod test_helpers;
use test_helpers::{create_rbf_tx, create_test_utxo};

#[test]
fn test_rbf_sequence_checks() {
    let pool = mempool::Mempool::new();
    let (utxo, _) = create_test_utxo(10_000);

    // RBF transaction (sequence < 0xfffffffe)
    let rbf_tx = create_rbf_tx(0xfffffffd);
    let non_rbf_tx = create_rbf_tx(0xffffffffu64);

    assert!(
        !mempool::replacement_checks(&rbf_tx, &non_rbf_tx, &utxo, &pool)
            .expect("replacement check"),
        "an existing transaction that does not signal replacement cannot be replaced"
    );

    let mut existing = create_rbf_tx(0xfffffffd);
    existing.outputs[0].value = 9_000;
    let mut replacement = existing.clone();
    replacement.outputs[0].value = 10_000 - 1_000 - MIN_RELAY_FEE - 1;
    assert!(
        mempool::replacement_checks(&replacement, &existing, &utxo, &pool)
            .expect("replacement check"),
        "a higher fee for the same output replaces a signaling transaction"
    );
}

#[test]
fn test_mempool_duplicate_detection() {
    let mut pool = mempool::Mempool::new();
    let tx = create_rbf_tx(0xffffffffu64);
    let utxo = UtxoSet::default();

    // Seed mempool so accept sees a duplicate (accept does not mutate pool)
    let tx_id = mempool::calculate_tx_id(&tx);
    pool.insert(tx_id);

    let result =
        mempool::accept_to_memory_pool(&tx, None, &utxo, &pool, 1, None, Network::Mainnet, None);
    assert!(
        matches!(result, Ok(mempool::MempoolResult::Rejected(ref r)) if r.contains("already")),
        "duplicate mempool entry should be rejected: {result:?}"
    );
}
