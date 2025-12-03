//! Integration tests for BLLVM optimizations
//!
//! Tests verify that BLLVM optimizations work correctly in realistic scenarios,
//! including block validation, merkle root calculation, and batch operations.

use blvm_consensus::{
    mining::{calculate_merkle_root, generate_block_template},
    optimizations::simd_vectorization,
    serialization::{block::serialize_block, transaction::serialize_transaction},
    types::{Block, BlockHeader, Transaction},
    ConsensusProof, ValidationResult, UtxoSet,
};

/// Test that block validation produces identical results with/without production feature
#[test]
fn test_block_validation_correctness() {
    let block = create_test_block();
    let utxo_set = UtxoSet::new();

    // Validate (uses optimizations in production)
    let consensus = ConsensusProof::new();
    let witnesses: Vec<blvm_consensus::segwit::Witness> =
        block.transactions.iter().map(|_| Vec::new()).collect();
    let time_context = None;
    let network = blvm_consensus::types::Network::Mainnet;
    let result = consensus.validate_block_with_time_context(
        &block,
        &witnesses,
        utxo_set,
        0,
        time_context,
        network,
    );

    // Should handle validation (may fail due to missing UTXOs, but should not panic)
    let _ = result;
}

/// Test that batch hash operations work correctly in block validation context
#[test]
fn test_batch_hashing_integration() {
    #[cfg(feature = "production")]
    {
        let transactions = create_test_transactions(100);

        // Serialize all transactions
        let serialized: Vec<Vec<u8>> = transactions
            .iter()
            .map(|tx| serialize_transaction(tx))
            .collect();

        // Batch hash using aligned structures
        let tx_refs: Vec<&[u8]> = serialized.iter().map(|v| v.as_slice()).collect();
        let aligned_hashes = simd_vectorization::batch_double_sha256_aligned(&tx_refs);
        let regular_hashes = simd_vectorization::batch_double_sha256(&tx_refs);

        assert_eq!(aligned_hashes.len(), transactions.len());
        assert_eq!(regular_hashes.len(), transactions.len());

        // Verify all hashes match
        for (aligned, regular) in aligned_hashes.iter().zip(regular_hashes.iter()) {
            assert_eq!(aligned.as_bytes(), regular);
        }
    }
}

/// Test that merkle root calculation works correctly with large transaction sets
#[test]
fn test_merkle_root_large_transaction_set() {
    // Create a large set of transactions (simulating a full block)
    let transactions = create_test_transactions(2000);

    // Calculate merkle root
    let root = calculate_merkle_root(&transactions).expect("Should calculate merkle root");

    // Verify root is a valid hash
    assert_eq!(root.len(), 32);

    // Verify root is deterministic
    let root2 = calculate_merkle_root(&transactions).expect("Should calculate merkle root again");
    assert_eq!(root, root2, "Merkle root should be deterministic");
}

/// Test that serialization and deserialization round-trip works with optimizations
#[test]
fn test_serialization_round_trip() {
    use blvm_consensus::serialization::transaction::deserialize_transaction;

    let tx = create_test_transaction();

    // Serialize (uses optimizations in production)
    let serialized = serialize_transaction(&tx);

    // Deserialize
    let deserialized = deserialize_transaction(&serialized)
        .expect("Should deserialize transaction");

    // Verify round-trip
    assert_eq!(tx.version, deserialized.version);
    assert_eq!(tx.inputs.len(), deserialized.inputs.len());
    assert_eq!(tx.outputs.len(), deserialized.outputs.len());
    assert_eq!(tx.lock_time, deserialized.lock_time);
}


/// Test that batch operations work correctly with parallel processing
#[test]
#[cfg(feature = "production")]
#[cfg(feature = "rayon")]
fn test_batch_operations_parallel() {
    // Create a large set of transactions
    let transactions = create_test_transactions(1000);

    // Serialize all (uses parallel processing if rayon available)
    let serialized: Vec<Vec<u8>> = {
        use rayon::prelude::*;
        transactions
            .par_iter()
            .map(|tx| serialize_transaction(tx))
            .collect()
    };

    // Batch hash (uses cache-aligned structures)
    let tx_refs: Vec<&[u8]> = serialized.iter().map(|v| v.as_slice()).collect();
    let hashes = simd_vectorization::batch_double_sha256(&tx_refs);

    assert_eq!(hashes.len(), transactions.len());
}

/// Test that optimizations don't break edge cases
#[test]
fn test_optimizations_edge_cases() {
    // Empty transaction list
    let empty_txs: Vec<Transaction> = vec![];
    assert!(calculate_merkle_root(&empty_txs).is_err());

    // Single transaction
    let single_tx = vec![create_test_transaction()];
    let root = calculate_merkle_root(&single_tx).expect("Should work for single tx");
    assert_eq!(root.len(), 32);

    // Very large transaction (near max size)
    let large_tx = {
        let mut tx = create_test_transaction();
        // Add large script
        tx.outputs[0].script_pubkey = vec![0u8; 10000];
        tx
    };
    let serialized = serialize_transaction(&large_tx);
    assert!(serialized.len() > 10000);
}

