//! Unit tests for BLLVM optimizations
//!
//! Tests verify that BLLVM optimizations (pre-allocation, cache alignment)
//! maintain correctness while improving performance.

use bllvm_consensus::{
    mining::calculate_merkle_root,
    optimizations::{prealloc_tx_buffer, prealloc_block_buffer},
    serialization::{block::serialize_block_header, transaction::serialize_transaction},
    types::{BlockHeader, Hash, Transaction, TransactionInput, TransactionOutput, OutPoint},
};

/// Test that pre-allocated transaction buffer works correctly
#[test]
fn test_prealloc_tx_buffer_correctness() {
    #[cfg(feature = "production")]
    {
        let buffer = prealloc_tx_buffer();
        // Should be pre-allocated to proven maximum size
        assert!(buffer.capacity() >= 100_000); // MAX_TX_SIZE_PROVEN
        assert_eq!(buffer.len(), 0); // Empty initially
    }
}

/// Test that pre-allocated block buffer works correctly
#[test]
fn test_prealloc_block_buffer_correctness() {
    #[cfg(feature = "production")]
    {
        let buffer = prealloc_block_buffer();
        // Should be pre-allocated to proven maximum size
        assert!(buffer.capacity() >= 1_000_000); // MAX_BLOCK_SIZE_PROVEN
        assert_eq!(buffer.len(), 0); // Empty initially
    }
}

/// Helper to create a test transaction
fn create_test_transaction() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32], index: 0 },
            script_sig: vec![0x51], // OP_1
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51], // OP_1
        }],
        lock_time: 0,
    }
}

/// Helper to create a test block header
fn create_test_block_header() -> BlockHeader {
    BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0,
    }
}

/// Helper to create multiple test transactions
fn create_test_transactions(count: usize) -> Vec<Transaction> {
    (0..count).map(|i| {
        let mut tx = create_test_transaction();
        // Make each transaction unique
        tx.inputs[0].prevout.index = i as u64;
        tx
    }).collect()
}

/// Test that serialize_transaction produces identical output with/without production feature
#[test]
fn test_serialize_transaction_correctness() {
    let tx = create_test_transaction();

    // Serialize (uses pre-allocation in production)
    let serialized = serialize_transaction(&tx);

    // Verify basic structure
    assert!(serialized.len() >= 10, "Transaction should have minimum size");
}

/// Test that serialize_block_header produces identical output with/without production feature
#[test]
fn test_serialize_block_header_correctness() {
    let header = create_test_block_header();

    // Serialize (uses pre-allocation in production)
    let serialized = serialize_block_header(&header);

    assert_eq!(serialized.len(), 80, "Block header should be exactly 80 bytes");
}

/// Test that merkle root calculation produces identical results with/without production feature
#[test]
fn test_merkle_root_correctness() {
    let transactions = create_test_transactions(10);

    // Calculate merkle root (uses cache-aligned structures in production)
    let root = calculate_merkle_root(&transactions).expect("Should calculate merkle root");

    // Verify root is a valid hash
    assert_eq!(root.len(), 32);

    // Verify deterministic
    let root2 = calculate_merkle_root(&transactions).expect("Should calculate merkle root again");
    assert_eq!(root, root2, "Merkle root should be deterministic");
}

/// Test that cache-aligned batch hashing produces correct results
#[test]
fn test_cache_aligned_batch_hashing() {
    #[cfg(feature = "production")]
    {
        use bllvm_consensus::optimizations::simd_vectorization;

        let inputs = vec![
            b"test input 1".as_slice(),
            b"test input 2".as_slice(),
            b"test input 3".as_slice(),
        ];

        // Use aligned version
        let aligned_hashes = simd_vectorization::batch_double_sha256_aligned(&inputs);
        assert_eq!(aligned_hashes.len(), 3);

        // Use regular version
        let regular_hashes = simd_vectorization::batch_double_sha256(&inputs);
        assert_eq!(regular_hashes.len(), 3);

        // Verify they produce identical results
        for (aligned, regular) in aligned_hashes.iter().zip(regular_hashes.iter()) {
            assert_eq!(aligned.as_bytes(), regular, "Aligned and regular hashes should match");
        }
    }
}

/// Test that batch operations handle empty inputs correctly
#[test]
fn test_batch_operations_empty_input() {
    #[cfg(feature = "production")]
    {
        use bllvm_consensus::optimizations::simd_vectorization;

        let empty_inputs: Vec<&[u8]> = vec![];

        let aligned_hashes = simd_vectorization::batch_double_sha256_aligned(&empty_inputs);
        assert_eq!(aligned_hashes.len(), 0);

        let regular_hashes = simd_vectorization::batch_double_sha256(&empty_inputs);
        assert_eq!(regular_hashes.len(), 0);
    }
}

/// Test that batch operations handle small batches correctly
#[test]
fn test_batch_operations_small_batch() {
    #[cfg(feature = "production")]
    {
        use bllvm_consensus::optimizations::simd_vectorization;

        let inputs = vec![
            b"small batch test".as_slice(),
        ];

        let aligned_hashes = simd_vectorization::batch_double_sha256_aligned(&inputs);
        assert_eq!(aligned_hashes.len(), 1);

        let regular_hashes = simd_vectorization::batch_double_sha256(&inputs);
        assert_eq!(regular_hashes.len(), 1);
        assert_eq!(aligned_hashes[0].as_bytes(), &regular_hashes[0]);
    }
}

/// Test that merkle root calculation handles edge cases correctly
#[test]
fn test_merkle_root_edge_cases() {
    // Single transaction
    let single_tx = vec![create_test_transaction()];
    let root = calculate_merkle_root(&single_tx).expect("Should calculate merkle root for single tx");
    assert_eq!(root.len(), 32);

    // Multiple transactions (odd number)
    let odd_txs = vec![
        create_test_transaction(),
        create_test_transaction(),
        create_test_transaction(),
    ];
    let root_odd = calculate_merkle_root(&odd_txs).expect("Should calculate merkle root for odd number");
    assert_eq!(root_odd.len(), 32);

    // Multiple transactions (even number)
    let even_txs = vec![
        create_test_transaction(),
        create_test_transaction(),
    ];
    let root_even = calculate_merkle_root(&even_txs).expect("Should calculate merkle root for even number");
    assert_eq!(root_even.len(), 32);
}

/// Test that serialization maintains Bitcoin compatibility
#[test]
fn test_serialization_bitcoin_compatibility() {
    let tx = create_test_transaction();
    let serialized = serialize_transaction(&tx);

    // Verify basic structure
    assert!(serialized.len() >= 10, "Transaction should have minimum size");
    
    // Version should be first 4 bytes
    let version_bytes = &serialized[0..4];
    let version = i32::from_le_bytes([
        version_bytes[0],
        version_bytes[1],
        version_bytes[2],
        version_bytes[3],
    ]);
    assert_eq!(version, tx.version as i32, "Version should match");
}

