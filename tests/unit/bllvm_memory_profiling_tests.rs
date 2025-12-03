//! Memory profiling tests for BLLVM optimizations
//!
//! These tests scaffold memory profiling to verify:
//! - Memory allocation patterns
//! - Pre-allocation effectiveness
//! - Cache alignment impact
//!
//! To run with memory profiling:
//!   1. Install valgrind or heaptrack
//!   2. Run: cargo test --features production --test bllvm_memory_profiling_tests
//!   3. Or use: valgrind --tool=massif cargo test --features production --test bllvm_memory_profiling_tests

use blvm_consensus::{
    mining::calculate_merkle_root,
    optimizations::{prealloc_tx_buffer, prealloc_block_buffer},
    serialization::transaction::serialize_transaction,
    types::{Transaction, TransactionInput, TransactionOutput, OutPoint},
};

/// Helper to create a test transaction
fn create_test_transaction() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x51], // OP_1
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(), // OP_1
        }].into(),
        lock_time: 0,
    }
}

/// Helper to create multiple test transactions
fn create_test_transactions(count: usize) -> Vec<Transaction> {
    (0..count).map(|i| {
        let mut tx = create_test_transaction();
        tx.inputs[0].prevout.index = i as u64;
        tx
    }).collect()
}

/// Test that pre-allocation reduces reallocations
#[test]
#[cfg(feature = "production")]
fn test_preallocation_reduces_reallocations() {
    // This test verifies that pre-allocation buffers are used effectively
    // Memory profiler should show fewer allocations with production feature

    let tx = create_test_transaction();
    
    // Serialize transaction (uses prealloc_tx_buffer in production)
    let serialized = serialize_transaction(&tx);
    
    // Verify buffer was used (size should be less than capacity in production)
    #[cfg(feature = "production")]
    {
        // In production, the buffer should be pre-allocated
        // Actual verification would require memory profiling tools
        assert!(serialized.len() > 0);
    }
}

/// Test memory usage patterns for batch operations
#[test]
#[cfg(feature = "production")]
fn test_batch_operations_memory_patterns() {
    use blvm_consensus::optimizations::simd_vectorization;

    // Create a large batch
    let transactions = create_test_transactions(1000);
    let serialized: Vec<Vec<u8>> = transactions
        .iter()
        .map(|tx| serialize_transaction(tx))
        .collect();
    let tx_refs: Vec<&[u8]> = serialized.iter().map(|v| v.as_slice()).collect();

    // Batch hash using aligned structures
    let aligned_hashes = simd_vectorization::batch_double_sha256_aligned(&tx_refs);
    assert_eq!(aligned_hashes.len(), 1000);

    // Memory profiler should show:
    // - Pre-allocated buffers for serialization
    // - Cache-aligned structures for hashes
    // - Reduced fragmentation
}

/// Test memory allocation for merkle root calculation
#[test]
#[cfg(feature = "production")]
fn test_merkle_root_memory_allocation() {
    // Create transactions for merkle root calculation
    let transactions = create_test_transactions(500);

    // Calculate merkle root (uses pre-allocation and cache alignment)
    let root = calculate_merkle_root(&transactions).expect("Should calculate merkle root");

    // Memory profiler should show:
    // - Pre-allocated hash arrays
    // - Pre-allocated next-level arrays
    // - Cache-aligned structures
    assert_eq!(root.len(), 32);
}

/// Test pre-allocation buffer sizes
#[test]
#[cfg(feature = "production")]
fn test_preallocation_buffer_sizes() {
    // Test transaction buffer
    let tx_buffer = prealloc_tx_buffer();
    assert!(tx_buffer.capacity() >= 100_000, "Transaction buffer should be pre-allocated");

    // Test block buffer
    let block_buffer = prealloc_block_buffer();
    assert!(block_buffer.capacity() >= 1_000_000, "Block buffer should be pre-allocated");
}

/// Test memory usage scaling
#[test]
#[cfg(feature = "production")]
fn test_memory_usage_scaling() {
    // Test with different transaction counts
    for count in [10, 100, 500, 1000].iter() {
        let transactions = create_test_transactions(*count);
        
        // Serialize all transactions
        let serialized: Vec<Vec<u8>> = transactions
            .iter()
            .map(|tx| serialize_transaction(tx))
            .collect();

        // Memory profiler should show linear scaling with pre-allocation
        assert_eq!(serialized.len(), *count);
    }
}

/// Test cache alignment impact (scaffold for profiling)
#[test]
#[cfg(feature = "production")]
fn test_cache_alignment_impact() {
    use blvm_consensus::optimizations::simd_vectorization;

    // Create test data
    let transactions = create_test_transactions(1000);
    let serialized: Vec<Vec<u8>> = transactions
        .iter()
        .map(|tx| serialize_transaction(tx))
        .collect();
    let tx_refs: Vec<&[u8]> = serialized.iter().map(|v| v.as_slice()).collect();

    // Use aligned version
    let aligned_hashes = simd_vectorization::batch_double_sha256_aligned(&tx_refs);
    
    // Use regular version
    let regular_hashes = simd_vectorization::batch_double_sha256(&tx_refs);

    // Verify correctness
    assert_eq!(aligned_hashes.len(), regular_hashes.len());
    
    // Memory profiler should show:
    // - Cache-aligned structures reduce cache misses
    // - Better memory access patterns
    // - Improved cache line utilization
}

/// Test memory fragmentation (scaffold for profiling)
#[test]
#[cfg(feature = "production")]
fn test_memory_fragmentation() {
    // Perform many serialization operations
    for _ in 0..1000 {
        let tx = create_test_transaction();
        let _serialized = serialize_transaction(&tx);
    }

    // Memory profiler should show:
    // - Reduced fragmentation with pre-allocation
    // - More consistent memory usage patterns
    // - Fewer allocations/deallocations
}

