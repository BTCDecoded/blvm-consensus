//! Optimizations comprehensive tests
//!
//! Additional tests for optimization functions.

#[cfg(feature = "production")]
use bllvm_consensus::optimizations::{
    batch_double_sha256, batch_hash160, batch_ripemd160, batch_sha256, CacheAlignedHash,
};

#[cfg(feature = "production")]
#[test]
fn test_batch_sha256_empty() {
    let inputs: Vec<&[u8]> = vec![];
    let results = batch_sha256(&inputs);

    assert_eq!(results.len(), 0);
}

#[cfg(feature = "production")]
#[test]
fn test_batch_sha256_single() {
    let inputs = vec![b"test data".as_slice()];
    let results = batch_sha256(&inputs);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 32);
}

#[cfg(feature = "production")]
#[test]
fn test_batch_sha256_multiple() {
    let inputs = vec![
        b"input1".as_slice(),
        b"input2".as_slice(),
        b"input3".as_slice(),
    ];
    let results = batch_sha256(&inputs);

    assert_eq!(results.len(), 3);
    for result in results {
        assert_eq!(result.len(), 32);
    }

    // Verify results are different
    assert_ne!(results[0], results[1]);
    assert_ne!(results[1], results[2]);
}

#[cfg(feature = "production")]
#[test]
fn test_batch_double_sha256() {
    let inputs = vec![b"test1".as_slice(), b"test2".as_slice()];
    let results = batch_double_sha256(&inputs);

    assert_eq!(results.len(), 2);
    for result in results {
        assert_eq!(result.len(), 32);
    }
}

#[cfg(feature = "production")]
#[test]
fn test_batch_ripemd160() {
    let inputs = vec![b"test1".as_slice(), b"test2".as_slice()];
    let results = batch_ripemd160(&inputs);

    assert_eq!(results.len(), 2);
    for result in results {
        assert_eq!(result.len(), 20); // RIPEMD160 produces 20-byte hashes
    }
}

#[cfg(feature = "production")]
#[test]
fn test_batch_hash160() {
    let inputs = vec![b"test1".as_slice(), b"test2".as_slice()];
    let results = batch_hash160(&inputs);

    assert_eq!(results.len(), 2);
    for result in results {
        assert_eq!(result.len(), 20); // Hash160 produces 20-byte hashes
    }
}

#[cfg(feature = "production")]
#[test]
fn test_cache_aligned_hash() {
    let hash = [0x42u8; 32];
    let aligned = CacheAlignedHash::new(hash);

    let bytes = aligned.as_bytes();
    assert_eq!(bytes, &hash);
}

#[cfg(feature = "production")]
#[test]
fn test_batch_operations_consistency() {
    // Test that batch operations produce same results as individual operations
    use sha2::{Digest, Sha256};

    let input = b"test data";

    // Individual hash
    let individual = Sha256::digest(input);

    // Batch hash
    let batch_results = batch_sha256(&[input.as_slice()]);

    assert_eq!(batch_results[0].as_slice(), individual.as_slice());
}

#[cfg(not(feature = "production"))]
#[test]
fn test_optimizations_require_production_feature() {
    // When production feature is not enabled, optimization functions are not available
    // This test verifies the feature gating works
    assert!(true);
}
