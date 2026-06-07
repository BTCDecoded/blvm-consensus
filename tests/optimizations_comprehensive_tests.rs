//! Optimizations comprehensive tests
//!
//! Additional tests for optimization functions.

#[cfg(feature = "production")]
use blvm_consensus::optimizations::{
    simd_vectorization::{batch_double_sha256, batch_hash160, batch_ripemd160, batch_sha256},
    CacheAlignedHash,
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
    for result in &results {
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

#[cfg(feature = "production")]
#[test]
fn test_constant_folding_hash_helpers() {
    use blvm_consensus::optimizations::constant_folding::{
        is_empty_double_hash, is_empty_hash, is_zero_hash, EMPTY_STRING_DOUBLE_HASH,
        EMPTY_STRING_HASH,
    };

    assert!(is_empty_hash(&EMPTY_STRING_HASH));
    assert!(is_empty_double_hash(&EMPTY_STRING_DOUBLE_HASH));
    assert!(is_zero_hash(&[0u8; 32]));
    assert!(!is_empty_hash(&[1u8; 32]));
}

#[cfg(feature = "production")]
#[test]
fn test_proven_bounds_slice_access() {
    use blvm_consensus::optimizations::optimized_access::{get_proven, prealloc_proven};

    let data = vec![1u8, 2, 3];
    assert_eq!(get_proven(&data, 1), Some(&2u8));
    assert!(get_proven(&data, 9).is_none());
    let buf = prealloc_proven::<u8>(16);
    assert!(buf.capacity() >= 16);
}

#[cfg(feature = "production")]
#[test]
fn test_batch_double_sha256_aligned() {
    use blvm_consensus::optimizations::simd_vectorization::batch_double_sha256_aligned;

    let inputs = vec![b"a".as_slice(), b"b".as_slice()];
    let aligned = batch_double_sha256_aligned(&inputs);
    let plain = batch_double_sha256(&inputs);
    assert_eq!(aligned.len(), plain.len());
    for (a, p) in aligned.iter().zip(plain.iter()) {
        assert_eq!(a.as_bytes(), p);
    }
}

#[cfg(feature = "production")]
#[test]
fn test_prefetch_helpers_do_not_panic() {
    use blvm_consensus::optimizations::prefetch::{prefetch_ahead, prefetch_slice};

    let data = vec![1u8, 2, 3, 4, 5];
    prefetch_slice(&data, 0);
    prefetch_ahead(&data, 0, 2);
}

#[cfg(not(feature = "production"))]
#[test]
fn test_optimizations_require_production_feature() {
    assert!(
        !cfg!(feature = "production"),
        "this test only runs when production feature is disabled"
    );
}
