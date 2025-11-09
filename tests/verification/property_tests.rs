//! Property-based tests for SHA256 optimizations
//!
//! Uses PropTest to generate thousands of random test cases.

use proptest::prelude::*;
use bllvm_consensus::crypto::OptimizedSha256;
use sha2::{Digest, Sha256};

proptest! {
    #[test]
    fn sha256_matches_reference(
        data in prop::collection::vec(any::<u8>(), 0..1024)
    ) {
        let reference = Sha256::digest(&data);
        let ours = OptimizedSha256::new().hash(&data);
        prop_assert_eq!(&reference[..], &ours[..]);
    }

    #[test]
    fn double_sha256_matches_reference(
        data in prop::collection::vec(any::<u8>(), 0..1024)
    ) {
        let reference = Sha256::digest(&Sha256::digest(&data));
        let ours = OptimizedSha256::new().hash256(&data);
        prop_assert_eq!(&reference[..], &ours[..]);
    }

    #[test]
    fn sha256_idempotent(data in prop::collection::vec(any::<u8>(), 0..1024)) {
        let hash1 = OptimizedSha256::new().hash(&data);
        let hash2 = OptimizedSha256::new().hash(&data);
        prop_assert_eq!(&hash1[..], &hash2[..]);
    }

    #[test]
    fn sha256_deterministic(
        data in prop::collection::vec(any::<u8>(), 0..1024)
    ) {
        // Same input should always produce same output
        let hash1 = OptimizedSha256::new().hash(&data);
        let hash2 = OptimizedSha256::new().hash(&data);
        prop_assert_eq!(&hash1[..], &hash2[..]);
    }

    #[test]
    fn sha256_output_length(data in prop::collection::vec(any::<u8>(), 0..1024)) {
        let hash = OptimizedSha256::new().hash(&data);
        prop_assert_eq!(hash.len(), 32);
    }

    #[test]
    fn double_sha256_output_length(data in prop::collection::vec(any::<u8>(), 0..1024)) {
        let hash = OptimizedSha256::new().hash256(&data);
        prop_assert_eq!(hash.len(), 32);
    }
}

