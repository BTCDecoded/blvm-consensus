//! Kani formal verification proofs for SHA256 optimizations
//!
//! These proofs verify that optimizations don't change correctness.

#[cfg(kani)]
mod kani_sha256 {
    use kani::*;
    use bllvm_consensus::crypto::OptimizedSha256;
    use sha2::{Digest, Sha256};

    /// Verify that OptimizedSha256 produces the same output as sha2 crate
    #[kani::proof]
    fn verify_sha256_correctness() {
        let input: [u8; 64] = kani::any();
        let reference = Sha256::digest(&input);
        let optimized = OptimizedSha256::new().hash(&input);
        assert_eq!(&reference[..], &optimized[..]);
    }

    /// Verify double SHA256 correctness
    #[kani::proof]
    fn verify_double_sha256_correctness() {
        let input: [u8; 32] = kani::any();
        let reference = Sha256::digest(&Sha256::digest(&input));
        let optimized = OptimizedSha256::new().hash256(&input);
        assert_eq!(&reference[..], &optimized[..]);
    }

    /// Verify no undefined behavior for variable-length inputs
    #[kani::proof]
    fn verify_no_undefined_behavior() {
        let len: usize = kani::any();
        kani::assume(len <= 1024); // Bound for tractability
        let input: Vec<u8> = kani::vec::any_vec::<u8>(len);
        let _ = OptimizedSha256::new().hash(&input);
    }

    /// Verify idempotency: same input produces same output
    #[kani::proof]
    fn verify_sha256_idempotent() {
        let input: [u8; 64] = kani::any();
        let hash1 = OptimizedSha256::new().hash(&input);
        let hash2 = OptimizedSha256::new().hash(&input);
        assert_eq!(&hash1[..], &hash2[..]);
    }
}

