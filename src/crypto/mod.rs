//! Cryptographic hash functions with CPU feature detection and optimizations
//!
//! Provides optimized SHA256 implementations that automatically select
//! the best available implementation based on CPU features:
//! - AVX2 SIMD (8-way parallel) - fastest on modern CPUs
//! - SSE4 SIMD (4-way parallel) - fallback for older CPUs  
//! - Generic (sha2 crate with asm) - baseline for all CPUs
//!
//! The sha2 crate with "asm" feature already includes optimized assembly,
//! but we can add batch processing and CPU feature detection for better
//! performance in batch scenarios.

use sha2::{Digest, Sha256};

#[cfg(target_arch = "x86_64")]
pub mod avx2_batch;
#[cfg(target_arch = "x86_64")]
pub mod hash_compare;
#[cfg(target_arch = "x86_64")]
pub mod sha256_avx2;
#[cfg(target_arch = "x86_64")]
pub mod sha_ni;
#[cfg(target_arch = "x86_64")]
pub mod simd_bytes;

pub mod int_ops;

/// CPU feature detection for runtime optimization selection
pub mod cpu_features {
    /// Check if AVX2 is available
    ///
    /// Hot-path function called during hash operation dispatch.
    /// Always inline for maximum performance.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn has_avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
    }

    /// Check if SSE4.1 is available
    ///
    /// Hot-path function called during hash operation dispatch.
    /// Always inline for maximum performance.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn has_sse41() -> bool {
        std::arch::is_x86_feature_detected!("sse4.1")
    }

    /// Check if Intel SHA-NI is available
    ///
    /// Hot-path function called during hash operation dispatch.
    /// Always inline for maximum performance.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn has_sha_ni() -> bool {
        std::arch::is_x86_feature_detected!("sha")
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn has_avx2() -> bool {
        false
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn has_sse41() -> bool {
        false
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn has_sha_ni() -> bool {
        false
    }
}

/// Optimized SHA256 hasher with CPU feature detection
pub struct OptimizedSha256;

impl OptimizedSha256 {
    /// Create a new SHA256 hasher with automatic CPU feature detection
    pub fn new() -> Self {
        Self
    }

    /// Hash data using the best available implementation
    ///
    /// Priority:
    /// 1. SHA-NI (Intel SHA Extensions) - 10-15x faster for single hashes
    /// 2. sha2 crate with asm - baseline fallback
    ///
    /// For batch operations, see batch_sha256 in optimizations module.
    pub fn hash(&self, data: &[u8]) -> [u8; 32] {
        #[cfg(target_arch = "x86_64")]
        {
            // Try SHA-NI first (hardware accelerated, optimal for single hashes)
            if sha_ni::is_sha_ni_available() {
                return sha_ni::sha256(data);
            }
        }

        // Fallback: sha2 crate with asm optimizations
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(&hash);
        result
    }

    /// Compute double SHA256 (SHA256(SHA256(data)))
    ///
    /// Uses SHA-NI if available for optimal single-hash performance.
    pub fn hash256(&self, data: &[u8]) -> [u8; 32] {
        #[cfg(target_arch = "x86_64")]
        {
            if sha_ni::is_sha_ni_available() {
                return sha_ni::hash256(data);
            }
        }

        let first = self.hash(data);
        self.hash(&first)
    }
}

impl Default for OptimizedSha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for single SHA256 hash
pub fn sha256(data: &[u8]) -> [u8; 32] {
    OptimizedSha256::new().hash(data)
}

/// Convenience function for double SHA256 hash (Bitcoin standard)
pub fn hash256(data: &[u8]) -> [u8; 32] {
    OptimizedSha256::new().hash256(data)
}
