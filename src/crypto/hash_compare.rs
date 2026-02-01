//! SIMD-optimized hash comparison utilities
//!
//! Provides fast hash comparison using AVX2 SIMD instructions when available,
//! with automatic fallback to sequential comparison for compatibility.

use crate::types::Hash;
use blvm_spec_lock::spec_locked;

/// Compare two 32-byte hashes for equality
///
/// Uses SIMD (AVX2) when available for faster comparison, otherwise falls back
/// to standard byte-by-byte comparison.
///
/// # Arguments
/// * `hash1` - First hash to compare
/// * `hash2` - Second hash to compare
///
/// # Returns
/// `true` if hashes are equal, `false` otherwise
#[spec_locked("2.1")]
#[inline]
pub fn hash_eq(hash1: &Hash, hash2: &Hash) -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "production"))]
    {
        // Try AVX2 SIMD comparison first
        if is_avx2_available() {
            unsafe { hash_eq_avx2(hash1, hash2) }
        } else {
            hash_eq_fallback(hash1, hash2)
        }
    }

    #[cfg(not(all(target_arch = "x86_64", feature = "production")))]
    {
        hash_eq_fallback(hash1, hash2)
    }
}

/// Fallback hash comparison (sequential byte-by-byte)
#[inline(always)]
fn hash_eq_fallback(hash1: &Hash, hash2: &Hash) -> bool {
    hash1 == hash2
}

/// AVX2-optimized hash comparison
///
/// Compares 32 bytes in parallel using a single AVX2 operation.
#[cfg(all(target_arch = "x86_64", feature = "production"))]
#[target_feature(enable = "avx2")]
unsafe fn hash_eq_avx2(hash1: &Hash, hash2: &Hash) -> bool {
    use std::arch::x86_64::*;

    // Load both hashes as 256-bit AVX2 vectors
    let h1 = _mm256_loadu_si256(hash1.as_ptr() as *const __m256i);
    let h2 = _mm256_loadu_si256(hash2.as_ptr() as *const __m256i);

    // Compare all 32 bytes in parallel (byte-wise equality)
    let cmp = _mm256_cmpeq_epi8(h1, h2);

    // Extract comparison mask (1 bit per byte)
    let mask = _mm256_movemask_epi8(cmp);

    // All 32 bytes equal if mask == -1 (all 32 bits set)
    // Note: _mm256_movemask_epi8 returns i32, so 0xFFFFFFFF is -1
    mask == -1i32
}

/// Check if AVX2 is available at runtime
#[cfg(all(target_arch = "x86_64", feature = "production"))]
fn is_avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_eq_identical() {
        let hash1 = [0u8; 32];
        let hash2 = [0u8; 32];
        assert!(hash_eq(&hash1, &hash2));
    }

    #[test]
    fn test_hash_eq_different() {
        let hash1 = [0u8; 32];
        let mut hash2 = [0u8; 32];
        hash2[0] = 1;
        assert!(!hash_eq(&hash1, &hash2));
    }

    #[test]
    fn test_hash_eq_different_last_byte() {
        let mut hash1 = [0u8; 32];
        let mut hash2 = [0u8; 32];
        hash1[31] = 0;
        hash2[31] = 1;
        assert!(!hash_eq(&hash1, &hash2));
    }

    #[test]
    fn test_hash_eq_random() {
        let hash1: Hash = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let hash2: Hash = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 33,
        ];

        // Should match standard comparison
        assert_eq!(hash_eq(&hash1, &hash1), hash1 == hash1);
        assert_eq!(hash_eq(&hash1, &hash2), hash1 == hash2);
    }
}
