//! SIMD-optimized byte array operations
//!
//! Provides fast byte array copying and concatenation using AVX2 SIMD instructions
//! when available, with automatic fallback to sequential operations for compatibility.

/// Copy bytes from source to destination using SIMD when beneficial
///
/// Uses AVX2 SIMD for large arrays (>64 bytes), falls back to sequential
/// copy for smaller arrays or non-AVX2 systems.
///
/// # Arguments
/// * `dst` - Destination slice (must be at least as long as `src`)
/// * `src` - Source slice
///
/// # Safety
/// Caller must ensure `dst.len() >= src.len()` to avoid buffer overflows.
#[inline]
pub fn copy_bytes_simd(dst: &mut [u8], src: &[u8]) {
    #[cfg(all(target_arch = "x86_64", feature = "production"))]
    {
        // Use SIMD for large arrays (threshold: 64 bytes)
        if src.len() >= 64 && is_avx2_available() {
            unsafe {
                copy_bytes_avx2(dst, src);
            }
            return;
        }
    }

    // Fallback: sequential copy
    dst[..src.len()].copy_from_slice(src);
}

/// AVX2-optimized byte copy
///
/// Copies data in 32-byte chunks using AVX2, then handles remainder sequentially.
#[cfg(all(target_arch = "x86_64", feature = "production"))]
#[target_feature(enable = "avx2")]
unsafe fn copy_bytes_avx2(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::*;

    let chunks = src.len() / 32;
    let mut dst_ptr = dst.as_mut_ptr();
    let mut src_ptr = src.as_ptr();

    // Process 32-byte chunks with AVX2
    for _ in 0..chunks {
        let data = _mm256_loadu_si256(src_ptr as *const __m256i);
        _mm256_storeu_si256(dst_ptr as *mut __m256i, data);
        dst_ptr = dst_ptr.add(32);
        src_ptr = src_ptr.add(32);
    }

    // Handle remainder sequentially
    let remainder = src.len() % 32;
    if remainder > 0 {
        let offset = chunks * 32;
        dst[offset..offset + remainder].copy_from_slice(&src[offset..offset + remainder]);
    }
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
    fn test_copy_bytes_small() {
        let src = [1, 2, 3, 4, 5];
        let mut dst = [0u8; 10];
            copy_bytes_simd(&mut dst, &src);
        assert_eq!(&dst[..5], &src);
    }

    #[test]
    fn test_copy_bytes_large() {
        let src: Vec<u8> = (0..128).collect();
        let mut dst = vec![0u8; 128];
            copy_bytes_simd(&mut dst, &src);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_copy_bytes_exact_32() {
        let src: Vec<u8> = (0..32).collect();
        let mut dst = vec![0u8; 32];
            copy_bytes_simd(&mut dst, &src);
        assert_eq!(dst, src);
    }
}
