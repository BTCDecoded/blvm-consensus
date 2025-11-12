//! Intel SHA Extensions (SHA-NI) optimized SHA256
//!
//! This module provides hardware-accelerated SHA256 using Intel SHA Extensions.
//! Optimized for single-hash and small-batch operations.
//!
//! # Performance
//! - Expected: 10-15x faster than sha2 crate for single hashes
//! - Uses dedicated SHA256 hardware instructions
//! - Low latency, optimized for single-hash speed
//!
//! # CPU Support
//! - Intel: Ice Lake (2019+), all newer generations
//! - AMD: All Ryzen (Zen microarchitecture, 2017+)
//!
//! # Reference
//! Based on Intel's SHA-NI reference implementation and Bitcoin Core's sha256_shani.cpp

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Check if SHA-NI is available at runtime
#[cfg(target_arch = "x86_64")]
pub fn is_sha_ni_available() -> bool {
    std::arch::is_x86_feature_detected!("sha")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn is_sha_ni_available() -> bool {
    false
}

/// Single SHA256 using Intel SHA-NI instructions
///
/// This uses hardware SHA256 acceleration for optimal single-hash performance.
/// For batch operations, use the AVX2 implementation instead.
///
/// # Safety
/// Requires SHA-NI, SSE2, SSSE3, and SSE4.1 support. This is checked at runtime
/// by the public API functions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn sha256_ni_impl(data: &[u8]) -> [u8; 32] {
    // SHA256 initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
    let mut state0 = _mm_setr_epi32(
        0x6a09e667u32 as i32,
        0xbb67ae85u32 as i32,
        0x3c6ef372u32 as i32,
        0xa54ff53au32 as i32,
    );
    let mut state1 = _mm_setr_epi32(
        0x510e527fu32 as i32,
        0x9b05688cu32 as i32,
        0x1f83d9abu32 as i32,
        0x5be0cd19u32 as i32,
    );

    // Byte swap mask for converting between little-endian and big-endian
    let shuf_mask = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);

    // SHA256 constants (first 32 bits of fractional parts of cube roots of first 64 primes)
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Process input data with padding
    let mut padded = Vec::with_capacity((data.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(data);

    // Add padding: 0x80 byte, then zeros, then 64-bit length
    padded.push(0x80);
    let bit_len = (data.len() as u64) * 8;
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 64-byte (512-bit) block
    for chunk in padded.chunks_exact(64) {
        let mut msg0 = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
        let mut msg1 = _mm_loadu_si128(chunk.as_ptr().add(16) as *const __m128i);
        let mut msg2 = _mm_loadu_si128(chunk.as_ptr().add(32) as *const __m128i);
        let mut msg3 = _mm_loadu_si128(chunk.as_ptr().add(48) as *const __m128i);

        // Byte swap to big-endian
        msg0 = _mm_shuffle_epi8(msg0, shuf_mask);
        msg1 = _mm_shuffle_epi8(msg1, shuf_mask);
        msg2 = _mm_shuffle_epi8(msg2, shuf_mask);
        msg3 = _mm_shuffle_epi8(msg3, shuf_mask);

        let mut tmp;
        let mut msg;

        // Save current state
        let state0_save = state0;
        let state1_save = state1;

        // Rounds 0-3
        msg = _mm_add_epi32(
            msg0,
            _mm_set_epi32(K[3] as i32, K[2] as i32, K[1] as i32, K[0] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

        // Rounds 4-7
        msg = _mm_add_epi32(
            msg1,
            _mm_set_epi32(K[7] as i32, K[6] as i32, K[5] as i32, K[4] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg0 = _mm_sha256msg1_epu32(msg0, msg1);

        // Rounds 8-11
        msg = _mm_add_epi32(
            msg2,
            _mm_set_epi32(K[11] as i32, K[10] as i32, K[9] as i32, K[8] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg1 = _mm_sha256msg1_epu32(msg1, msg2);

        // Rounds 12-15
        msg = _mm_add_epi32(
            msg3,
            _mm_set_epi32(K[15] as i32, K[14] as i32, K[13] as i32, K[12] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg3, msg2, 4);
        msg0 = _mm_add_epi32(msg0, tmp);
        msg0 = _mm_sha256msg2_epu32(msg0, msg3);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg2 = _mm_sha256msg1_epu32(msg2, msg3);

        // Rounds 16-19
        msg = _mm_add_epi32(
            msg0,
            _mm_set_epi32(K[19] as i32, K[18] as i32, K[17] as i32, K[16] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg0, msg3, 4);
        msg1 = _mm_add_epi32(msg1, tmp);
        msg1 = _mm_sha256msg2_epu32(msg1, msg0);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg3 = _mm_sha256msg1_epu32(msg3, msg0);

        // Rounds 20-23
        msg = _mm_add_epi32(
            msg1,
            _mm_set_epi32(K[23] as i32, K[22] as i32, K[21] as i32, K[20] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg1, msg0, 4);
        msg2 = _mm_add_epi32(msg2, tmp);
        msg2 = _mm_sha256msg2_epu32(msg2, msg1);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg0 = _mm_sha256msg1_epu32(msg0, msg1);

        // Rounds 24-27
        msg = _mm_add_epi32(
            msg2,
            _mm_set_epi32(K[27] as i32, K[26] as i32, K[25] as i32, K[24] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg2, msg1, 4);
        msg3 = _mm_add_epi32(msg3, tmp);
        msg3 = _mm_sha256msg2_epu32(msg3, msg2);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg1 = _mm_sha256msg1_epu32(msg1, msg2);

        // Rounds 28-31
        msg = _mm_add_epi32(
            msg3,
            _mm_set_epi32(K[31] as i32, K[30] as i32, K[29] as i32, K[28] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg3, msg2, 4);
        msg0 = _mm_add_epi32(msg0, tmp);
        msg0 = _mm_sha256msg2_epu32(msg0, msg3);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg2 = _mm_sha256msg1_epu32(msg2, msg3);

        // Rounds 32-35
        msg = _mm_add_epi32(
            msg0,
            _mm_set_epi32(K[35] as i32, K[34] as i32, K[33] as i32, K[32] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg0, msg3, 4);
        msg1 = _mm_add_epi32(msg1, tmp);
        msg1 = _mm_sha256msg2_epu32(msg1, msg0);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg3 = _mm_sha256msg1_epu32(msg3, msg0);

        // Rounds 36-39
        msg = _mm_add_epi32(
            msg1,
            _mm_set_epi32(K[39] as i32, K[38] as i32, K[37] as i32, K[36] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg1, msg0, 4);
        msg2 = _mm_add_epi32(msg2, tmp);
        msg2 = _mm_sha256msg2_epu32(msg2, msg1);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg0 = _mm_sha256msg1_epu32(msg0, msg1);

        // Rounds 40-43
        msg = _mm_add_epi32(
            msg2,
            _mm_set_epi32(K[43] as i32, K[42] as i32, K[41] as i32, K[40] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg2, msg1, 4);
        msg3 = _mm_add_epi32(msg3, tmp);
        msg3 = _mm_sha256msg2_epu32(msg3, msg2);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg1 = _mm_sha256msg1_epu32(msg1, msg2);

        // Rounds 44-47
        msg = _mm_add_epi32(
            msg3,
            _mm_set_epi32(K[47] as i32, K[46] as i32, K[45] as i32, K[44] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg3, msg2, 4);
        msg0 = _mm_add_epi32(msg0, tmp);
        msg0 = _mm_sha256msg2_epu32(msg0, msg3);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg2 = _mm_sha256msg1_epu32(msg2, msg3);

        // Rounds 48-51
        msg = _mm_add_epi32(
            msg0,
            _mm_set_epi32(K[51] as i32, K[50] as i32, K[49] as i32, K[48] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg0, msg3, 4);
        msg1 = _mm_add_epi32(msg1, tmp);
        msg1 = _mm_sha256msg2_epu32(msg1, msg0);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        msg3 = _mm_sha256msg1_epu32(msg3, msg0);

        // Rounds 52-55
        msg = _mm_add_epi32(
            msg1,
            _mm_set_epi32(K[55] as i32, K[54] as i32, K[53] as i32, K[52] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg1, msg0, 4);
        msg2 = _mm_add_epi32(msg2, tmp);
        msg2 = _mm_sha256msg2_epu32(msg2, msg1);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

        // Rounds 56-59
        msg = _mm_add_epi32(
            msg2,
            _mm_set_epi32(K[59] as i32, K[58] as i32, K[57] as i32, K[56] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        tmp = _mm_alignr_epi8(msg2, msg1, 4);
        msg3 = _mm_add_epi32(msg3, tmp);
        msg3 = _mm_sha256msg2_epu32(msg3, msg2);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

        // Rounds 60-63
        msg = _mm_add_epi32(
            msg3,
            _mm_set_epi32(K[63] as i32, K[62] as i32, K[61] as i32, K[60] as i32),
        );
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        msg = _mm_shuffle_epi32(msg, 0x0E);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

        // Add back to state
        state0 = _mm_add_epi32(state0, state0_save);
        state1 = _mm_add_epi32(state1, state1_save);
    }

    // Combine state0 and state1 into final hash
    // After _mm_sha256rnds2_epu32, the state layout is interleaved:
    // state0: [h0, h1, h4, h5]
    // state1: [h2, h3, h6, h7]
    // We need to reorder to h0,h1,h2,h3,h4,h5,h6,h7 and byte-swap each word to big-endian

    let mut result = [0u8; 32];

    // Extract and byte-swap state0: h0, h1, h4, h5
    let state0_swapped = _mm_shuffle_epi8(state0, shuf_mask);
    _mm_storeu_si128(result.as_mut_ptr() as *mut __m128i, state0_swapped);

    // Extract and byte-swap state1: h2, h3, h6, h7
    let state1_swapped = _mm_shuffle_epi8(state1, shuf_mask);
    _mm_storeu_si128(result.as_mut_ptr().add(16) as *mut __m128i, state1_swapped);

    // Now we need to reorder: currently [h0, h1, h4, h5, h2, h3, h6, h7]
    // Need: [h0, h1, h2, h3, h4, h5, h6, h7]
    // Swap the middle two 16-byte blocks
    let temp = [
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7],
        result[8], result[9], result[10], result[11], result[12], result[13], result[14],
        result[15], result[16], result[17], result[18], result[19], result[20], result[21],
        result[22], result[23], result[24], result[25], result[26], result[27], result[28],
        result[29], result[30], result[31],
    ];

    // Reorder: h0, h1 from temp[0-7], h2, h3 from temp[16-23], h4, h5 from temp[8-15], h6, h7 from temp[24-31]
    result[0..8].copy_from_slice(&temp[0..8]); // h0, h1
    result[8..16].copy_from_slice(&temp[16..24]); // h2, h3
    result[16..24].copy_from_slice(&temp[8..16]); // h4, h5
    result[24..32].copy_from_slice(&temp[24..32]); // h6, h7

    result
}

/// Fallback SHA256 using sha2 crate
fn fallback_sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(data);
    let mut result = [0u8; 32];
    result.copy_from_slice(&hash);
    result
}

/// Public API: Single SHA256 with automatic dispatch
///
/// Uses Intel SHA-NI if available, otherwise falls back to sha2 crate.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_sha_ni_available() {
            unsafe { sha256_ni_impl(data) }
        } else {
            fallback_sha256(data)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        fallback_sha256(data)
    }
}

/// Double SHA256 (SHA256D) using SHA-NI
///
/// Computes SHA256(SHA256(data)), which is Bitcoin's standard hash function.
pub fn hash256(data: &[u8]) -> [u8; 32] {
    let first = sha256(data);
    sha256(&first)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha_ni_availability() {
        // Just check that detection doesn't panic
        let available = is_sha_ni_available();
        println!("SHA-NI available: {available}");
    }

    #[test]
    fn test_sha256_empty() {
        let input = b"";
        let result = sha256(input);

        // Expected SHA256(""): e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let expected = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sha256_hello_world() {
        let input = b"hello world";
        let result = sha256(input);

        // Compare with sha2 crate
        use sha2::{Digest, Sha256};
        let expected = Sha256::digest(input);

        assert_eq!(&result[..], &expected[..]);
    }

    #[test]
    fn test_sha256_matches_reference() {
        // Test various input sizes
        let zeros_64 = vec![0u8; 64];
        let ff_128 = vec![0xffu8; 128];

        let test_cases: Vec<&[u8]> = vec![
            b"" as &[u8],
            b"a",
            b"abc",
            b"message digest",
            b"abcdefghijklmnopqrstuvwxyz",
            b"The quick brown fox jumps over the lazy dog",
            &zeros_64,
            &ff_128,
        ];

        for input in test_cases {
            let result = sha256(input);

            // Must match sha2 crate
            use sha2::{Digest, Sha256};
            let expected = Sha256::digest(input);

            assert_eq!(
                &result[..],
                &expected[..],
                "Mismatch for input length {}",
                input.len()
            );
        }
    }

    #[test]
    fn test_double_sha256() {
        let input = b"hello world";
        let result = hash256(input);

        // Manually compute double SHA256 with sha2 crate
        use sha2::{Digest, Sha256};
        let first = Sha256::digest(input);
        let expected = Sha256::digest(first);

        assert_eq!(&result[..], &expected[..]);
    }

    #[test]
    fn test_double_sha256_zero() {
        // Test with 64 zero bytes (same as AVX2 test)
        let input = vec![0u8; 64];
        let result = hash256(&input);

        // Compare with sha2 crate
        use sha2::{Digest, Sha256};
        let first = Sha256::digest(&input);
        let expected = Sha256::digest(first);

        assert_eq!(&result[..], &expected[..]);
    }
}
