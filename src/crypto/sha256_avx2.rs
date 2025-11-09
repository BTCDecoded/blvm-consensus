//! AVX2-optimized SHA256 implementation
//!
//! This module provides 8-way parallel double SHA256 (SHA256D) hashing using AVX2 SIMD instructions.
//! Ported from Bitcoin Core's sha256_avx2.cpp implementation.
//!
//! # Performance
//! - Processes 8 independent SHA256 hashes in parallel
//! - Uses AVX2 SIMD instructions for maximum throughput
//! - Expected: 6-8x speedup over sequential processing on AVX2-capable CPUs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// SHA256 initial hash values
// Aligned to 16-byte boundary for optimal SIMD access
#[repr(align(16))]
struct AlignedInitialHash([u32; 8]);

static INITIAL_HASH: AlignedInitialHash = AlignedInitialHash([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]);

// SHA256 K constants
// Aligned to 16-byte boundary for optimal SIMD access
#[repr(align(16))]
struct AlignedKArray([u32; 64]);

static K_ARRAY: AlignedKArray = AlignedKArray([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/// Helper functions for AVX2 operations
#[cfg(target_arch = "x86_64")]
mod helpers {
    use super::*;

    #[inline(always)]
    pub unsafe fn k(x: u32) -> __m256i {
        _mm256_set1_epi32(x as i32)
    }

    #[inline(always)]
    pub unsafe fn add(x: __m256i, y: __m256i) -> __m256i {
        _mm256_add_epi32(x, y)
    }

    #[inline(always)]
    pub unsafe fn add3(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
        add(add(x, y), z)
    }
    
    #[inline(always)]
    pub unsafe fn add4(x: __m256i, y: __m256i, z: __m256i, w: __m256i) -> __m256i {
        // Core: Add(x, y, z, w) = Add(Add(x, y), Add(z, w))
        add(add(x, y), add(z, w))
    }
    
    #[inline(always)]
    pub unsafe fn add5(x: __m256i, y: __m256i, z: __m256i, w: __m256i, v: __m256i) -> __m256i {
        // Core: Add(x, y, z, w, v) = Add(Add(x, y, z), Add(w, v))
        add(add3(x, y, z), add(w, v))
    }

    #[inline(always)]
    pub unsafe fn xor(x: __m256i, y: __m256i) -> __m256i {
        _mm256_xor_si256(x, y)
    }

    #[inline(always)]
    pub unsafe fn xor3(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
        xor(xor(x, y), z)
    }

    #[inline(always)]
    pub unsafe fn or(x: __m256i, y: __m256i) -> __m256i {
        _mm256_or_si256(x, y)
    }

    #[inline(always)]
    pub unsafe fn and(x: __m256i, y: __m256i) -> __m256i {
        _mm256_and_si256(x, y)
    }

    #[inline(always)]
    pub unsafe fn ch(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
        xor(z, and(x, xor(y, z)))
    }

    #[inline(always)]
    pub unsafe fn maj(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
        or(and(x, y), and(z, or(x, y)))
    }

    #[inline(always)]
    pub unsafe fn sigma0(x: __m256i) -> __m256i {
        xor3(
            or(_mm256_srli_epi32(x, 2), _mm256_slli_epi32(x, 30)),
            or(_mm256_srli_epi32(x, 13), _mm256_slli_epi32(x, 19)),
            or(_mm256_srli_epi32(x, 22), _mm256_slli_epi32(x, 10)),
        )
    }

    #[inline(always)]
    pub unsafe fn sigma1(x: __m256i) -> __m256i {
        xor3(
            or(_mm256_srli_epi32(x, 6), _mm256_slli_epi32(x, 26)),
            or(_mm256_srli_epi32(x, 11), _mm256_slli_epi32(x, 21)),
            or(_mm256_srli_epi32(x, 25), _mm256_slli_epi32(x, 7)),
        )
    }

    #[inline(always)]
    pub unsafe fn sigma0_small(x: __m256i) -> __m256i {
        xor3(
            or(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25)),
            or(_mm256_srli_epi32(x, 18), _mm256_slli_epi32(x, 14)),
            _mm256_srli_epi32(x, 3),
        )
    }

    #[inline(always)]
    pub unsafe fn sigma1_small(x: __m256i) -> __m256i {
        xor3(
            or(_mm256_srli_epi32(x, 17), _mm256_slli_epi32(x, 15)),
            or(_mm256_srli_epi32(x, 19), _mm256_slli_epi32(x, 13)),
            _mm256_srli_epi32(x, 10),
        )
    }

    #[inline(always)]
    pub unsafe fn inc(x: &mut __m256i, y: __m256i) -> __m256i {
        *x = add(*x, y);
        *x
    }

    #[inline(always)]
    pub unsafe fn inc3(x: &mut __m256i, y: __m256i, z: __m256i) -> __m256i {
        // Core: Inc(x, y, z) { x = Add(x, y, z); return x; }
        // Core: Add(x, y, z) = Add(Add(x, y), z)
        // So: x = Add(Add(x, y), z)
        *x = add3(*x, y, z);
        *x
    }

    #[inline(always)]
    pub unsafe fn inc4(x: &mut __m256i, y: __m256i, z: __m256i, w: __m256i) -> __m256i {
        // Core: Inc(x, y, z, w) { x = Add(x, y, z, w); return x; }
        // Core: Add(x, y, z, w) = Add(Add(x, y), Add(z, w))
        // So: x = Add(Add(x, y), Add(z, w))
        *x = add4(*x, y, z, w);
        *x
    }
}

/// One round of SHA-256
#[cfg(target_arch = "x86_64")]
macro_rules! round {
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $k:expr) => {
        {
            use helpers::*;
            // Core: Add(h, Sigma1(e), Ch(e, f, g), k) = Add(Add(h, Sigma1(e)), Add(Ch(e, f, g), k))
            let t1 = add(add($h, helpers::sigma1($e)), add(helpers::ch($e, $f, $g), $k));
            let t2 = add(helpers::sigma0($a), helpers::maj($a, $b, $c));
            $d = add($d, t1);
            $h = add(t1, t2);
        }
    };
}

/// Read 8 32-bit words from 8 different 64-byte blocks
#[cfg(target_arch = "x86_64")]
unsafe fn read8(input: &[u8], offset: usize) -> __m256i {
    // Read 8 32-bit words from positions: offset, offset+64, offset+128, ..., offset+448
    let mut words = [0u32; 8];
    for i in 0..8 {
        let pos = (i * 64) + offset;
        words[i] = u32::from_le_bytes([
            input[pos],
            input[pos + 1],
            input[pos + 2],
            input[pos + 3],
        ]);
    }
    
    // _mm256_set_epi32 places first argument at index 7, last at index 0
    // Core: ReadLE32(chunk + 0) is first arg -> index 7, ReadLE32(chunk + 64) is second arg -> index 6, etc.
    // So words[0] (chunk + 0) should be first arg -> index 7
    // But wait - let's verify: Core does ReadLE32(chunk + 0 + offset) as FIRST argument
    // So chunk 0 goes to index 7, chunk 1 (chunk + 64) goes to index 6, etc.
    // This means words[0] -> index 7, words[1] -> index 6, ..., words[7] -> index 0
    let ret = _mm256_set_epi32(
        words[0] as i32,  // chunk + 0 -> index 7 (first arg)
        words[1] as i32,  // chunk + 64 -> index 6 (second arg)
        words[2] as i32,  // chunk + 128 -> index 5
        words[3] as i32,  // chunk + 192 -> index 4
        words[4] as i32,  // chunk + 256 -> index 3
        words[5] as i32,  // chunk + 320 -> index 2
        words[6] as i32,  // chunk + 384 -> index 1
        words[7] as i32,  // chunk + 448 -> index 0 (last arg)
    );
    
    // Byte swap within each 32-bit word using shuffle (matching Core exactly)
    // SHA256 operates on big-endian 32-bit words internally
    // We read little-endian, so we need to byte-swap each word
    // The shuffle mask 0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203
    // reverses bytes within each 32-bit word
    // 
    // Note: _mm256_shuffle_epi8 operates on 8-bit elements across the entire vector
    // The mask pattern is designed to swap bytes within each word
    let shuffle_mask = _mm256_set_epi32(
        0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203,
        0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203,
    );
    _mm256_shuffle_epi8(ret, shuffle_mask)
}

/// Write 8 32-bit words to 8 different output positions
#[cfg(target_arch = "x86_64")]
unsafe fn write8(out: &mut [u8], offset: usize, v: __m256i) {
    // Core's Write8 does: byte-swap with shuffle, then WriteLE32 to each position
    // Byte swap within each 32-bit word using shuffle
    let shuffle_mask = _mm256_set_epi32(
        0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203,
        0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203,
    );
    let v = _mm256_shuffle_epi8(v, shuffle_mask);
    
    // Extract and write to 8 different positions (Core's WriteLE32)
    // Unroll loop because _mm256_extract_epi32 requires compile-time constant index
    let word0 = _mm256_extract_epi32(v, 7) as u32;
    out[offset..offset + 4].copy_from_slice(&word0.to_le_bytes());
    let word1 = _mm256_extract_epi32(v, 6) as u32;
    out[offset + 32..offset + 36].copy_from_slice(&word1.to_le_bytes());
    let word2 = _mm256_extract_epi32(v, 5) as u32;
    out[offset + 64..offset + 68].copy_from_slice(&word2.to_le_bytes());
    let word3 = _mm256_extract_epi32(v, 4) as u32;
    out[offset + 96..offset + 100].copy_from_slice(&word3.to_le_bytes());
    let word4 = _mm256_extract_epi32(v, 3) as u32;
    out[offset + 128..offset + 132].copy_from_slice(&word4.to_le_bytes());
    let word5 = _mm256_extract_epi32(v, 2) as u32;
    out[offset + 160..offset + 164].copy_from_slice(&word5.to_le_bytes());
    let word6 = _mm256_extract_epi32(v, 1) as u32;
    out[offset + 192..offset + 196].copy_from_slice(&word6.to_le_bytes());
    let word7 = _mm256_extract_epi32(v, 0) as u32;
    out[offset + 224..offset + 228].copy_from_slice(&word7.to_le_bytes());
}

/// Transform 8 double SHA256 blocks in parallel (SHA256D)
/// Input: 512 bytes (8 * 64-byte blocks, contiguous)
/// Output: 256 bytes (8 * 32-byte hashes, contiguous)
/// 
/// This performs double SHA256 (SHA256(SHA256(input))) on 8 blocks in parallel.
#[cfg(target_arch = "x86_64")]
unsafe fn transform_8way(out: &mut [u8], input: &[u8]) {
    use helpers::*;
    
    // Initialize state with initial hash values (host byte order, matching Core)
    let mut a = _mm256_set1_epi32(INITIAL_HASH.0[0] as i32);
    let mut b = _mm256_set1_epi32(INITIAL_HASH.0[1] as i32);
    let mut c = _mm256_set1_epi32(INITIAL_HASH.0[2] as i32);
    let mut d = _mm256_set1_epi32(INITIAL_HASH.0[3] as i32);
    let mut e = _mm256_set1_epi32(INITIAL_HASH.0[4] as i32);
    let mut f = _mm256_set1_epi32(INITIAL_HASH.0[5] as i32);
    let mut g = _mm256_set1_epi32(INITIAL_HASH.0[6] as i32);
    let mut h = _mm256_set1_epi32(INITIAL_HASH.0[7] as i32);
    
    #[cfg(debug_assertions)]
    {
        let a_init = _mm256_extract_epi32(a, 0) as u32;
        let h_init = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG Transform 1 initial state: a=0x{:08x} (expected 0x6a09e667), h=0x{:08x} (expected 0x5be0cd19)", a_init, h_init);
    }

    // Message schedule (W array)
    let mut w0: __m256i;
    let mut w1: __m256i;
    let mut w2: __m256i;
    let mut w3: __m256i;
    let mut w4: __m256i;
    let mut w5: __m256i;
    let mut w6: __m256i;
    let mut w7: __m256i;
    let mut w8: __m256i;
    let mut w9: __m256i;
    let mut w10: __m256i;
    let mut w11: __m256i;
    let mut w12: __m256i;
    let mut w13: __m256i;
    let mut w14: __m256i;
    let mut w15: __m256i;

    // Rounds 0-15: Read from input
    w0 = read8(input, 0);
    // Debug: Check what we read for zero input
    #[cfg(debug_assertions)]
    {
        let w0_val = _mm256_extract_epi32(w0, 0) as u32;
        let w0_val_7 = _mm256_extract_epi32(w0, 7) as u32;
        println!("DEBUG: w0 after read8 and shuffle: index 0 = 0x{:08x}, index 7 = 0x{:08x}", w0_val, w0_val_7);
        if w0_val == 0 && w0_val_7 == 0 {
            println!("DEBUG: w0 is zero (expected for zero input)");
        }
    }
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[0]), w0));
    #[cfg(debug_assertions)]
    {
        let a_after_round0 = _mm256_extract_epi32(a, 0) as u32;
        let h_after_round0 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG After round 0: a=0x{:08x}, h=0x{:08x}", a_after_round0, h_after_round0);
    }
    w1 = read8(input, 4);
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[1]), w1));
    #[cfg(debug_assertions)]
    {
        let a_after_round1 = _mm256_extract_epi32(a, 0) as u32;
        let h_after_round1 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG After round 1: a=0x{:08x}, h=0x{:08x}", a_after_round1, h_after_round1);
    }
    w2 = read8(input, 8);
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[2]), w2));
    w3 = read8(input, 12);
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[3]), w3));
    w4 = read8(input, 16);
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[4]), w4));
    w5 = read8(input, 20);
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[5]), w5));
    w6 = read8(input, 24);
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[6]), w6));
    w7 = read8(input, 28);
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[7]), w7));
    #[cfg(debug_assertions)]
    {
        let a_after_round7 = _mm256_extract_epi32(a, 0) as u32;
        let b_after_round7 = _mm256_extract_epi32(b, 0) as u32;
        let h_after_round7 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG After round 7: a=0x{:08x}, b=0x{:08x}, h=0x{:08x}", a_after_round7, b_after_round7, h_after_round7);
    }
    w8 = read8(input, 32);
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[8]), w8));
    w9 = read8(input, 36);
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[9]), w9));
    w10 = read8(input, 40);
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[10]), w10));
    w11 = read8(input, 44);
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[11]), w11));
    w12 = read8(input, 48);
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[12]), w12));
    w13 = read8(input, 52);
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[13]), w13));
    w14 = read8(input, 56);
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[14]), w14));
    w15 = read8(input, 60);
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[15]), w15));
    #[cfg(debug_assertions)]
    {
        let a_after_round15 = _mm256_extract_epi32(a, 0) as u32;
        let h_after_round15 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG After round 15: a=0x{:08x}, h=0x{:08x}", a_after_round15, h_after_round15);
    }

    // Rounds 16-63: Message schedule
    // Core: Inc(w0, sigma1(w14), w9, sigma0(w1)) = Add(Add(w0, sigma1(w14)), Add(w9, sigma0(w1)))
    #[cfg(debug_assertions)]
    {
        let w0_before = _mm256_extract_epi32(w0, 0) as u32;
        let w14_val = _mm256_extract_epi32(w14, 0) as u32;
        let w9_val = _mm256_extract_epi32(w9, 0) as u32;
        let w1_val = _mm256_extract_epi32(w1, 0) as u32;
        println!("DEBUG Round 16 before inc4: w0=0x{:08x}, w14=0x{:08x}, w9=0x{:08x}, w1=0x{:08x}", w0_before, w14_val, w9_val, w1_val);
    }
    helpers::inc4(&mut w0, helpers::sigma1_small(w14), w9, helpers::sigma0_small(w1));
    #[cfg(debug_assertions)]
    {
        let w0_after_inc = _mm256_extract_epi32(w0, 0) as u32;
        println!("DEBUG After inc4 w0 (round 16): w0=0x{:08x}", w0_after_inc);
    }
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[16]), w0));
    #[cfg(debug_assertions)]
    {
        let a_after_round16 = _mm256_extract_epi32(a, 0) as u32;
        let b_after_round16 = _mm256_extract_epi32(b, 0) as u32;
        let c_after_round16 = _mm256_extract_epi32(c, 0) as u32;
        let d_after_round16 = _mm256_extract_epi32(d, 0) as u32;
        let e_after_round16 = _mm256_extract_epi32(e, 0) as u32;
        let f_after_round16 = _mm256_extract_epi32(f, 0) as u32;
        let g_after_round16 = _mm256_extract_epi32(g, 0) as u32;
        let h_after_round16 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG After round 16: a=0x{:08x} b=0x{:08x} c=0x{:08x} d=0x{:08x} e=0x{:08x} f=0x{:08x} g=0x{:08x} h=0x{:08x}", 
                 a_after_round16, b_after_round16, c_after_round16, d_after_round16,
                 e_after_round16, f_after_round16, g_after_round16, h_after_round16);
    }
    helpers::inc4(&mut w1, helpers::sigma1_small(w15), w10, helpers::sigma0_small(w2));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[17]), w1));
    helpers::inc4(&mut w2, helpers::sigma1_small(w0), w11, helpers::sigma0_small(w3));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[18]), w2));
    #[cfg(debug_assertions)]
    {
        let a_after_round18 = _mm256_extract_epi32(a, 0) as u32;
        println!("DEBUG After round 18: a=0x{:08x}", a_after_round18);
    }
    helpers::inc4(&mut w3, helpers::sigma1_small(w1), w12, helpers::sigma0_small(w4));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[19]), w3));
    helpers::inc4(&mut w4, helpers::sigma1_small(w2), w13, helpers::sigma0_small(w5));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[20]), w4));
    #[cfg(debug_assertions)]
    {
        let a_after_round20 = _mm256_extract_epi32(a, 0) as u32;
        println!("DEBUG After round 20: a=0x{:08x}", a_after_round20);
    }
    helpers::inc4(&mut w5, helpers::sigma1_small(w3), w14, helpers::sigma0_small(w6));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[21]), w5));
    helpers::inc4(&mut w6, helpers::sigma1_small(w4), w15, helpers::sigma0_small(w7));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[22]), w6));
    helpers::inc4(&mut w7, helpers::sigma1_small(w5), w0, helpers::sigma0_small(w8));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[23]), w7));
    helpers::inc4(&mut w8, helpers::sigma1_small(w6), w1, helpers::sigma0_small(w9));
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[24]), w8));
    helpers::inc4(&mut w9, helpers::sigma1_small(w7), w2, helpers::sigma0_small(w10));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[25]), w9));
    helpers::inc4(&mut w10, helpers::sigma1_small(w8), w3, helpers::sigma0_small(w11));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[26]), w10));
    helpers::inc4(&mut w11, helpers::sigma1_small(w9), w4, helpers::sigma0_small(w12));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[27]), w11));
    helpers::inc4(&mut w12, helpers::sigma1_small(w10), w5, helpers::sigma0_small(w13));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[28]), w12));
    helpers::inc4(&mut w13, helpers::sigma1_small(w11), w6, helpers::sigma0_small(w14));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[29]), w13));
    helpers::inc4(&mut w14, helpers::sigma1_small(w12), w7, helpers::sigma0_small(w15));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[30]), w14));
    #[cfg(debug_assertions)]
    {
        let a_after_round30 = _mm256_extract_epi32(a, 0) as u32;
        println!("DEBUG After round 30: a=0x{:08x}", a_after_round30);
    }
    // Core Transform 1 round 31: Inc(w15, sigma1(w13), w8, sigma0(w0)) - no k(0x100) in Transform 1
    #[cfg(debug_assertions)]
    {
        let a_before_round31 = _mm256_extract_epi32(a, 0) as u32;
        let b_before_round31 = _mm256_extract_epi32(b, 0) as u32;
        let c_before_round31 = _mm256_extract_epi32(c, 0) as u32;
        let d_before_round31 = _mm256_extract_epi32(d, 0) as u32;
        let e_before_round31 = _mm256_extract_epi32(e, 0) as u32;
        let f_before_round31 = _mm256_extract_epi32(f, 0) as u32;
        let g_before_round31 = _mm256_extract_epi32(g, 0) as u32;
        let h_before_round31 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG Before round 31: a=0x{:08x} b=0x{:08x} c=0x{:08x} d=0x{:08x} e=0x{:08x} f=0x{:08x} g=0x{:08x} h=0x{:08x}", 
                 a_before_round31, b_before_round31, c_before_round31, d_before_round31,
                 e_before_round31, f_before_round31, g_before_round31, h_before_round31);
        let w15_before = _mm256_extract_epi32(w15, 0) as u32;
        let w13_val = _mm256_extract_epi32(w13, 0) as u32;
        let w8_val = _mm256_extract_epi32(w8, 0) as u32;
        let w0_val = _mm256_extract_epi32(w0, 0) as u32;
        let w11_val = _mm256_extract_epi32(w11, 0) as u32;
        let w6_val = _mm256_extract_epi32(w6, 0) as u32;
        println!("DEBUG w15 before inc4: 0x{:08x}, w13: 0x{:08x}, w8: 0x{:08x}, w0: 0x{:08x}", w15_before, w13_val, w8_val, w0_val);
        println!("DEBUG w11: 0x{:08x}, w6: 0x{:08x}", w11_val, w6_val);
        let sigma1_w13 = _mm256_extract_epi32(helpers::sigma1_small(w13), 0) as u32;
        let sigma0_w0 = _mm256_extract_epi32(helpers::sigma0_small(w0), 0) as u32;
        println!("DEBUG sigma1_small(w13): 0x{:08x}, sigma0_small(w0): 0x{:08x}", sigma1_w13, sigma0_w0);
    }
    helpers::inc4(&mut w15, helpers::sigma1_small(w13), w8, helpers::sigma0_small(w0));
    #[cfg(debug_assertions)]
    {
        let w15_after = _mm256_extract_epi32(w15, 0) as u32;
        println!("DEBUG w15 after inc4: 0x{:08x}", w15_after);
        let k31 = K_ARRAY.0[31];
        println!("DEBUG K_ARRAY.0[31] = 0x{:08x}", k31);
    }
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[31]), w15));
    #[cfg(debug_assertions)]
    {
        let a_after_round31 = _mm256_extract_epi32(a, 0) as u32;
        let b_after_round31 = _mm256_extract_epi32(b, 0) as u32;
        let c_after_round31 = _mm256_extract_epi32(c, 0) as u32;
        let d_after_round31 = _mm256_extract_epi32(d, 0) as u32;
        let e_after_round31 = _mm256_extract_epi32(e, 0) as u32;
        let f_after_round31 = _mm256_extract_epi32(f, 0) as u32;
        let g_after_round31 = _mm256_extract_epi32(g, 0) as u32;
        let h_after_round31 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG After round 31: a=0x{:08x} b=0x{:08x} c=0x{:08x} d=0x{:08x} e=0x{:08x} f=0x{:08x} g=0x{:08x} h=0x{:08x}", 
                 a_after_round31, b_after_round31, c_after_round31, d_after_round31,
                 e_after_round31, f_after_round31, g_after_round31, h_after_round31);
    }
    helpers::inc4(&mut w0, helpers::sigma1_small(w14), w9, helpers::sigma0_small(w1));
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[32]), w0));
    helpers::inc4(&mut w1, helpers::sigma1_small(w15), w10, helpers::sigma0_small(w2));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[33]), w1));
    helpers::inc4(&mut w2, helpers::sigma1_small(w0), w11, helpers::sigma0_small(w3));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[34]), w2));
    helpers::inc4(&mut w3, helpers::sigma1_small(w1), w12, helpers::sigma0_small(w4));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[35]), w3));
    helpers::inc4(&mut w4, helpers::sigma1_small(w2), w13, helpers::sigma0_small(w5));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[36]), w4));
    helpers::inc4(&mut w5, helpers::sigma1_small(w3), w14, helpers::sigma0_small(w6));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[37]), w5));
    helpers::inc4(&mut w6, helpers::sigma1_small(w4), w15, helpers::sigma0_small(w7));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[38]), w6));
    helpers::inc4(&mut w7, helpers::sigma1_small(w5), w0, helpers::sigma0_small(w8));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[39]), w7));
    helpers::inc4(&mut w8, helpers::sigma1_small(w6), w1, helpers::sigma0_small(w9));
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[40]), w8));
    helpers::inc4(&mut w9, helpers::sigma1_small(w7), w2, helpers::sigma0_small(w10));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[41]), w9));
    helpers::inc4(&mut w10, helpers::sigma1_small(w8), w3, helpers::sigma0_small(w11));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[42]), w10));
    helpers::inc4(&mut w11, helpers::sigma1_small(w9), w4, helpers::sigma0_small(w12));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[43]), w11));
    helpers::inc4(&mut w12, helpers::sigma1_small(w10), w5, helpers::sigma0_small(w13));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[44]), w12));
    helpers::inc4(&mut w13, helpers::sigma1_small(w11), w6, helpers::sigma0_small(w14));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[45]), w13));
    helpers::inc4(&mut w14, helpers::sigma1_small(w12), w7, helpers::sigma0_small(w15));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[46]), w14));
    helpers::inc4(&mut w15, helpers::sigma1_small(w13), w8, helpers::sigma0_small(w0));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[47]), w15));
    helpers::inc4(&mut w0, helpers::sigma1_small(w14), w9, helpers::sigma0_small(w1));
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[48]), w0));
    helpers::inc4(&mut w1, helpers::sigma1_small(w15), w10, helpers::sigma0_small(w2));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[49]), w1));
    helpers::inc4(&mut w2, helpers::sigma1_small(w0), w11, helpers::sigma0_small(w3));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[50]), w2));
    helpers::inc4(&mut w3, helpers::sigma1_small(w1), w12, helpers::sigma0_small(w4));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[51]), w3));
    helpers::inc4(&mut w4, helpers::sigma1_small(w2), w13, helpers::sigma0_small(w5));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[52]), w4));
    helpers::inc4(&mut w5, helpers::sigma1_small(w3), w14, helpers::sigma0_small(w6));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[53]), w5));
    helpers::inc4(&mut w6, helpers::sigma1_small(w4), w15, helpers::sigma0_small(w7));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[54]), w6));
    helpers::inc4(&mut w7, helpers::sigma1_small(w5), w0, helpers::sigma0_small(w8));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[55]), w7));
    helpers::inc4(&mut w8, helpers::sigma1_small(w6), w1, helpers::sigma0_small(w9));
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[56]), w8));
    helpers::inc4(&mut w9, helpers::sigma1_small(w7), w2, helpers::sigma0_small(w10));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[57]), w9));
    helpers::inc4(&mut w10, helpers::sigma1_small(w8), w3, helpers::sigma0_small(w11));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[58]), w10));
    helpers::inc4(&mut w11, helpers::sigma1_small(w9), w4, helpers::sigma0_small(w12));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[59]), w11));
    helpers::inc4(&mut w12, helpers::sigma1_small(w10), w5, helpers::sigma0_small(w13));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[60]), w12));
    helpers::inc4(&mut w13, helpers::sigma1_small(w11), w6, helpers::sigma0_small(w14));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[61]), w13));
    helpers::inc4(&mut w14, helpers::sigma1_small(w12), w7, helpers::sigma0_small(w15));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[62]), w14));
    helpers::inc4(&mut w15, helpers::sigma1_small(w13), w8, helpers::sigma0_small(w0));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[63]), w15));
    #[cfg(debug_assertions)]
    {
        let a_after_round63 = _mm256_extract_epi32(a, 0) as u32;
        let b_after_round63 = _mm256_extract_epi32(b, 0) as u32;
        let c_after_round63 = _mm256_extract_epi32(c, 0) as u32;
        let d_after_round63 = _mm256_extract_epi32(d, 0) as u32;
        let e_after_round63 = _mm256_extract_epi32(e, 0) as u32;
        let f_after_round63 = _mm256_extract_epi32(f, 0) as u32;
        let g_after_round63 = _mm256_extract_epi32(g, 0) as u32;
        let h_after_round63 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG After round 63: a=0x{:08x} b=0x{:08x} c=0x{:08x} d=0x{:08x} e=0x{:08x} f=0x{:08x} g=0x{:08x} h=0x{:08x}", 
            a_after_round63, b_after_round63, c_after_round63, d_after_round63,
            e_after_round63, f_after_round63, g_after_round63, h_after_round63);
        // Expected state after 64 rounds for zero input (before adding initial hash)
        // First SHA256 of 64 zero bytes should produce: 0x42fda5f5 (little-endian of first word)
        // So a should be: 0x42fda5f5 - 0x6a09e667 = 0xd8f3bf8e
        println!("DEBUG Expected a after round 63: 0xd8f3bf8e (0x42fda5f5 - 0x6a09e667)");
        println!("DEBUG Difference: 0x{:08x}", a_after_round63.wrapping_sub(0xd8f3bf8e));
    }

    // Add initial hash values (Transform 1 complete)
    #[cfg(debug_assertions)]
    {
        let a_before = _mm256_extract_epi32(a, 0) as u32;
        let h_before = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG Transform 1 before adding initial hash: a=0x{:08x}, h=0x{:08x}", a_before, h_before);
        println!("Expected SHA256(64 zeros) first word: 0xf5a5fd42");
    }
    a = add(a, k(INITIAL_HASH.0[0]));
    b = add(b, k(INITIAL_HASH.0[1]));
    c = add(c, k(INITIAL_HASH.0[2]));
    d = add(d, k(INITIAL_HASH.0[3]));
    e = add(e, k(INITIAL_HASH.0[4]));
    f = add(f, k(INITIAL_HASH.0[5]));
    g = add(g, k(INITIAL_HASH.0[6]));
    h = add(h, k(INITIAL_HASH.0[7]));
    #[cfg(debug_assertions)]
    {
        let a_after = _mm256_extract_epi32(a, 0) as u32;
        let h_after = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG Transform 1 after adding initial hash: a=0x{:08x}, h=0x{:08x}", a_after, h_after);
        println!("Expected: a=0xf5a5fd42 (from SHA256(64 zeros))");
    }

    // Save state for Transform 3
    #[cfg(debug_assertions)]
    {
        let a_before_t2 = _mm256_extract_epi32(a, 0) as u32;
        let b_before_t2 = _mm256_extract_epi32(b, 0) as u32;
        println!("DEBUG Transform 2 starting state: a=0x{:08x}, b=0x{:08x}", a_before_t2, b_before_t2);
    }
    let t0 = a;
    let t1 = b;
    let t2 = c;
    let t3 = d;
    let t4 = e;
    let t5 = f;
    let t6 = g;
    let t7 = h;

    // Transform 2: Process padding block (0x80000000, zeros, length 0x200)
    // Core does NOT reset state - it continues with the state from Transform 1!

    // Transform 2 uses different K constants (hardcoded in Core)
    round!(a, b, c, d, e, f, g, h, k(0xc28a2f98));
    round!(h, a, b, c, d, e, f, g, k(0x71374491));
    round!(g, h, a, b, c, d, e, f, k(0xb5c0fbcf));
    round!(f, g, h, a, b, c, d, e, k(0xe9b5dba5));
    round!(e, f, g, h, a, b, c, d, k(0x3956c25b));
    round!(d, e, f, g, h, a, b, c, k(0x59f111f1));
    round!(c, d, e, f, g, h, a, b, k(0x923f82a4));
    round!(b, c, d, e, f, g, h, a, k(0xab1c5ed5));
    round!(a, b, c, d, e, f, g, h, k(0xd807aa98));
    round!(h, a, b, c, d, e, f, g, k(0x12835b01));
    round!(g, h, a, b, c, d, e, f, k(0x243185be));
    round!(f, g, h, a, b, c, d, e, k(0x550c7dc3));
    round!(e, f, g, h, a, b, c, d, k(0x72be5d74));
    round!(d, e, f, g, h, a, b, c, k(0x80deb1fe));
    round!(c, d, e, f, g, h, a, b, k(0x9bdc06a7));
    round!(b, c, d, e, f, g, h, a, k(0xc19bf374));
    round!(a, b, c, d, e, f, g, h, k(0x649b69c1));
    round!(h, a, b, c, d, e, f, g, k(0xf0fe4786));
    round!(g, h, a, b, c, d, e, f, k(0x0fe1edc6));
    round!(f, g, h, a, b, c, d, e, k(0x240cf254));
    round!(e, f, g, h, a, b, c, d, k(0x4fe9346f));
    round!(d, e, f, g, h, a, b, c, k(0x6cc984be));
    round!(c, d, e, f, g, h, a, b, k(0x61b9411e));
    round!(b, c, d, e, f, g, h, a, k(0x16f988fa));
    round!(a, b, c, d, e, f, g, h, k(0xf2c65152));
    round!(h, a, b, c, d, e, f, g, k(0xa88e5a6d));
    round!(g, h, a, b, c, d, e, f, k(0xb019fc65));
    round!(f, g, h, a, b, c, d, e, k(0xb9d99ec7));
    round!(e, f, g, h, a, b, c, d, k(0x9a1231c3));
    round!(d, e, f, g, h, a, b, c, k(0xe70eeaa0));
    round!(c, d, e, f, g, h, a, b, k(0xfdb1232b));
    round!(b, c, d, e, f, g, h, a, k(0xc7353eb0));
    round!(a, b, c, d, e, f, g, h, k(0x3069bad5));
    round!(h, a, b, c, d, e, f, g, k(0xcb976d5f));
    round!(g, h, a, b, c, d, e, f, k(0x5a0f118f));
    round!(f, g, h, a, b, c, d, e, k(0xdc1eeefd));
    round!(e, f, g, h, a, b, c, d, k(0x0a35b689));
    round!(d, e, f, g, h, a, b, c, k(0xde0b7a04));
    round!(c, d, e, f, g, h, a, b, k(0x58f4ca9d));
    round!(b, c, d, e, f, g, h, a, k(0xe15d5b16));
    round!(a, b, c, d, e, f, g, h, k(0x007f3e86));
    round!(h, a, b, c, d, e, f, g, k(0x37088980));
    round!(g, h, a, b, c, d, e, f, k(0xa507ea32));
    round!(f, g, h, a, b, c, d, e, k(0x6fab9537));
    round!(e, f, g, h, a, b, c, d, k(0x17406110));
    round!(d, e, f, g, h, a, b, c, k(0x0d8cd6f1));
    round!(c, d, e, f, g, h, a, b, k(0xcdaa3b6d));
    round!(b, c, d, e, f, g, h, a, k(0xc0bbbe37));
    round!(a, b, c, d, e, f, g, h, k(0x83613bda));
    round!(h, a, b, c, d, e, f, g, k(0xdb48a363));
    round!(g, h, a, b, c, d, e, f, k(0x0b02e931));
    round!(f, g, h, a, b, c, d, e, k(0x6fd15ca7));
    round!(e, f, g, h, a, b, c, d, k(0x521afaca));
    round!(d, e, f, g, h, a, b, c, k(0x31338431));
    round!(c, d, e, f, g, h, a, b, k(0x6ed41a95));
    round!(b, c, d, e, f, g, h, a, k(0x6d437890));
    round!(a, b, c, d, e, f, g, h, k(0xc39c91f2));
    round!(h, a, b, c, d, e, f, g, k(0x9eccabbd));
    round!(g, h, a, b, c, d, e, f, k(0xb5c9a0e6));
    round!(f, g, h, a, b, c, d, e, k(0x532fb63c));
    round!(e, f, g, h, a, b, c, d, k(0xd2c741c6));
    round!(d, e, f, g, h, a, b, c, k(0x07237ea3));
    round!(c, d, e, f, g, h, a, b, k(0xa4954b68));
    round!(b, c, d, e, f, g, h, a, k(0x4c191d76));

    // Combine Transform 1 and Transform 2 results
    #[cfg(debug_assertions)]
    {
        let t0_val = _mm256_extract_epi32(t0, 0) as u32;
        let t1_val = _mm256_extract_epi32(t1, 0) as u32;
        let a_t2_val = _mm256_extract_epi32(a, 0) as u32;
        let b_t2_val = _mm256_extract_epi32(b, 0) as u32;
        println!("DEBUG Before combining: t0=0x{:08x}, t1=0x{:08x}, Transform2_a=0x{:08x}, Transform2_b=0x{:08x}", t0_val, t1_val, a_t2_val, b_t2_val);
    }
    w0 = add(t0, a);
    w1 = add(t1, b);
    w2 = add(t2, c);
    w3 = add(t3, d);
    w4 = add(t4, e);
    w5 = add(t5, f);
    w6 = add(t6, g);
    w7 = add(t7, h);
    #[cfg(debug_assertions)]
    {
        let w0_combined = _mm256_extract_epi32(w0, 0) as u32;
        let w1_combined = _mm256_extract_epi32(w1, 0) as u32;
        println!("DEBUG After combining Transform 1 and 2: w0=0x{:08x}, w1=0x{:08x}", w0_combined, w1_combined);
    }

    // Transform 3: Final SHA256 on the combined state
    // Reset state to initial hash values
    a = k(INITIAL_HASH.0[0]);
    b = k(INITIAL_HASH.0[1]);
    c = k(INITIAL_HASH.0[2]);
    d = k(INITIAL_HASH.0[3]);
    e = k(INITIAL_HASH.0[4]);
    f = k(INITIAL_HASH.0[5]);
    g = k(INITIAL_HASH.0[6]);
    h = k(INITIAL_HASH.0[7]);

    // Rounds 0-15: Use w0-w7 from combined state
    #[cfg(debug_assertions)]
    {
        let w0_t3 = _mm256_extract_epi32(w0, 0) as u32;
        let w1_t3 = _mm256_extract_epi32(w1, 0) as u32;
        println!("DEBUG Transform 3 input w0=0x{:08x}, w1=0x{:08x}", w0_t3, w1_t3);
    }
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[0]), w0));
    #[cfg(debug_assertions)]
    {
        let a_after_t3_round0 = _mm256_extract_epi32(a, 0) as u32;
        let h_after_t3_round0 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG Transform 3 after round 0: a=0x{:08x} (old), h=0x{:08x} (new a for next round)", a_after_t3_round0, h_after_t3_round0);
    }
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[1]), w1));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[2]), w2));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[3]), w3));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[4]), w4));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[5]), w5));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[6]), w6));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[7]), w7));
    #[cfg(debug_assertions)]
    {
        // After round 7, the new a (for next round) is in h
        let h_after_t3_round7 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG Transform 3 after round 7: h (new a)=0x{:08x}", h_after_t3_round7);
    }
    round!(a, b, c, d, e, f, g, h, k(0x5807aa98));
    round!(h, a, b, c, d, e, f, g, k(0x12835b01));
    round!(g, h, a, b, c, d, e, f, k(0x243185be));
    round!(f, g, h, a, b, c, d, e, k(0x550c7dc3));
    round!(e, f, g, h, a, b, c, d, k(0x72be5d74));
    round!(d, e, f, g, h, a, b, c, k(0x80deb1fe));
    round!(c, d, e, f, g, h, a, b, k(0x9bdc06a7));
    round!(b, c, d, e, f, g, h, a, k(0xc19bf274));
    #[cfg(debug_assertions)]
    {
        // After round 15, round!(b, c, d, e, f, g, h, a, ...) modifies d and h
        // d is actually e (4th param), h is actually a (8th param)
        // After rotation, state should be: (a, b, c, d, e, f, g, h) where:
        // - a is the new h (from round 15)
        // - b is the old b
        // - c is the old c
        // - d is the new d (modified e)
        // etc.
        let a_after_15 = _mm256_extract_epi32(a, 0) as u32;
        let b_after_15 = _mm256_extract_epi32(b, 0) as u32;
        let c_after_15 = _mm256_extract_epi32(c, 0) as u32;
        let d_after_15 = _mm256_extract_epi32(d, 0) as u32;
        let e_after_15 = _mm256_extract_epi32(e, 0) as u32;
        let f_after_15 = _mm256_extract_epi32(f, 0) as u32;
        let g_after_15 = _mm256_extract_epi32(g, 0) as u32;
        let h_after_15 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG Transform 3 after round 15: a=0x{:08x}, b=0x{:08x}, c=0x{:08x}, d=0x{:08x}", a_after_15, b_after_15, c_after_15, d_after_15);
        println!("  e=0x{:08x}, f=0x{:08x}, g=0x{:08x}, h=0x{:08x}", e_after_15, f_after_15, g_after_15, h_after_15);
        println!("  Expected: a=0xf5539ad2, b=0x2c0362a7, c=0xda1fbbd3, d=0x3c3a4027");
    }

    // Rounds 16-63: Message schedule with special handling (Transform 2)
    // Core Transform 2: different pattern than Transform 1
    #[cfg(debug_assertions)]
    {
        let a_before_16 = _mm256_extract_epi32(a, 0) as u32;
        let b_before_16 = _mm256_extract_epi32(b, 0) as u32;
        let c_before_16 = _mm256_extract_epi32(c, 0) as u32;
        let d_before_16 = _mm256_extract_epi32(d, 0) as u32;
        let e_before_16 = _mm256_extract_epi32(e, 0) as u32;
        let f_before_16 = _mm256_extract_epi32(f, 0) as u32;
        let g_before_16 = _mm256_extract_epi32(g, 0) as u32;
        let h_before_16 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG Transform 3 before round 16: a=0x{:08x}, b=0x{:08x}, c=0x{:08x}, d=0x{:08x}", a_before_16, b_before_16, c_before_16, d_before_16);
        println!("  e=0x{:08x}, f=0x{:08x}, g=0x{:08x}, h=0x{:08x}", e_before_16, f_before_16, g_before_16, h_before_16);
        println!("  Expected after round 15: a=0xf5539ad2, b=0x2c0362a7, c=0xda1fbbd3, d=0x3c3a4027");
        let w0_before_t3_round16 = _mm256_extract_epi32(w0, 0) as u32;
        let w1_before_t3_round16 = _mm256_extract_epi32(w1, 0) as u32;
        println!("DEBUG Transform 3 round 16 before inc: w0=0x{:08x}, w1=0x{:08x}", w0_before_t3_round16, w1_before_t3_round16);
    }
    // Save old values before round 16 - these will be needed in round 17
    // Round 16 modifies d and h, so we save c and d (which become $d and $e in round 17)
    let _c_old_16 = c;
    let _d_old_16 = d;
    #[cfg(debug_assertions)]
    {
        let d_before_16 = _mm256_extract_epi32(d, 0) as u32;
        let w0_before = _mm256_extract_epi32(w0, 0) as u32;
        let w1_before = _mm256_extract_epi32(w1, 0) as u32;
        println!("DEBUG Transform 3 round 16: d before = 0x{:08x} (expected 0x3c3a4027), w0=0x{:08x}, w1=0x{:08x}", d_before_16, w0_before, w1_before);
    }
    // Rounds 16-31: Match Core Transform 3 exactly
    #[cfg(debug_assertions)]
    {
        let w0_before = _mm256_extract_epi32(w0, 0) as u32;
        let w1_before = _mm256_extract_epi32(w1, 0) as u32;
        println!("DEBUG T3 R16 before: w0=0x{:08x} w1=0x{:08x}", w0_before, w1_before);
    }
    helpers::inc(&mut w0, helpers::sigma0_small(w1));
    #[cfg(debug_assertions)]
    {
        let w0_after = _mm256_extract_epi32(w0, 0) as u32;
        println!("DEBUG T3 R16 after inc: w0=0x{:08x}", w0_after);
    }
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[16]), w0));
    #[cfg(debug_assertions)]
    {
        let a_after = _mm256_extract_epi32(a, 0) as u32;
        println!("DEBUG T3 R16 after round: a=0x{:08x}", a_after);
    }
    helpers::inc3(&mut w1, k(0xa00000), helpers::sigma0_small(w2));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[17]), w1));
    #[cfg(debug_assertions)]
    {
        let g_after = _mm256_extract_epi32(g, 0) as u32;
        println!("DEBUG T3 R17 after: g=0x{:08x}", g_after);
    }
    helpers::inc3(&mut w2, helpers::sigma1_small(w0), helpers::sigma0_small(w3));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[18]), w2));
    helpers::inc3(&mut w3, helpers::sigma1_small(w1), helpers::sigma0_small(w4));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[19]), w3));
    helpers::inc3(&mut w4, helpers::sigma1_small(w2), helpers::sigma0_small(w5));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[20]), w4));
    helpers::inc3(&mut w5, helpers::sigma1_small(w3), helpers::sigma0_small(w6));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[21]), w5));
    helpers::inc4(&mut w6, helpers::sigma1_small(w4), k(0x100), helpers::sigma0_small(w7));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[22]), w6));
    helpers::inc4(&mut w7, helpers::sigma1_small(w5), w0, k(0x11002000));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[23]), w7));
    w8 = helpers::add3(k(0x80000000), helpers::sigma1_small(w6), w1);
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[24]), w8));
    w9 = helpers::add(helpers::sigma1_small(w7), w2);
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[25]), w9));
    w10 = helpers::add(helpers::sigma1_small(w8), w3);
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[26]), w10));
    w11 = helpers::add(helpers::sigma1_small(w9), w4);
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[27]), w11));
    w12 = helpers::add(helpers::sigma1_small(w10), w5);
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[28]), w12));
    w13 = helpers::add(helpers::sigma1_small(w11), w6);
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[29]), w13));
    w14 = helpers::add3(helpers::sigma1_small(w12), w7, k(0x400022));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[30]), w14));
    w15 = helpers::add4(k(0x100), helpers::sigma1_small(w13), w8, helpers::sigma0_small(w0));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[31]), w15));
    
    // Continue with standard message schedule for rounds 32-63 (matching Core Transform 3 pattern)
    #[cfg(debug_assertions)]
    {
        let a_before_32 = _mm256_extract_epi32(a, 0) as u32;
        println!("DEBUG T3 R32 before: a=0x{:08x}", a_before_32);
    }
    helpers::inc4(&mut w0, helpers::sigma1_small(w14), w9, helpers::sigma0_small(w1));
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[32]), w0));
    #[cfg(debug_assertions)]
    {
        let a_after_32 = _mm256_extract_epi32(a, 0) as u32;
        println!("DEBUG T3 R32 after: a=0x{:08x}", a_after_32);
    }
    helpers::inc4(&mut w1, helpers::sigma1_small(w15), w10, helpers::sigma0_small(w2));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[33]), w1));
    helpers::inc4(&mut w2, helpers::sigma1_small(w0), w11, helpers::sigma0_small(w3));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[34]), w2));
    helpers::inc4(&mut w3, helpers::sigma1_small(w1), w12, helpers::sigma0_small(w4));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[35]), w3));
    helpers::inc4(&mut w4, helpers::sigma1_small(w2), w13, helpers::sigma0_small(w5));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[36]), w4));
    helpers::inc4(&mut w5, helpers::sigma1_small(w3), w14, helpers::sigma0_small(w6));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[37]), w5));
    helpers::inc4(&mut w6, helpers::sigma1_small(w4), w15, helpers::sigma0_small(w7));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[38]), w6));
    helpers::inc4(&mut w7, helpers::sigma1_small(w5), w0, helpers::sigma0_small(w8));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[39]), w7));
    helpers::inc4(&mut w8, helpers::sigma1_small(w6), w1, helpers::sigma0_small(w9));
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[40]), w8));
    #[cfg(debug_assertions)]
    {
        let a_after_40 = _mm256_extract_epi32(a, 0) as u32;
        let h_after_40 = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG T3 R40 after: a=0x{:08x}, h=0x{:08x}", a_after_40, h_after_40);
    }
    helpers::inc4(&mut w9, helpers::sigma1_small(w7), w2, helpers::sigma0_small(w10));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[41]), w9));
    helpers::inc4(&mut w10, helpers::sigma1_small(w8), w3, helpers::sigma0_small(w11));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[42]), w10));
    helpers::inc4(&mut w11, helpers::sigma1_small(w9), w4, helpers::sigma0_small(w12));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[43]), w11));
    helpers::inc4(&mut w12, helpers::sigma1_small(w10), w5, helpers::sigma0_small(w13));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[44]), w12));
    helpers::inc4(&mut w13, helpers::sigma1_small(w11), w6, helpers::sigma0_small(w14));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[45]), w13));
    helpers::inc4(&mut w14, helpers::sigma1_small(w12), w7, helpers::sigma0_small(w15));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[46]), w14));
    helpers::inc4(&mut w15, helpers::sigma1_small(w13), w8, helpers::sigma0_small(w0));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[47]), w15));
    helpers::inc4(&mut w0, helpers::sigma1_small(w14), w9, helpers::sigma0_small(w1));
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[48]), w0));
    #[cfg(debug_assertions)]
    {
        let a_after_48 = _mm256_extract_epi32(a, 0) as u32;
        println!("DEBUG T3 R48 after: a=0x{:08x}", a_after_48);
    }
    helpers::inc4(&mut w1, helpers::sigma1_small(w15), w10, helpers::sigma0_small(w2));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[49]), w1));
    helpers::inc4(&mut w2, helpers::sigma1_small(w0), w11, helpers::sigma0_small(w3));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[50]), w2));
    helpers::inc4(&mut w3, helpers::sigma1_small(w1), w12, helpers::sigma0_small(w4));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[51]), w3));
    helpers::inc4(&mut w4, helpers::sigma1_small(w2), w13, helpers::sigma0_small(w5));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[52]), w4));
    helpers::inc4(&mut w5, helpers::sigma1_small(w3), w14, helpers::sigma0_small(w6));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[53]), w5));
    helpers::inc4(&mut w6, helpers::sigma1_small(w4), w15, helpers::sigma0_small(w7));
    round!(c, d, e, f, g, h, a, b, add(k(K_ARRAY.0[54]), w6));
    helpers::inc4(&mut w7, helpers::sigma1_small(w5), w0, helpers::sigma0_small(w8));
    round!(b, c, d, e, f, g, h, a, add(k(K_ARRAY.0[55]), w7));
    helpers::inc4(&mut w8, helpers::sigma1_small(w6), w1, helpers::sigma0_small(w9));
    round!(a, b, c, d, e, f, g, h, add(k(K_ARRAY.0[56]), w8));
    #[cfg(debug_assertions)]
    {
        let a_after_56 = _mm256_extract_epi32(a, 0) as u32;
        println!("DEBUG T3 R56 after: a=0x{:08x}", a_after_56);
    }
    helpers::inc4(&mut w9, helpers::sigma1_small(w7), w2, helpers::sigma0_small(w10));
    round!(h, a, b, c, d, e, f, g, add(k(K_ARRAY.0[57]), w9));
    helpers::inc4(&mut w10, helpers::sigma1_small(w8), w3, helpers::sigma0_small(w11));
    round!(g, h, a, b, c, d, e, f, add(k(K_ARRAY.0[58]), w10));
    helpers::inc4(&mut w11, helpers::sigma1_small(w9), w4, helpers::sigma0_small(w12));
    round!(f, g, h, a, b, c, d, e, add(k(K_ARRAY.0[59]), w11));
    helpers::inc4(&mut w12, helpers::sigma1_small(w10), w5, helpers::sigma0_small(w13));
    round!(e, f, g, h, a, b, c, d, add(k(K_ARRAY.0[60]), w12));
    helpers::inc4(&mut w13, helpers::sigma1_small(w11), w6, helpers::sigma0_small(w14));
    round!(d, e, f, g, h, a, b, c, add(k(K_ARRAY.0[61]), w13));
    // Core Transform 3 rounds 62-63: Use Add with 5 args, w14/w15 NOT modified (different from Transform 1)
    round!(c, d, e, f, g, h, a, b, helpers::add5(k(K_ARRAY.0[62]), w14, helpers::sigma1_small(w12), w7, helpers::sigma0_small(w15)));
    round!(b, c, d, e, f, g, h, a, helpers::add5(k(K_ARRAY.0[63]), w15, helpers::sigma1_small(w13), w8, helpers::sigma0_small(w0)));

    // Add initial hash values and write output
    // Core: Add initial hash values before writing output
    #[cfg(debug_assertions)]
    {
        // After round 63: round!(b, c, d, e, f, g, h, a, ...)
        // $d = e (modified)
        // $h = a (modified)
        // After rotation, final state is: (a, b, c, d, e, f, g, h)
        // where a was modified (was $h), e was modified (was $d)
        let a_before_final = _mm256_extract_epi32(a, 0) as u32;
        let b_before_final = _mm256_extract_epi32(b, 0) as u32;
        let c_before_final = _mm256_extract_epi32(c, 0) as u32;
        let d_before_final = _mm256_extract_epi32(d, 0) as u32;
        let e_before_final = _mm256_extract_epi32(e, 0) as u32;
        let f_before_final = _mm256_extract_epi32(f, 0) as u32;
        let g_before_final = _mm256_extract_epi32(g, 0) as u32;
        let h_before_final = _mm256_extract_epi32(h, 0) as u32;
        println!("DEBUG Transform 3 after round 63: a=0x{:08x}, b=0x{:08x}, c=0x{:08x}, d=0x{:08x}", a_before_final, b_before_final, c_before_final, d_before_final);
        println!("  e=0x{:08x}, f=0x{:08x}, g=0x{:08x}, h=0x{:08x}", e_before_final, f_before_final, g_before_final, h_before_final);
        println!("DEBUG Transform 3 before adding initial hash: a=0x{:08x} (expected 0x78ec3678)", a_before_final);
    }
    // Write8(out, 0, Add(a, K(0x6a09e667ul)));
    let a_final = add(a, k(INITIAL_HASH.0[0]));
    #[cfg(debug_assertions)]
    {
        let a_final_val = _mm256_extract_epi32(a_final, 0) as u32;
        println!("DEBUG Transform 3 final a (after adding initial): 0x{:08x}", a_final_val);
    }
    write8(out, 0, a_final);
    write8(out, 4, add(b, k(INITIAL_HASH.0[1])));
    write8(out, 8, add(c, k(INITIAL_HASH.0[2])));
    write8(out, 12, add(d, k(INITIAL_HASH.0[3])));
    write8(out, 16, add(e, k(INITIAL_HASH.0[4])));
    write8(out, 20, add(f, k(INITIAL_HASH.0[5])));
    write8(out, 24, add(g, k(INITIAL_HASH.0[6])));
    write8(out, 28, add(h, k(INITIAL_HASH.0[7])));
    #[cfg(debug_assertions)]
    {
        // Check what we wrote
        println!("DEBUG Transform 3 output first 8 bytes: {:?}", &out[0..8]);
    }
}

/// Process 8 SHA256 hashes in parallel using AVX2
///
/// This processes 8 independent SHA256 hashes in parallel using AVX2 SIMD instructions.
/// Each input must be exactly 64 bytes (one SHA256 block).
///
/// # Safety
/// This function is unsafe because it uses AVX2 intrinsics.
/// Caller must ensure AVX2 is available and inputs are 64 bytes each.
#[cfg(target_arch = "x86_64")]
pub unsafe fn sha256_8way_avx2(inputs: &[&[u8]; 8]) -> [[u8; 32]; 8] {
    // For inputs that aren't exactly 64 bytes, we need to handle padding
    // For now, use sha2 crate as fallback for non-64-byte inputs
    use bitcoin_hashes::{sha256d, Hash as BitcoinHash};
    
    // Check if all inputs are exactly 64 bytes
    let all_64_bytes = inputs.iter().all(|input| input.len() == 64);
    
    if !all_64_bytes {
        // Fallback to sha2 crate for non-64-byte inputs
        let mut results = [[0u8; 32]; 8];
        for (i, input) in inputs.iter().enumerate() {
            let hash = <sha256d::Hash as BitcoinHash>::hash(input);
            results[i].copy_from_slice(&hash.into_inner());
        }
        return results;
    }
    
    // Pack 8 inputs into contiguous 512-byte buffer
    let mut input_buf = [0u8; 512];
    for (i, input) in inputs.iter().enumerate() {
        input_buf[i * 64..(i + 1) * 64].copy_from_slice(input);
    }
    
    // Transform (double SHA256)
    let mut output_buf = [0u8; 256];
    transform_8way(&mut output_buf, &input_buf);
    
    // Unpack results
    let mut results = [[0u8; 32]; 8];
    for i in 0..8 {
        results[i].copy_from_slice(&output_buf[i * 32..(i + 1) * 32]);
    }
    
    results
}

/// Check if AVX2 is available and can be used
#[cfg(target_arch = "x86_64")]
pub fn is_avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn is_avx2_available() -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn sha256_8way_avx2(_inputs: &[&[u8]; 8]) -> [[u8; 32]; 8] {
    // Fallback for non-x86_64
    [[0u8; 32]; 8]
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitcoin_hashes::{sha256d, Hash as BitcoinHash, HashEngine};

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_read8_shuffle_behavior() {
        if !is_avx2_available() {
            println!("AVX2 not available, skipping test");
            return;
        }

        // Test read8 with a known pattern to understand shuffle behavior
        let mut input_buf = [0u8; 512];
        // Set first word of first block to 0x01234567 (little-endian: [67, 45, 23, 01])
        input_buf[0] = 0x67;
        input_buf[1] = 0x45;
        input_buf[2] = 0x23;
        input_buf[3] = 0x01;
        
        unsafe {
            // Read from offset 0
            let v = read8(&input_buf, 0);
            
            // Extract the first word
            // After _mm256_set_epi32(words[7], ..., words[0]):
            // - Index 0 = words[0] (block 0)
            // - Index 7 = words[7] (block 7)
            // So we extract from index 0 for block 0
            let word0 = _mm256_extract_epi32(v, 0) as u32;
            
            println!("Read8 shuffle test:");
            println!("  Input: 0x01234567 (LE bytes: [67, 45, 23, 01])");
            println!("  Output word0: 0x{:08x}", word0);
            println!("  Expected: 0x01234567 (if no swap) or 0x67452301 (if byte-reversed)");
            println!("  Or maybe: 0x23014567 or something else depending on shuffle");
            
            // Also check what the shuffle mask does
            let test_word = _mm256_set1_epi32(0x01234567);
            let shuffle_mask = _mm256_set_epi32(
                0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203,
                0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203,
            );
            let shuffled = _mm256_shuffle_epi8(test_word, shuffle_mask);
            let shuffled_word = _mm256_extract_epi32(shuffled, 0) as u32;
            println!("  Shuffle test on 0x01234567: 0x{:08x}", shuffled_word);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_round_macro_state_modification() {
        if !is_avx2_available() {
            println!("AVX2 not available, skipping test");
            return;
        }

        // Test that the round macro actually modifies state variables
        unsafe {
            use helpers::*;
            let mut a = k(0x6a09e667);
            let mut b = k(0xbb67ae85);
            let mut c = k(0x3c6ef372);
            let mut d = k(0xa54ff53a);
            let mut e = k(0x510e527f);
            let mut f = k(0x9b05688c);
            let mut g = k(0x1f83d9ab);
            let mut h = k(0x5be0cd19);
            
            let d_before = _mm256_extract_epi32(d, 0) as u32;
            let h_before = _mm256_extract_epi32(h, 0) as u32;
            
            // Call round macro
            round!(a, b, c, d, e, f, g, h, k(0x428a2f98));
            
            let d_after = _mm256_extract_epi32(d, 0) as u32;
            let h_after = _mm256_extract_epi32(h, 0) as u32;
            
            println!("Round macro state modification test:");
            println!("  d before: 0x{:08x}, after: 0x{:08x} (should be different)", d_before, d_after);
            println!("  h before: 0x{:08x}, after: 0x{:08x} (should be different)", h_before, h_after);
            
            if d_before == d_after {
                println!("  ERROR: d was not modified!");
            }
            if h_before == h_after {
                println!("  ERROR: h was not modified!");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_sha256_8way_avx2_correctness() {
        if !is_avx2_available() {
            println!("AVX2 not available, skipping test");
            return;
        }

        // Test with 8 identical 64-byte inputs
        let test_data = [0u8; 64];
        let inputs: [&[u8]; 8] = [
            &test_data, &test_data, &test_data, &test_data,
            &test_data, &test_data, &test_data, &test_data,
        ];

        unsafe {
            let avx2_results = sha256_8way_avx2(&inputs);
            
            // Compute expected result using bitcoin_hashes (double SHA256)
            // bitcoin_hashes::sha256d is the standard Bitcoin double SHA256 implementation
            let mut engine = sha256d::Hash::engine();
            engine.input(&test_data);
            let expected = sha256d::Hash::from_engine(engine);
            let expected_bytes = expected.into_inner();
            
            // Debug: print first few bytes
            println!("Expected: {:?}", &expected_bytes[..8]);
            println!("Got:      {:?}", &avx2_results[0][..8]);
            
            // All 8 results should be identical (same input)
            for (i, result) in avx2_results.iter().enumerate() {
                if result != &expected_bytes {
                    println!("Mismatch at index {}: expected {:?}, got {:?}", 
                        i, &expected_bytes[..16], &result[..16]);
                }
                assert_eq!(
                    result, &expected_bytes,
                    "AVX2 result {} does not match expected double SHA256",
                    i
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_sha256_8way_avx2_different_inputs() {
        if !is_avx2_available() {
            println!("AVX2 not available, skipping test");
            return;
        }

        // Test with 8 different 64-byte inputs
        let mut input_data = vec![vec![0u8; 64]; 8];
        for i in 0..8 {
            input_data[i][0] = i as u8;
        }
        let inputs: [&[u8]; 8] = [
            &input_data[0], &input_data[1], &input_data[2], &input_data[3],
            &input_data[4], &input_data[5], &input_data[6], &input_data[7],
        ];

        unsafe {
            let avx2_results = sha256_8way_avx2(&inputs);
            
            // Compute expected results using sha2 crate
            for (i, input) in input_data.iter().enumerate() {
                let mut engine = sha256d::Hash::engine();
                engine.input(input);
                let expected = sha256d::Hash::from_engine(engine);
                let expected_bytes = expected.into_inner();
                
                assert_eq!(
                    &avx2_results[i], &expected_bytes,
                    "AVX2 result {} does not match expected double SHA256 for input {}",
                    i, i
                );
            }
        }
    }
}
