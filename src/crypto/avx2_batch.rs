//! AVX2-optimized batch SHA256 hashing
//!
//! This module provides 8-way parallel SHA256 hashing using AVX2 SIMD instructions.
//! This matches Bitcoin Core's approach for batch hashing operations.
//!
//! # Performance
//! - Single hash: Uses sha2 crate with asm (already optimized)
//! - Batch (8+ items): Uses AVX2 8-way parallel processing
//! - Expected: 4-8x speedup for batch operations on AVX2-capable CPUs

use sha2::{Digest, Sha256};
use crate::crypto::sha256_avx2;

/// Batch SHA256 using AVX2 8-way parallel processing
///
/// This processes 8 hashes in parallel using AVX2 SIMD instructions.
/// For inputs that aren't multiples of 8, the remainder is processed sequentially.
///
/// # Arguments
/// * `inputs` - Slice of byte slices to hash
///
/// # Returns
/// Vector of 32-byte hashes, one per input (in same order)
///
/// # Performance
/// - 8-way parallel processing for batches of 8 or more
/// - Falls back to sequential for smaller batches or non-AVX2 CPUs
#[cfg(target_arch = "x86_64")]
pub fn batch_sha256_avx2(inputs: &[&[u8]]) -> Vec<[u8; 32]> {
    if inputs.is_empty() {
        return Vec::new();
    }

    // Check if AVX2 is available
    if !is_x86_feature_detected!("avx2") {
        // Fallback to sequential processing
        return inputs
            .iter()
            .map(|input| {
                let hash = Sha256::digest(input);
                let mut result = [0u8; 32];
                result.copy_from_slice(&hash);
                result
            })
            .collect();
    }

    let mut results = Vec::with_capacity(inputs.len());
    
    // Process in chunks of 8 using AVX2
    let chunks = inputs.chunks_exact(8);
    let remainder = chunks.remainder();

    // Process full chunks of 8 in parallel using AVX2
    for chunk in chunks {
        // Convert chunk to array of 8 references
        let chunk_array: [&[u8]; 8] = [
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ];
        
        // Use AVX2 8-way parallel double SHA256
        // Note: sha256_8way_avx2 expects 64-byte inputs (one SHA256 block)
        // For non-64-byte inputs, it will fall back to sequential processing
        unsafe {
            let avx2_results = sha256_avx2::sha256_8way_avx2(&chunk_array);
            results.extend_from_slice(&avx2_results);
        }
    }

    // Process remainder sequentially
    for input in remainder {
        let hash = Sha256::digest(input);
        let mut result = [0u8; 32];
        result.copy_from_slice(&hash);
        results.push(result);
    }

    results
}

/// Fallback for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
pub fn batch_sha256_avx2(inputs: &[&[u8]]) -> Vec<[u8; 32]> {
    // Sequential processing for non-x86_64
    inputs
        .iter()
        .map(|input| {
            let hash = Sha256::digest(input);
            let mut result = [0u8; 32];
            result.copy_from_slice(&hash);
            result
        })
        .collect()
}

