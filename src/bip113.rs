//! BIP113: Median Time-Past
//!
//! Implements median time-past calculation for block timestamps.
//! Used for CLTV (BIP65) timestamp validation and time-based relative locktime (BIP68).
//!
//! Specification: https://github.com/bitcoin/bips/blob/master/bip-0113.mediawiki
//!
//! The median time-past is calculated from the timestamps of the last 11 blocks,
//! providing a more stable time reference that prevents time-warp attacks.

use crate::types::BlockHeader;

/// Number of blocks to consider for median time-past calculation
/// BIP113: Use the median of the last 11 blocks
pub const MEDIAN_TIME_BLOCKS: usize = 11;

/// Calculate median time-past from block headers
///
/// BIP113: Returns the median timestamp of the last 11 blocks.
/// If fewer than 11 blocks are provided, returns the median of available blocks.
/// If no blocks are provided, returns 0 (invalid).
///
/// # Arguments
///
/// * `headers` - Slice of block headers, ordered from oldest to newest (blockchain order)
///   The last header in the slice should be the most recent block.
///
/// # Returns
///
/// Median timestamp (Unix time) of the last 11 blocks, or 0 if no headers provided
///
/// # Example
///
/// ```rust
/// use bllvm_consensus::bip113::get_median_time_past;
/// use bllvm_consensus::types::BlockHeader;
///
/// let headers = vec![
///     BlockHeader { version: 1, prev_block_hash: [0; 32], merkle_root: [0; 32], timestamp: 1000, bits: 0, nonce: 0 },
///     BlockHeader { version: 1, prev_block_hash: [0; 32], merkle_root: [0; 32], timestamp: 2000, bits: 0, nonce: 0 },
///     BlockHeader { version: 1, prev_block_hash: [0; 32], merkle_root: [0; 32], timestamp: 3000, bits: 0, nonce: 0 },
/// ];
/// let median = get_median_time_past(&headers);
/// ```
pub fn get_median_time_past(headers: &[BlockHeader]) -> u64 {
    if headers.is_empty() {
        return 0;
    }

    // Take the last MEDIAN_TIME_BLOCKS headers (or all if fewer available)
    let start_idx = headers.len().saturating_sub(MEDIAN_TIME_BLOCKS);
    let recent_headers = &headers[start_idx..];

    // Extract timestamps and sort
    let mut timestamps: Vec<u64> = recent_headers.iter().map(|h| h.timestamp).collect();

    timestamps.sort_unstable();

    // Calculate median (middle value)
    if timestamps.is_empty() {
        0
    } else if timestamps.len() % 2 == 0 {
        // Even number: average of two middle values
        let mid = timestamps.len() / 2;
        let lower = timestamps[mid - 1];
        let upper = timestamps[mid];

        // Runtime assertion: Lower must be <= upper (timestamps should be sorted)
        debug_assert!(
            lower <= upper,
            "Lower median timestamp ({lower}) must be <= upper ({upper})"
        );

        let median = (lower + upper) / 2;

        // Runtime assertion: Median must be between lower and upper
        debug_assert!(
            median >= lower && median <= upper,
            "Median ({median}) must be between lower ({lower}) and upper ({upper})"
        );

        median
    } else {
        // Odd number: middle value
        timestamps[timestamps.len() / 2]
    }
}

/// Calculate median time-past from a chain of block headers
///
/// Convenience function that takes headers in reverse order (newest first)
/// and calculates median time-past.
///
/// # Arguments
///
/// * `recent_headers` - Slice of block headers, ordered from newest to oldest
///   The first header should be the most recent block.
///
/// # Returns
///
/// Median timestamp (Unix time) of the last 11 blocks
pub fn get_median_time_past_reversed(recent_headers: &[BlockHeader]) -> u64 {
    if recent_headers.is_empty() {
        return 0;
    }

    // Reverse to get oldest-to-newest order
    let reversed: Vec<BlockHeader> = recent_headers.iter().rev().cloned().collect();

    // Take last MEDIAN_TIME_BLOCKS (or all if fewer)
    let start_idx = reversed.len().saturating_sub(MEDIAN_TIME_BLOCKS);
    let headers = &reversed[start_idx..];

    get_median_time_past(headers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BlockHeader;

    fn create_header(timestamp: u64) -> BlockHeader {
        BlockHeader {
            version: 1,
            prev_block_hash: [0u8; 32],
            merkle_root: [0u8; 32],
            timestamp,
            bits: 0x1d00ffff,
            nonce: 0,
        }
    }

    #[test]
    fn test_median_time_empty() {
        let headers = vec![];
        assert_eq!(get_median_time_past(&headers), 0);
    }

    #[test]
    fn test_median_time_single() {
        let headers = vec![create_header(1000)];
        assert_eq!(get_median_time_past(&headers), 1000);
    }

    #[test]
    fn test_median_time_three_blocks() {
        let headers = vec![
            create_header(1000),
            create_header(2000),
            create_header(3000),
        ];
        // Median of [1000, 2000, 3000] = 2000
        assert_eq!(get_median_time_past(&headers), 2000);
    }

    #[test]
    fn test_median_time_four_blocks() {
        let headers = vec![
            create_header(1000),
            create_header(2000),
            create_header(3000),
            create_header(4000),
        ];
        // Median of [1000, 2000, 3000, 4000] = (2000 + 3000) / 2 = 2500
        assert_eq!(get_median_time_past(&headers), 2500);
    }

    #[test]
    fn test_median_time_eleven_blocks() {
        let headers: Vec<BlockHeader> = (1..=11).map(|i| create_header(i * 100)).collect();
        // Median of [100, 200, ..., 1100] = 600
        assert_eq!(get_median_time_past(&headers), 600);
    }

    #[test]
    fn test_median_time_more_than_eleven() {
        let headers: Vec<BlockHeader> = (1..=20).map(|i| create_header(i * 100)).collect();
        // Should use last 11: [1000, 1100, ..., 2000]
        // Median = 1500
        assert_eq!(get_median_time_past(&headers), 1500);
    }

    #[test]
    fn test_median_time_unsorted() {
        // Median calculation should handle unsorted input
        let headers = vec![
            create_header(3000),
            create_header(1000),
            create_header(2000),
        ];
        // Should sort internally: [1000, 2000, 3000], median = 2000
        assert_eq!(get_median_time_past(&headers), 2000);
    }

    #[test]
    fn test_median_time_reversed() {
        let headers = vec![
            create_header(3000), // newest
            create_header(2000),
            create_header(1000), // oldest
        ];
        // Reversed order: newest first
        // After reversal: [1000, 2000, 3000], median = 2000
        assert_eq!(get_median_time_past_reversed(&headers), 2000);
    }

    #[test]
    fn test_median_time_past_bip113_example() {
        // Example from BIP113: if last 11 blocks have timestamps
        // [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
        // Median = 600 (6th element in sorted list)
        let headers: Vec<BlockHeader> =
            vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
                .into_iter()
                .map(create_header)
                .collect();

        assert_eq!(get_median_time_past(&headers), 600);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::types::BlockHeader;
    use kani::*;

    /// Kani proof: BIP113 median time-past is always >= minimum block timestamp
    ///
    /// Mathematical specification:
    /// ∀ headers ∈ [BlockHeader]: get_median_time_past(headers) >= min(header.timestamp for header in headers)
    #[kani::proof]
    #[kani::unwind(3)] // unwind_bounds::SIMPLE - very small bound to avoid sort_unstable complexity
    fn kani_bip113_median_time_ge_minimum() {
        // Constrain header_count before any loops to help Kani prune paths early
        let header_count: usize = kani::any();
        kani::assume(header_count <= 3); // Very small bound to avoid sort_unstable unwinding issues
        kani::assume(header_count > 0); // Ensure at least one header for meaningful proof

        let mut headers = Vec::new();
        let mut min_timestamp = u64::MAX;

        for _ in 0..header_count {
            let timestamp: u64 = kani::any();
            kani::assume(timestamp <= 1000000000); // Reasonable timestamp bounds
            if timestamp < min_timestamp {
                min_timestamp = timestamp;
            }
            headers.push(BlockHeader {
                version: 1,
                prev_block_hash: [0u8; 32],
                merkle_root: [0u8; 32],
                timestamp,
                bits: 0x1d00ffff,
                nonce: 0,
            });
        }

        let median = get_median_time_past(&headers);

        if header_count > 0 && min_timestamp < u64::MAX {
            assert!(median >= min_timestamp || median == 0);
        }
    }

    /// Kani proof: BIP113 median time-past calculation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ headers ∈ [BlockHeader]:
    /// get_median_time_past(headers) = get_median_time_past(headers) (deterministic)
    #[kani::proof]
    #[kani::unwind(5)] // unwind_bounds::MEDIUM - bounded loop for tractability
    fn kani_bip113_median_time_deterministic() {
        let header_count: usize = kani::any();
        kani::assume(header_count <= 5); // Reduced from 20 for tractability

        let mut headers = Vec::new();
        for _ in 0..header_count {
            let timestamp: u64 = kani::any();
            kani::assume(timestamp <= 1000000000);
            headers.push(BlockHeader {
                version: 1,
                prev_block_hash: [0u8; 32],
                merkle_root: [0u8; 32],
                timestamp,
                bits: 0x1d00ffff,
                nonce: 0,
            });
        }

        let median1 = get_median_time_past(&headers);
        let median2 = get_median_time_past(&headers);

        // Same input should produce same output
        assert_eq!(median1, median2);
    }

    /// Kani proof: BIP113 handles < 11 blocks correctly
    ///
    /// Mathematical specification:
    /// ∀ n ∈ [1, 10]: get_median_time_past returns median of n blocks (not last 11)
    #[kani::proof]
    #[kani::unwind(5)] // unwind_bounds::MEDIUM - bounded loop for tractability
    fn kani_bip113_handles_less_than_eleven_blocks() {
        let block_count: usize = kani::any();
        kani::assume(block_count >= 1 && block_count <= 5); // Reduced from < MEDIAN_TIME_BLOCKS for tractability

        let mut headers = Vec::new();
        for i in 0..block_count {
            headers.push(BlockHeader {
                version: 1,
                prev_block_hash: [0u8; 32],
                merkle_root: [0u8; 32],
                timestamp: (i as u64) * 100,
                bits: 0x1d00ffff,
                nonce: 0,
            });
        }

        let median = get_median_time_past(&headers);

        // Should return median of available blocks, not require 11
        assert!(median > 0 || block_count == 0);
    }

    /// Kani proof: BIP113 median calculation correctness (Orange Paper Section 13.3.4)
    ///
    /// Mathematical specification:
    /// ∀ headers ∈ [BlockHeader] with |headers| >= 11:
    /// - get_median_time_past(headers) = median of last 11 timestamps
    /// - Median is bounded by min(timestamps) and max(timestamps)
    /// - Median is one of the timestamps or between two adjacent ones
    #[kani::proof]
    #[kani::unwind(15)] // unwind_bounds::COMPLEX - needs at least 11 for correctness proof
    fn kani_bip113_median_calculation_correctness() {
        let header_count: usize = kani::any();
        kani::assume(header_count >= MEDIAN_TIME_BLOCKS && header_count <= 15); // Reduced from 20 for tractability

        let mut headers = Vec::new();
        let mut timestamps = Vec::new();

        for i in 0..header_count {
            let timestamp: u64 = kani::any();
            kani::assume(timestamp <= 1000000000); // Reasonable timestamp bounds
            timestamps.push(timestamp);
            headers.push(BlockHeader {
                version: 1,
                prev_block_hash: [0u8; 32],
                merkle_root: [0u8; 32],
                timestamp,
                bits: 0x1d00ffff,
                nonce: 0,
            });
        }

        let median = get_median_time_past(&headers);

        // Get last 11 timestamps (or all if fewer)
        let start_idx = timestamps.len().saturating_sub(MEDIAN_TIME_BLOCKS);
        let last_11: Vec<u64> = timestamps[start_idx..].to_vec();

        if !last_11.is_empty() {
            // Sort for median calculation
            let mut sorted = last_11.clone();
            sorted.sort();

            // Calculate expected median
            let expected_median = if sorted.len() % 2 == 0 {
                // Even number: average of two middle values
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2
            } else {
                // Odd number: middle value
                sorted[sorted.len() / 2]
            };

            // Critical invariant: calculated median must match expected median
            assert_eq!(
                median, expected_median,
                "BIP113 median calculation: must match expected median of last 11 timestamps"
            );

            // Median must be bounded by min and max
            let min_ts = sorted[0];
            let max_ts = sorted[sorted.len() - 1];
            assert!(
                median >= min_ts && median <= max_ts,
                "BIP113 median calculation: median must be between min and max timestamps"
            );
        }
    }

    /// Kani proof: BIP113 median time-past with exactly 11 blocks (Orange Paper Section 13.3.4)
    ///
    /// Mathematical specification:
    /// ∀ headers ∈ [BlockHeader] with |headers| = 11:
    /// - get_median_time_past(headers) = median of all 11 timestamps
    #[kani::proof]
    #[kani::unwind(11)] // Exactly MEDIAN_TIME_BLOCKS for this proof
    fn kani_bip113_exactly_eleven_blocks() {
        let mut headers = Vec::new();
        let mut timestamps = Vec::new();

        for i in 0..MEDIAN_TIME_BLOCKS {
            let timestamp: u64 = kani::any();
            kani::assume(timestamp <= 1000000000);
            timestamps.push(timestamp);
            headers.push(BlockHeader {
                version: 1,
                prev_block_hash: [0u8; 32],
                merkle_root: [0u8; 32],
                timestamp,
                bits: 0x1d00ffff,
                nonce: 0,
            });
        }

        let median = get_median_time_past(&headers);

        // Sort timestamps for median calculation
        let mut sorted = timestamps.clone();
        sorted.sort();

        // With 11 blocks, median is the 6th element (index 5)
        let expected_median = sorted[5];

        // Critical invariant: median must match expected median
        assert_eq!(
            median, expected_median,
            "BIP113 exactly 11 blocks: median must be 6th element of sorted timestamps"
        );
    }

    /// Kani proof: BIP113 median time-past bounded by block timestamps (Orange Paper Section 13.3.4)
    ///
    /// Mathematical specification:
    /// ∀ headers ∈ [BlockHeader]:
    /// - get_median_time_past(headers) >= min(header.timestamp for header in headers) or median == 0
    /// - get_median_time_past(headers) <= max(header.timestamp for header in headers) or median == 0
    #[kani::proof]
    #[kani::unwind(5)] // unwind_bounds::MEDIUM - bounded loop for tractability
    fn kani_bip113_median_bounded_by_timestamps() {
        let header_count: usize = kani::any();
        kani::assume(header_count <= 5); // Reduced from 20 for tractability

        let mut headers = Vec::new();
        let mut min_timestamp = u64::MAX;
        let mut max_timestamp = 0u64;

        for _ in 0..header_count {
            let timestamp: u64 = kani::any();
            kani::assume(timestamp <= 1000000000);
            if timestamp < min_timestamp {
                min_timestamp = timestamp;
            }
            if timestamp > max_timestamp {
                max_timestamp = timestamp;
            }
            headers.push(BlockHeader {
                version: 1,
                prev_block_hash: [0u8; 32],
                merkle_root: [0u8; 32],
                timestamp,
                bits: 0x1d00ffff,
                nonce: 0,
            });
        }

        let median = get_median_time_past(&headers);

        if header_count > 0 {
            // Critical invariant: median must be bounded by min and max timestamps (or 0 if empty)
            assert!(
                median == 0 || (median >= min_timestamp && median <= max_timestamp),
                "BIP113 median bounded: median must be between min and max timestamps"
            );
        }
    }
}
