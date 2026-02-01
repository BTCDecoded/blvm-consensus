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

use blvm_spec_lock::spec_locked;

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
/// use blvm_consensus::bip113::get_median_time_past;
/// use blvm_consensus::types::BlockHeader;
///
/// let headers = vec![
///     BlockHeader { version: 1, prev_block_hash: [0; 32], merkle_root: [0; 32], timestamp: 1000, bits: 0, nonce: 0 },
///     BlockHeader { version: 1, prev_block_hash: [0; 32], merkle_root: [0; 32], timestamp: 2000, bits: 0, nonce: 0 },
///     BlockHeader { version: 1, prev_block_hash: [0; 32], merkle_root: [0; 32], timestamp: 3000, bits: 0, nonce: 0 },
/// ];
/// let median = get_median_time_past(&headers);
/// ```
#[spec_locked("5.5")]
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
#[spec_locked("5.5")]
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

