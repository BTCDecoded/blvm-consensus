//! Shared locktime validation logic for BIP65 (CLTV) and BIP112 (CSV)
//!
//! Provides common functions for locktime type detection, value encoding/decoding,
//! and validation that are shared between CLTV and CSV implementations.

use crate::constants::LOCKTIME_THRESHOLD;
use crate::types::*;
use blvm_spec_lock::spec_locked;

/// Locktime type (block height vs timestamp)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocktimeType {
    /// Block height locktime (< LOCKTIME_THRESHOLD)
    BlockHeight,
    /// Unix timestamp locktime (>= LOCKTIME_THRESHOLD)
    Timestamp,
}

/// Determine locktime type from value
///
/// BIP65/BIP68: If locktime < 500000000, it's block height; otherwise it's Unix timestamp.
#[spec_locked("5.4.7")]
#[inline]
pub fn get_locktime_type(locktime: u32) -> LocktimeType {
    if locktime < LOCKTIME_THRESHOLD {
        LocktimeType::BlockHeight
    } else {
        LocktimeType::Timestamp
    }
}

/// Check if two locktime values have matching types
///
/// Used by both BIP65 (CLTV) and BIP112 (CSV) to ensure type consistency.
#[inline]
#[spec_locked("5.4.7")]
pub fn locktime_types_match(locktime1: u32, locktime2: u32) -> bool {
    get_locktime_type(locktime1) == get_locktime_type(locktime2)
}

/// Decode locktime value from minimal-encoding byte string
///
/// Decodes a little-endian, minimal-encoding locktime value from script stack.
/// Used by both BIP65 (CLTV) and BIP112 (CSV) for stack value decoding.
///
/// # Arguments
/// * `bytes` - Byte string from stack (max 5 bytes)
///
/// # Returns
/// Decoded u32 value, or None if invalid encoding
#[spec_locked("5.4.7")]
pub fn decode_locktime_value(bytes: &ByteString) -> Option<u32> {
    if bytes.len() > 5 {
        return None; // Invalid encoding (too large)
    }

    // Runtime assertion: Byte string length must be <= 5
    debug_assert!(
        bytes.len() <= 5,
        "Locktime byte string length ({}) must be <= 5",
        bytes.len()
    );

    let mut value: u32 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if i >= 4 {
            break; // Only use first 4 bytes
        }

        // Runtime assertion: Index must be < 4
        debug_assert!(i < 4, "Byte index ({i}) must be < 4 for locktime decoding");

        // Runtime assertion: Shift amount must be valid (0-24, multiples of 8)
        let shift_amount = i * 8;
        debug_assert!(
            shift_amount < 32,
            "Shift amount ({shift_amount}) must be < 32 (i: {i})"
        );

        value |= (byte as u32) << shift_amount;
    }

    // value is u32, so it always fits in u32 - no assertion needed

    Some(value)
}

/// Encode locktime value to minimal-encoding byte string
///
/// Encodes a u32 locktime value to minimal little-endian encoding for script stack.
/// Used for script construction and testing.
#[spec_locked("5.4.7")]
pub fn encode_locktime_value(value: u32) -> ByteString {
    let mut bytes = Vec::new();

    // Minimal encoding: only include bytes up to the highest non-zero byte
    let mut temp = value;
    while temp > 0 {
        bytes.push((temp & 0xff) as u8);
        temp >>= 8;

        // Runtime assertion: Encoding loop must terminate (temp decreases each iteration)
        // This is guaranteed by right shift, but documents the invariant
        debug_assert!(
            temp < value || bytes.len() <= 4,
            "Locktime encoding loop must terminate (temp: {}, value: {}, bytes: {})",
            temp,
            value,
            bytes.len()
        );
    }

    // If value is 0, return single zero byte
    if bytes.is_empty() {
        bytes.push(0);
    }

    // Runtime assertion: Encoded length must be between 1 and 4 bytes (u32 max)
    let len = bytes.len();
    debug_assert!(
        !bytes.is_empty() && len <= 4,
        "Encoded locktime length ({len}) must be between 1 and 4 bytes"
    );

    bytes
}

/// BIP68: Extract relative locktime type flag from sequence number
///
/// Bit 22 (0x00400000) indicates locktime type:
/// - 0 = block-based relative locktime
/// - 1 = time-based relative locktime
#[inline]
#[spec_locked("5.5")]
pub fn extract_sequence_type_flag(sequence: u32) -> bool {
    (sequence & 0x00400000) != 0
}

/// BIP68: Extract relative locktime value from sequence number
///
/// Masks out flags (bits 31, 22) and returns only the locktime value (bits 0-15).
#[inline]
#[spec_locked("5.5")]
pub fn extract_sequence_locktime_value(sequence: u32) -> u16 {
    (sequence & 0x0000ffff) as u16
}

/// BIP68: Check if sequence number has disabled bit set
///
/// Bit 31 (0x80000000) disables relative locktime when set.
#[inline]
#[spec_locked("5.5")]
pub fn is_sequence_disabled(sequence: u32) -> bool {
    (sequence & 0x80000000) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_locktime_type_block_height() {
        assert_eq!(get_locktime_type(100), LocktimeType::BlockHeight);
        assert_eq!(
            get_locktime_type(LOCKTIME_THRESHOLD - 1),
            LocktimeType::BlockHeight
        );
    }

    #[test]
    fn test_get_locktime_type_timestamp() {
        assert_eq!(
            get_locktime_type(LOCKTIME_THRESHOLD),
            LocktimeType::Timestamp
        );
        assert_eq!(get_locktime_type(1_000_000_000), LocktimeType::Timestamp);
    }

    #[test]
    fn test_locktime_types_match() {
        assert!(locktime_types_match(100, 200));
        assert!(locktime_types_match(
            LOCKTIME_THRESHOLD,
            LOCKTIME_THRESHOLD + 1000
        ));
        assert!(!locktime_types_match(100, LOCKTIME_THRESHOLD));
    }

    #[test]
    fn test_decode_locktime_value() {
        assert_eq!(decode_locktime_value(&vec![100, 0, 0, 0]), Some(100));
        assert_eq!(decode_locktime_value(&vec![0]), Some(0));
        assert_eq!(
            decode_locktime_value(&vec![0xff, 0xff, 0xff, 0xff]),
            Some(0xffffffff)
        );
        assert_eq!(decode_locktime_value(&vec![0; 6]), None); // Too large
    }

    #[test]
    fn test_encode_locktime_value() {
        // Minimal encoding: only include bytes up to highest non-zero byte
        assert_eq!(encode_locktime_value(100), vec![100]); // 0x64 fits in one byte
        assert_eq!(encode_locktime_value(0), vec![0]);
        assert_eq!(
            encode_locktime_value(0x12345678),
            vec![0x78, 0x56, 0x34, 0x12]
        );
        // Test multi-byte values
        assert_eq!(encode_locktime_value(0x00001234), vec![0x34, 0x12]);
        assert_eq!(
            encode_locktime_value(0x12345600),
            vec![0x00, 0x56, 0x34, 0x12]
        );
    }

    #[test]
    fn test_extract_sequence_type_flag() {
        assert!(extract_sequence_type_flag(0x00400000));
        assert!(!extract_sequence_type_flag(0x00000000));
        assert!(extract_sequence_type_flag(0x00410000));
    }

    #[test]
    fn test_extract_sequence_locktime_value() {
        assert_eq!(extract_sequence_locktime_value(0x00001234), 0x1234);
        assert_eq!(extract_sequence_locktime_value(0x00401234), 0x1234); // Flags don't affect value
    }

    #[test]
    fn test_is_sequence_disabled() {
        assert!(is_sequence_disabled(0x80000000));
        assert!(!is_sequence_disabled(0x00000000));
        assert!(is_sequence_disabled(0x80010000));
    }
}

