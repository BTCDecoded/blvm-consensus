//! Shared locktime validation logic for BIP65 (CLTV) and BIP112 (CSV)
//!
//! Provides common functions for locktime type detection, value encoding/decoding,
//! and validation that are shared between CLTV and CSV implementations.

use crate::constants::LOCKTIME_THRESHOLD;
use crate::types::*;

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
pub fn decode_locktime_value(bytes: &ByteString) -> Option<u32> {
    if bytes.len() > 5 {
        return None; // Invalid encoding (too large)
    }

    let mut value: u32 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if i >= 4 {
            break; // Only use first 4 bytes
        }
        value |= (byte as u32) << (i * 8);
    }

    Some(value)
}

/// Encode locktime value to minimal-encoding byte string
///
/// Encodes a u32 locktime value to minimal little-endian encoding for script stack.
/// Used for script construction and testing.
pub fn encode_locktime_value(value: u32) -> ByteString {
    let mut bytes = Vec::new();

    // Minimal encoding: only include bytes up to the highest non-zero byte
    let mut temp = value;
    while temp > 0 {
        bytes.push((temp & 0xff) as u8);
        temp >>= 8;
    }

    // If value is 0, return single zero byte
    if bytes.is_empty() {
        bytes.push(0);
    }

    bytes
}

/// BIP68: Extract relative locktime type flag from sequence number
///
/// Bit 22 (0x00400000) indicates locktime type:
/// - 0 = block-based relative locktime
/// - 1 = time-based relative locktime
pub fn extract_sequence_type_flag(sequence: u32) -> bool {
    (sequence & 0x00400000) != 0
}

/// BIP68: Extract relative locktime value from sequence number
///
/// Masks out flags (bits 31, 22) and returns only the locktime value (bits 0-15).
pub fn extract_sequence_locktime_value(sequence: u32) -> u16 {
    (sequence & 0x0000ffff) as u16
}

/// BIP68: Check if sequence number has disabled bit set
///
/// Bit 31 (0x80000000) disables relative locktime when set.
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
        assert_eq!(extract_sequence_type_flag(0x00400000), true);
        assert_eq!(extract_sequence_type_flag(0x00000000), false);
        assert_eq!(extract_sequence_type_flag(0x00410000), true);
    }

    #[test]
    fn test_extract_sequence_locktime_value() {
        assert_eq!(extract_sequence_locktime_value(0x00001234), 0x1234);
        assert_eq!(extract_sequence_locktime_value(0x00401234), 0x1234); // Flags don't affect value
    }

    #[test]
    fn test_is_sequence_disabled() {
        assert_eq!(is_sequence_disabled(0x80000000), true);
        assert_eq!(is_sequence_disabled(0x00000000), false);
        assert_eq!(is_sequence_disabled(0x80010000), true);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: Locktime encoding round-trip correctness (BIP65/BIP112)
    ///
    /// Mathematical specification:
    /// ∀ value ∈ [0, 2^32):
    /// - decode_locktime_value(encode_locktime_value(value)) = value
    #[kani::proof]
    fn kani_locktime_encoding_round_trip() {
        let value: u32 = kani::any();

        let encoded = encode_locktime_value(value);
        let decoded = decode_locktime_value(&encoded);

        if decoded.is_some() {
            let decoded_value = decoded.unwrap();

            // Critical invariant: round-trip must preserve value
            assert_eq!(
                decoded_value, value,
                "Locktime encoding round-trip: decoded value must match original"
            );
        }
    }

    /// Kani proof: Locktime type determination correctness (BIP65/BIP112)
    ///
    /// Mathematical specification:
    /// ∀ locktime ∈ [0, 2^32):
    /// - get_locktime_type(locktime) = BlockHeight if locktime < LOCKTIME_THRESHOLD
    /// - get_locktime_type(locktime) = Timestamp if locktime >= LOCKTIME_THRESHOLD
    #[kani::proof]
    fn kani_locktime_type_determination() {
        let locktime: u32 = kani::any();

        let locktime_type = get_locktime_type(locktime);

        // Critical invariant: type must match threshold
        if locktime < LOCKTIME_THRESHOLD {
            assert_eq!(
                locktime_type,
                LocktimeType::BlockHeight,
                "Locktime type determination: values < LOCKTIME_THRESHOLD must be BlockHeight"
            );
        } else {
            assert_eq!(
                locktime_type,
                LocktimeType::Timestamp,
                "Locktime type determination: values >= LOCKTIME_THRESHOLD must be Timestamp"
            );
        }
    }

    /// Kani proof: locktime_types_match correctness (BIP65/BIP112)
    ///
    /// Mathematical specification:
    /// ∀ locktime1, locktime2 ∈ [0, 2^32):
    /// - locktime_types_match(locktime1, locktime2) = true ⟺
    ///   get_locktime_type(locktime1) = get_locktime_type(locktime2)
    ///
    /// This ensures locktime type matching is correct for all value ranges.
    #[kani::proof]
    fn kani_locktime_types_match_correctness() {
        let locktime1: u32 = kani::any();
        let locktime2: u32 = kani::any();

        // Calculate according to specification
        let type1 = get_locktime_type(locktime1);
        let type2 = get_locktime_type(locktime2);
        let spec_match = type1 == type2;

        // Calculate using implementation
        let impl_match = locktime_types_match(locktime1, locktime2);

        // Critical invariant: implementation must match specification
        assert_eq!(impl_match, spec_match,
            "locktime_types_match must match specification: types match if and only if get_locktime_type values are equal");
    }

    /// Kani proof: extract_sequence_locktime_value correctness (BIP68)
    ///
    /// Mathematical specification:
    /// ∀ sequence ∈ u32:
    /// - extract_sequence_locktime_value(sequence) = sequence & 0x0000ffff
    /// - This masks out flags (bits 31, 22) and preserves only locktime value (bits 0-15)
    ///
    /// This ensures sequence locktime value extraction matches BIP68 specification exactly.
    #[kani::proof]
    fn kani_extract_sequence_locktime_value_correctness() {
        let sequence: u32 = kani::any();

        // Calculate according to BIP68 spec: mask bits 0-15 (0x0000ffff)
        let spec_value = (sequence & 0x0000ffff) as u16;

        // Calculate using implementation
        let impl_value = extract_sequence_locktime_value(sequence);

        // Critical invariant: implementation must match specification
        assert_eq!(impl_value, spec_value,
            "extract_sequence_locktime_value must match BIP68 specification: extract bits 0-15 (0x0000ffff)");

        // Critical invariant: extracted value must be <= 0xffff (u16 max)
        assert!(
            impl_value <= 0xffff,
            "extract_sequence_locktime_value: extracted value must be <= u16::MAX"
        );

        // Critical invariant: extraction preserves lower 16 bits
        assert_eq!(
            impl_value as u32,
            sequence & 0x0000ffff,
            "extract_sequence_locktime_value: must preserve lower 16 bits exactly"
        );
    }

    /// Kani proof: Locktime threshold boundary (BIP65/BIP112)
    ///
    /// Mathematical specification:
    /// - get_locktime_type(LOCKTIME_THRESHOLD - 1) = BlockHeight
    /// - get_locktime_type(LOCKTIME_THRESHOLD) = Timestamp
    #[kani::proof]
    fn kani_locktime_threshold_boundary() {
        // Test boundary value: LOCKTIME_THRESHOLD - 1
        let block_height_locktime = LOCKTIME_THRESHOLD - 1;
        assert_eq!(
            get_locktime_type(block_height_locktime),
            LocktimeType::BlockHeight,
            "Locktime threshold boundary: LOCKTIME_THRESHOLD - 1 must be BlockHeight"
        );

        // Test boundary value: LOCKTIME_THRESHOLD
        let timestamp_locktime = LOCKTIME_THRESHOLD;
        assert_eq!(
            get_locktime_type(timestamp_locktime),
            LocktimeType::Timestamp,
            "Locktime threshold boundary: LOCKTIME_THRESHOLD must be Timestamp"
        );
    }
}
