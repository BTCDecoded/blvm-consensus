//! Bitcoin VarInt encoding/decoding
//!
//! VarInt (Variable Integer) is a compact encoding for integers used throughout
//! Bitcoin's wire format. It uses 1-9 bytes depending on the value.
//!
//! Encoding rules:
//! - If value < 0xfd: single byte
//! - If value <= 0xffff: 0xfd prefix + 2 bytes (little-endian)
//! - If value <= 0xffffffff: 0xfe prefix + 4 bytes (little-endian)  
//! - Otherwise: 0xff prefix + 8 bytes (little-endian)
//!
//! This must match Bitcoin Core's CVarInt implementation exactly.

use crate::error::{ConsensusError, Result};
use blvm_spec_lock::spec_locked;
use std::borrow::Cow;

/// Error type for VarInt encoding/decoding failures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarIntError {
    /// Insufficient bytes to decode VarInt
    InsufficientBytes,
    /// Invalid VarInt encoding format
    InvalidEncoding,
    /// VarInt value exceeds maximum (u64::MAX)
    ValueTooLarge,
}

impl std::fmt::Display for VarIntError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VarIntError::InsufficientBytes => write!(f, "Insufficient bytes to decode VarInt"),
            VarIntError::InvalidEncoding => write!(f, "Invalid VarInt encoding"),
            VarIntError::ValueTooLarge => write!(f, "VarInt value too large"),
        }
    }
}

impl std::error::Error for VarIntError {}

/// Encode a u64 value as a Bitcoin VarInt
///
/// # Examples
///
/// ```
/// use blvm_consensus::serialization::varint::encode_varint;
///
/// assert_eq!(encode_varint(0), vec![0]);
/// assert_eq!(encode_varint(252), vec![252]);
/// assert_eq!(encode_varint(253), vec![0xfd, 253, 0]);
/// assert_eq!(encode_varint(65535), vec![0xfd, 255, 255]);
/// assert_eq!(encode_varint(65536), vec![0xfe, 0, 0, 1, 0]);
/// ```
#[spec_locked("3.2")]
pub fn encode_varint(value: u64) -> Vec<u8> {
    if value < 0xfd {
        vec![value as u8]
    } else if value <= 0xffff {
        // Runtime assertion: Value must be >= 0xfd for 2-byte encoding
        debug_assert!(
            value >= 0xfd,
            "Value ({value}) must be >= 0xfd for 2-byte encoding"
        );

        let mut result = vec![0xfd];
        result.extend_from_slice(&(value as u16).to_le_bytes());

        // Runtime assertion: Result must be exactly 3 bytes
        let len = result.len();
        debug_assert!(
            len == 3,
            "2-byte VarInt encoding must produce exactly 3 bytes, got {len}"
        );

        result
    } else if value <= 0xffffffff {
        // Runtime assertion: Value must be > 0xffff for 4-byte encoding
        debug_assert!(
            value > 0xffff,
            "Value ({value}) must be > 0xffff for 4-byte encoding"
        );

        let mut result = vec![0xfe];
        result.extend_from_slice(&(value as u32).to_le_bytes());

        // Runtime assertion: Result must be exactly 5 bytes
        let len = result.len();
        debug_assert!(
            len == 5,
            "4-byte VarInt encoding must produce exactly 5 bytes, got {len}"
        );

        result
    } else {
        // Runtime assertion: Value must be > 0xffffffff for 8-byte encoding
        debug_assert!(
            value > 0xffffffff,
            "Value ({value}) must be > 0xffffffff for 8-byte encoding"
        );

        let mut result = vec![0xff];
        result.extend_from_slice(&value.to_le_bytes());

        // Runtime assertion: Result must be exactly 9 bytes
        let len = result.len();
        debug_assert!(
            len == 9,
            "8-byte VarInt encoding must produce exactly 9 bytes, got {len}"
        );

        result
    }
}

/// Decode a Bitcoin VarInt from bytes
///
/// Returns the decoded value and the number of bytes consumed.
///
/// # Errors
///
/// Returns `VarIntError` if the input is malformed or insufficient.
///
/// # Examples
///
/// ```
/// use blvm_consensus::serialization::varint::decode_varint;
///
/// assert_eq!(decode_varint(&[0]), Ok((0, 1)));
/// assert_eq!(decode_varint(&[252]), Ok((252, 1)));
/// assert_eq!(decode_varint(&[0xfd, 253, 0]), Ok((253, 3)));
/// assert_eq!(decode_varint(&[0xfd, 255, 255]), Ok((65535, 3)));
/// assert_eq!(decode_varint(&[0xfe, 0, 0, 1, 0]), Ok((65536, 5)));
/// assert!(decode_varint(&[]).is_err());
/// ```
#[spec_locked("3.2")]
pub fn decode_varint(data: &[u8]) -> Result<(u64, usize)> {
    if data.is_empty() {
        return Err(ConsensusError::Serialization(Cow::Owned(
            VarIntError::InsufficientBytes.to_string(),
        )));
    }

    let first_byte = data[0];

    match first_byte {
        // Single byte encoding
        b if b < 0xfd => Ok((b as u64, 1)),

        // 2-byte encoding (0xfd prefix)
        0xfd => {
            if data.len() < 3 {
                return Err(ConsensusError::Serialization(Cow::Owned(
                    VarIntError::InsufficientBytes.to_string(),
                )));
            }

            // Runtime assertion: Must have at least 3 bytes
            let len = data.len();
            debug_assert!(
                len >= 3,
                "2-byte VarInt decoding requires at least 3 bytes, got {len}"
            );

            let value = u16::from_le_bytes([data[1], data[2]]) as u64;

            // Bitcoin Core rejects values < 0xfd encoded with 0xfd prefix
            if value < 0xfd {
                return Err(ConsensusError::Serialization(Cow::Owned(
                    VarIntError::InvalidEncoding.to_string(),
                )));
            }

            // Runtime assertion: Decoded value must be >= 0xfd
            debug_assert!(
                value >= 0xfd,
                "Decoded 2-byte VarInt value ({value}) must be >= 0xfd"
            );

            Ok((value, 3))
        }

        // 4-byte encoding (0xfe prefix)
        0xfe => {
            if data.len() < 5 {
                return Err(ConsensusError::Serialization(Cow::Owned(
                    VarIntError::InsufficientBytes.to_string(),
                )));
            }

            // Runtime assertion: Must have at least 5 bytes
            let len = data.len();
            debug_assert!(
                len >= 5,
                "4-byte VarInt decoding requires at least 5 bytes, got {len}"
            );

            let value = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as u64;

            // Bitcoin Core rejects values <= 0xffff encoded with 0xfe prefix
            if value <= 0xffff {
                return Err(ConsensusError::Serialization(Cow::Owned(
                    VarIntError::InvalidEncoding.to_string(),
                )));
            }

            // Runtime assertion: Decoded value must be > 0xffff
            debug_assert!(
                value > 0xffff,
                "Decoded 4-byte VarInt value ({value}) must be > 0xffff"
            );

            Ok((value, 5))
        }

        // 8-byte encoding (0xff prefix)
        0xff => {
            if data.len() < 9 {
                return Err(ConsensusError::Serialization(Cow::Owned(
                    VarIntError::InsufficientBytes.to_string(),
                )));
            }

            // Runtime assertion: Must have at least 9 bytes
            let len = data.len();
            debug_assert!(
                len >= 9,
                "8-byte VarInt decoding requires at least 9 bytes, got {len}"
            );

            let value = u64::from_le_bytes([
                data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
            ]);

            // Bitcoin Core rejects values <= 0xffffffff encoded with 0xff prefix
            if value <= 0xffffffff {
                return Err(ConsensusError::Serialization(Cow::Owned(
                    VarIntError::InvalidEncoding.to_string(),
                )));
            }

            // Runtime assertion: Decoded value must be > 0xffffffff
            debug_assert!(
                value > 0xffffffff,
                "Decoded 8-byte VarInt value ({value}) must be > 0xffffffff"
            );

            Ok((value, 9))
        }

        _ => Err(ConsensusError::Serialization(Cow::Owned(
            VarIntError::InvalidEncoding.to_string(),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_varint_small() {
        assert_eq!(encode_varint(0), vec![0]);
        assert_eq!(encode_varint(1), vec![1]);
        assert_eq!(encode_varint(252), vec![252]);
    }

    #[test]
    fn test_encode_varint_medium() {
        assert_eq!(encode_varint(253), vec![0xfd, 253, 0]);
        assert_eq!(encode_varint(255), vec![0xfd, 255, 0]);
        assert_eq!(encode_varint(256), vec![0xfd, 0, 1]);
        assert_eq!(encode_varint(65535), vec![0xfd, 255, 255]);
    }

    #[test]
    fn test_encode_varint_large() {
        assert_eq!(encode_varint(65536), vec![0xfe, 0, 0, 1, 0]);
        assert_eq!(encode_varint(65537), vec![0xfe, 1, 0, 1, 0]);
        assert_eq!(encode_varint(0xffffffff), vec![0xfe, 255, 255, 255, 255]);
    }

    #[test]
    fn test_encode_varint_huge() {
        assert_eq!(
            encode_varint(0x100000000),
            vec![0xff, 0, 0, 0, 0, 1, 0, 0, 0]
        );
        assert_eq!(
            encode_varint(u64::MAX),
            vec![0xff, 255, 255, 255, 255, 255, 255, 255, 255]
        );
    }

    #[test]
    fn test_decode_varint_small() {
        assert_eq!(decode_varint(&[0]), Ok((0, 1)));
        assert_eq!(decode_varint(&[1]), Ok((1, 1)));
        assert_eq!(decode_varint(&[252]), Ok((252, 1)));
    }

    #[test]
    fn test_decode_varint_medium() {
        assert_eq!(decode_varint(&[0xfd, 253, 0]), Ok((253, 3)));
        assert_eq!(decode_varint(&[0xfd, 255, 255]), Ok((65535, 3)));
    }

    #[test]
    fn test_decode_varint_large() {
        assert_eq!(decode_varint(&[0xfe, 0, 0, 1, 0]), Ok((65536, 5)));
        assert_eq!(
            decode_varint(&[0xfe, 255, 255, 255, 255]),
            Ok((0xffffffff, 5))
        );
    }

    #[test]
    fn test_decode_varint_huge() {
        assert_eq!(
            decode_varint(&[0xff, 0, 0, 0, 0, 1, 0, 0, 0]),
            Ok((0x100000000, 9))
        );
        assert_eq!(
            decode_varint(&[0xff, 255, 255, 255, 255, 255, 255, 255, 255]),
            Ok((u64::MAX, 9))
        );
    }

    #[test]
    fn test_decode_varint_insufficient_bytes() {
        assert!(decode_varint(&[]).is_err());
        assert!(decode_varint(&[0xfd]).is_err());
        assert!(decode_varint(&[0xfd, 0]).is_err());
        assert!(decode_varint(&[0xfe]).is_err());
        assert!(decode_varint(&[0xfe, 0, 0, 0]).is_err());
        assert!(decode_varint(&[0xff]).is_err());
        assert!(decode_varint(&[0xff, 0, 0, 0, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn test_decode_varint_invalid_encoding() {
        // Value 252 should use single byte, not 0xfd prefix
        assert!(decode_varint(&[0xfd, 252, 0]).is_err());
        // Value 65535 should use 0xfd, not 0xfe prefix
        assert!(decode_varint(&[0xfe, 255, 255, 0, 0]).is_err());
        // Value 0xffffffff should use 0xfe, not 0xff prefix
        assert!(decode_varint(&[0xff, 255, 255, 255, 255, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn test_round_trip_encoding() {
        // Test round-trip for all boundary values
        let test_values = vec![
            0,
            252,
            253,
            254,
            255,
            256,
            65534,
            65535,
            65536,
            65537,
            0xffffffff - 1,
            0xffffffff,
            0x100000000,
            0x100000001,
            u64::MAX / 2,
            u64::MAX,
        ];

        for value in test_values {
            let encoded = encode_varint(value);
            let (decoded, bytes_consumed) = decode_varint(&encoded).unwrap();
            assert_eq!(decoded, value, "Round-trip failed for value {value}");
            assert_eq!(
                bytes_consumed,
                encoded.len(),
                "Bytes consumed mismatch for value {value}"
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        // Edge case: 0xfc (last single-byte value)
        let encoded = encode_varint(0xfc);
        assert_eq!(encoded, vec![0xfc]);
        assert_eq!(decode_varint(&encoded), Ok((0xfc, 1)));

        // Edge case: 0xfd (first two-byte value)
        let encoded = encode_varint(0xfd);
        assert_eq!(encoded, vec![0xfd, 0xfd, 0]);
        assert_eq!(decode_varint(&encoded), Ok((0xfd, 3)));

        // Edge case: 0xffff (last two-byte value)
        let encoded = encode_varint(0xffff);
        assert_eq!(encoded, vec![0xfd, 255, 255]);
        assert_eq!(decode_varint(&encoded), Ok((0xffff, 3)));

        // Edge case: 0x10000 (first four-byte value)
        let encoded = encode_varint(0x10000);
        assert_eq!(encoded, vec![0xfe, 0, 0, 1, 0]);
        assert_eq!(decode_varint(&encoded), Ok((0x10000, 5)));

        // Edge case: 0xffffffff (last four-byte value)
        let encoded = encode_varint(0xffffffff);
        assert_eq!(encoded, vec![0xfe, 255, 255, 255, 255]);
        assert_eq!(decode_varint(&encoded), Ok((0xffffffff, 5)));

        // Edge case: 0x100000000 (first eight-byte value)
        let encoded = encode_varint(0x100000000);
        assert_eq!(encoded, vec![0xff, 0, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(decode_varint(&encoded), Ok((0x100000000, 9)));
    }
}

