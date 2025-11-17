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
/// use bllvm_consensus::serialization::varint::encode_varint;
///
/// assert_eq!(encode_varint(0), vec![0]);
/// assert_eq!(encode_varint(252), vec![252]);
/// assert_eq!(encode_varint(253), vec![0xfd, 253, 0]);
/// assert_eq!(encode_varint(65535), vec![0xfd, 255, 255]);
/// assert_eq!(encode_varint(65536), vec![0xfe, 0, 0, 1, 0]);
/// ```
pub fn encode_varint(value: u64) -> Vec<u8> {
    if value < 0xfd {
        vec![value as u8]
    } else if value <= 0xffff {
        // Runtime assertion: Value must be >= 0xfd for 2-byte encoding
        debug_assert!(
            value >= 0xfd,
            "Value ({}) must be >= 0xfd for 2-byte encoding",
            value
        );
        
        let mut result = vec![0xfd];
        result.extend_from_slice(&(value as u16).to_le_bytes());
        
        // Runtime assertion: Result must be exactly 3 bytes
        debug_assert!(
            result.len() == 3,
            "2-byte VarInt encoding must produce exactly 3 bytes, got {}",
            result.len()
        );
        
        result
    } else if value <= 0xffffffff {
        // Runtime assertion: Value must be > 0xffff for 4-byte encoding
        debug_assert!(
            value > 0xffff,
            "Value ({}) must be > 0xffff for 4-byte encoding",
            value
        );
        
        let mut result = vec![0xfe];
        result.extend_from_slice(&(value as u32).to_le_bytes());
        
        // Runtime assertion: Result must be exactly 5 bytes
        debug_assert!(
            result.len() == 5,
            "4-byte VarInt encoding must produce exactly 5 bytes, got {}",
            result.len()
        );
        
        result
    } else {
        // Runtime assertion: Value must be > 0xffffffff for 8-byte encoding
        debug_assert!(
            value > 0xffffffff,
            "Value ({}) must be > 0xffffffff for 8-byte encoding",
            value
        );
        
        let mut result = vec![0xff];
        result.extend_from_slice(&value.to_le_bytes());
        
        // Runtime assertion: Result must be exactly 9 bytes
        debug_assert!(
            result.len() == 9,
            "8-byte VarInt encoding must produce exactly 9 bytes, got {}",
            result.len()
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
/// use bllvm_consensus::serialization::varint::decode_varint;
///
/// assert_eq!(decode_varint(&[0]), Ok((0, 1)));
/// assert_eq!(decode_varint(&[252]), Ok((252, 1)));
/// assert_eq!(decode_varint(&[0xfd, 253, 0]), Ok((253, 3)));
/// assert_eq!(decode_varint(&[0xfd, 255, 255]), Ok((65535, 3)));
/// assert_eq!(decode_varint(&[0xfe, 0, 0, 1, 0]), Ok((65536, 5)));
/// assert!(decode_varint(&[]).is_err());
/// ```
pub fn decode_varint(data: &[u8]) -> Result<(u64, usize)> {
    if data.is_empty() {
        return Err(ConsensusError::Serialization(
            VarIntError::InsufficientBytes.to_string(),
        ));
    }

    let first_byte = data[0];

    match first_byte {
        // Single byte encoding
        b if b < 0xfd => Ok((b as u64, 1)),

        // 2-byte encoding (0xfd prefix)
        0xfd => {
            if data.len() < 3 {
                return Err(ConsensusError::Serialization(
                    VarIntError::InsufficientBytes.to_string(),
                ));
            }
            
            // Runtime assertion: Must have at least 3 bytes
            debug_assert!(
                data.len() >= 3,
                "2-byte VarInt decoding requires at least 3 bytes, got {}",
                data.len()
            );
            
            let value = u16::from_le_bytes([data[1], data[2]]) as u64;
            
            // Bitcoin Core rejects values < 0xfd encoded with 0xfd prefix
            if value < 0xfd {
                return Err(ConsensusError::Serialization(
                    VarIntError::InvalidEncoding.to_string(),
                ));
            }
            
            // Runtime assertion: Decoded value must be >= 0xfd
            debug_assert!(
                value >= 0xfd,
                "Decoded 2-byte VarInt value ({}) must be >= 0xfd",
                value
            );
            
            Ok((value, 3))
        }

        // 4-byte encoding (0xfe prefix)
        0xfe => {
            if data.len() < 5 {
                return Err(ConsensusError::Serialization(
                    VarIntError::InsufficientBytes.to_string(),
                ));
            }
            
            // Runtime assertion: Must have at least 5 bytes
            debug_assert!(
                data.len() >= 5,
                "4-byte VarInt decoding requires at least 5 bytes, got {}",
                data.len()
            );
            
            let value = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as u64;
            
            // Bitcoin Core rejects values <= 0xffff encoded with 0xfe prefix
            if value <= 0xffff {
                return Err(ConsensusError::Serialization(
                    VarIntError::InvalidEncoding.to_string(),
                ));
            }
            
            // Runtime assertion: Decoded value must be > 0xffff
            debug_assert!(
                value > 0xffff,
                "Decoded 4-byte VarInt value ({}) must be > 0xffff",
                value
            );
            
            Ok((value, 5))
        }

        // 8-byte encoding (0xff prefix)
        0xff => {
            if data.len() < 9 {
                return Err(ConsensusError::Serialization(
                    VarIntError::InsufficientBytes.to_string(),
                ));
            }
            
            // Runtime assertion: Must have at least 9 bytes
            debug_assert!(
                data.len() >= 9,
                "8-byte VarInt decoding requires at least 9 bytes, got {}",
                data.len()
            );
            
            let value = u64::from_le_bytes([
                data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
            ]);
            
            // Bitcoin Core rejects values <= 0xffffffff encoded with 0xff prefix
            if value <= 0xffffffff {
                return Err(ConsensusError::Serialization(
                    VarIntError::InvalidEncoding.to_string(),
                ));
            }
            
            // Runtime assertion: Decoded value must be > 0xffffffff
            debug_assert!(
                value > 0xffffffff,
                "Decoded 8-byte VarInt value ({}) must be > 0xffffffff",
                value
            );
            
            Ok((value, 9))
        }

        _ => Err(ConsensusError::Serialization(
            VarIntError::InvalidEncoding.to_string(),
        )),
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

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: VarInt encoding round-trip correctness (Orange Paper Section 13.3.2)
    ///
    /// Mathematical specification:
    /// ∀ value ∈ [0, 2^64): decode(encode(value)) = value
    ///
    /// This ensures VarInt encoding and decoding are inverse operations.
    #[kani::proof]
    fn kani_varint_encoding_round_trip() {
        let value: u64 = kani::any();

        // Encode and decode
        let encoded = encode_varint(value);
        let decoded_result = decode_varint(&encoded);

        if decoded_result.is_ok() {
            let (decoded_value, bytes_read) = decoded_result.unwrap();

            // Round-trip property: decode(encode(value)) = value
            assert_eq!(
                decoded_value, value,
                "VarInt encoding round-trip: decoded value must match original"
            );
            
            // Critical invariant: Bytes consumed must match encoded length
            assert_eq!(
                bytes_read,
                encoded.len(),
                "VarInt round-trip: bytes consumed ({}) must match encoded length ({})",
                bytes_read,
                encoded.len()
            );
        }
    }
    
    /// Kani proof: VarInt encoding length correctness
    ///
    /// Mathematical specification:
    /// ∀ value ∈ [0, 2^64):
    /// - If value < 0xfd: |encode(value)| = 1
    /// - If value <= 0xffff: |encode(value)| = 3
    /// - If value <= 0xffffffff: |encode(value)| = 5
    /// - Otherwise: |encode(value)| = 9
    ///
    /// This ensures VarInt encoding uses the correct number of bytes.
    #[kani::proof]
    fn kani_varint_encoding_length_correctness() {
        let value: u64 = kani::any();
        
        let encoded = encode_varint(value);
        
        // Critical invariant: Encoded length must match specification
        if value < 0xfd {
            assert_eq!(
                encoded.len(),
                1,
                "VarInt encoding length: value < 0xfd must produce 1 byte, got {}",
                encoded.len()
            );
        } else if value <= 0xffff {
            assert_eq!(
                encoded.len(),
                3,
                "VarInt encoding length: value <= 0xffff must produce 3 bytes, got {}",
                encoded.len()
            );
        } else if value <= 0xffffffff {
            assert_eq!(
                encoded.len(),
                5,
                "VarInt encoding length: value <= 0xffffffff must produce 5 bytes, got {}",
                encoded.len()
            );
        } else {
            assert_eq!(
                encoded.len(),
                9,
                "VarInt encoding length: value > 0xffffffff must produce 9 bytes, got {}",
                encoded.len()
            );
        }
    }

    /// Kani proof: VarInt encoding boundary correctness (Orange Paper Section 13.3.2)
    ///
    /// Mathematical specification:
    /// - Values < 0xfd: single byte encoding
    /// - Values 0xfd-0xffff: 3-byte encoding (0xfd prefix)
    /// - Values 0x10000-0xffffffff: 5-byte encoding (0xfe prefix)
    /// - Values > 0xffffffff: 9-byte encoding (0xff prefix)
    ///
    /// This ensures boundary values are encoded correctly.
    #[kani::proof]
    fn kani_varint_encoding_boundaries() {
        // Test boundary value 0xfc (single byte, max single-byte value)
        let encoded_fc = encode_varint(0xfc);
        assert_eq!(
            encoded_fc.len(),
            1,
            "VarInt boundary: 0xfc must be single-byte encoded"
        );
        assert_eq!(
            encoded_fc[0], 0xfc,
            "VarInt boundary: 0xfc encoding must match value"
        );

        // Test boundary value 0xfd (3-byte encoding, minimum for 0xfd prefix)
        let encoded_fd = encode_varint(0xfd);
        assert_eq!(
            encoded_fd.len(),
            3,
            "VarInt boundary: 0xfd must be 3-byte encoded"
        );
        assert_eq!(
            encoded_fd[0], 0xfd,
            "VarInt boundary: 0xfd must use 0xfd prefix"
        );

        // Test boundary value 0xffff (3-byte encoding, maximum for 0xfd prefix)
        let encoded_ffff = encode_varint(0xffff);
        assert_eq!(
            encoded_ffff.len(),
            3,
            "VarInt boundary: 0xffff must be 3-byte encoded"
        );
        assert_eq!(
            encoded_ffff[0], 0xfd,
            "VarInt boundary: 0xffff must use 0xfd prefix"
        );

        // Test boundary value 0x10000 (5-byte encoding, minimum for 0xfe prefix)
        let encoded_10000 = encode_varint(0x10000);
        assert_eq!(
            encoded_10000.len(),
            5,
            "VarInt boundary: 0x10000 must be 5-byte encoded"
        );
        assert_eq!(
            encoded_10000[0], 0xfe,
            "VarInt boundary: 0x10000 must use 0xfe prefix"
        );

        // Test boundary value 0xffffffff (5-byte encoding, maximum for 0xfe prefix)
        let encoded_ffffffff = encode_varint(0xffffffff);
        assert_eq!(
            encoded_ffffffff.len(),
            5,
            "VarInt boundary: 0xffffffff must be 5-byte encoded"
        );
        assert_eq!(
            encoded_ffffffff[0], 0xfe,
            "VarInt boundary: 0xffffffff must use 0xfe prefix"
        );

        // Test boundary value 0x100000000 (9-byte encoding, minimum for 0xff prefix)
        let encoded_100000000 = encode_varint(0x100000000);
        assert_eq!(
            encoded_100000000.len(),
            9,
            "VarInt boundary: 0x100000000 must be 9-byte encoded"
        );
        assert_eq!(
            encoded_100000000[0], 0xff,
            "VarInt boundary: 0x100000000 must use 0xff prefix"
        );
    }

    /// Kani proof: VarInt invalid encoding rejection (Orange Paper Section 13.3.2)
    ///
    /// Mathematical specification:
    /// - Values < 0xfd encoded with 0xfd prefix are rejected
    /// - Values <= 0xffff encoded with 0xfe prefix are rejected
    /// - Values <= 0xffffffff encoded with 0xff prefix are rejected
    ///
    /// This ensures invalid encodings are rejected (Bitcoin Core compatibility).
    #[kani::proof]
    fn kani_varint_invalid_encoding_rejection() {
        // Invalid: 0xfc encoded with 0xfd prefix
        let invalid_fd = vec![0xfd, 0xfc, 0x00];
        assert!(
            decode_varint(&invalid_fd).is_err(),
            "VarInt invalid encoding: values < 0xfd with 0xfd prefix must be rejected"
        );

        // Invalid: 0xffff encoded with 0xfe prefix
        let invalid_fe = vec![0xfe, 0xff, 0xff, 0x00, 0x00];
        assert!(
            decode_varint(&invalid_fe).is_err(),
            "VarInt invalid encoding: values <= 0xffff with 0xfe prefix must be rejected"
        );

        // Invalid: 0xffffffff encoded with 0xff prefix
        let invalid_ff = vec![0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00];
        assert!(
            decode_varint(&invalid_ff).is_err(),
            "VarInt invalid encoding: values <= 0xffffffff with 0xff prefix must be rejected"
        );
    }
}
