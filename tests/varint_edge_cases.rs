//! Varint encoding edge cases for consensus verification
//!
//! Tests for consensus-critical varint encoding that must match
//! Bitcoin Core's CompactSize encoding byte-for-byte.
//!
//! Consensus-critical: Varint encoding differences = network incompatibility

use bllvm_consensus::serialization::varint::{decode_varint, encode_varint};

/// Test all boundary values match Bitcoin Core's CompactSize encoding
///
/// Bitcoin Core uses CompactSize encoding:
/// - Values < 0xfd: single byte
/// - Values 0xfd-0xffff: 0xfd prefix + 2 bytes (little-endian)
/// - Values 0x10000-0xffffffff: 0xfe prefix + 4 bytes (little-endian)
/// - Values > 0xffffffff: 0xff prefix + 8 bytes (little-endian)
#[test]
fn test_varint_boundary_values() {
    // Boundary: 0xfc (last single-byte value)
    let encoded = encode_varint(0xfc);
    assert_eq!(encoded, vec![0xfc], "0xfc must be single-byte encoded");
    let (decoded, bytes) = decode_varint(&encoded).unwrap();
    assert_eq!(decoded, 0xfc);
    assert_eq!(bytes, 1);

    // Boundary: 0xfd (first 3-byte value)
    let encoded = encode_varint(0xfd);
    assert_eq!(encoded, vec![0xfd, 0xfd, 0x00], "0xfd must use 0xfd prefix");
    let (decoded, bytes) = decode_varint(&encoded).unwrap();
    assert_eq!(decoded, 0xfd);
    assert_eq!(bytes, 3);

    // Boundary: 0xffff (last 3-byte value)
    let encoded = encode_varint(0xffff);
    assert_eq!(
        encoded,
        vec![0xfd, 0xff, 0xff],
        "0xffff must use 0xfd prefix"
    );
    let (decoded, bytes) = decode_varint(&encoded).unwrap();
    assert_eq!(decoded, 0xffff);
    assert_eq!(bytes, 3);

    // Boundary: 0x10000 (first 5-byte value)
    let encoded = encode_varint(0x10000);
    assert_eq!(
        encoded,
        vec![0xfe, 0x00, 0x00, 0x01, 0x00],
        "0x10000 must use 0xfe prefix"
    );
    let (decoded, bytes) = decode_varint(&encoded).unwrap();
    assert_eq!(decoded, 0x10000);
    assert_eq!(bytes, 5);

    // Boundary: 0xffffffff (last 5-byte value)
    let encoded = encode_varint(0xffffffff);
    assert_eq!(
        encoded,
        vec![0xfe, 0xff, 0xff, 0xff, 0xff],
        "0xffffffff must use 0xfe prefix"
    );
    let (decoded, bytes) = decode_varint(&encoded).unwrap();
    assert_eq!(decoded, 0xffffffff);
    assert_eq!(bytes, 5);

    // Boundary: 0x100000000 (first 9-byte value)
    let encoded = encode_varint(0x100000000);
    assert_eq!(
        encoded,
        vec![0xff, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00],
        "0x100000000 must use 0xff prefix"
    );
    let (decoded, bytes) = decode_varint(&encoded).unwrap();
    assert_eq!(decoded, 0x100000000);
    assert_eq!(bytes, 9);
}

/// Test that invalid encodings are rejected (Bitcoin Core compatibility)
///
/// Bitcoin Core rejects:
/// - Values < 0xfd encoded with 0xfd prefix
/// - Values <= 0xffff encoded with 0xfe prefix
/// - Values <= 0xffffffff encoded with 0xff prefix
#[test]
fn test_varint_invalid_encoding_rejection() {
    // Invalid: 0xfc encoded with 0xfd prefix (should use single byte)
    assert!(
        decode_varint(&[0xfd, 0xfc, 0x00]).is_err(),
        "Values < 0xfd with 0xfd prefix must be rejected"
    );

    // Invalid: 0xffff encoded with 0xfe prefix (should use 0xfd)
    assert!(
        decode_varint(&[0xfe, 0xff, 0xff, 0x00, 0x00]).is_err(),
        "Values <= 0xffff with 0xfe prefix must be rejected"
    );

    // Invalid: 0xffffffff encoded with 0xff prefix (should use 0xfe)
    assert!(
        decode_varint(&[0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00]).is_err(),
        "Values <= 0xffffffff with 0xff prefix must be rejected"
    );
}

/// Test round-trip encoding for all boundary values
#[test]
fn test_varint_round_trip_boundaries() {
    let boundary_values = vec![
        0,
        0xfc,        // Last single-byte
        0xfd,        // First 3-byte
        0xffff,      // Last 3-byte
        0x10000,     // First 5-byte
        0xffffffff,  // Last 5-byte
        0x100000000, // First 9-byte
        u64::MAX,    // Maximum value
    ];

    for value in boundary_values {
        let encoded = encode_varint(value);
        let (decoded, bytes_consumed) = decode_varint(&encoded).unwrap();
        assert_eq!(
            decoded, value,
            "Round-trip failed for boundary value {value:#x}"
        );
        assert_eq!(
            bytes_consumed,
            encoded.len(),
            "Bytes consumed mismatch for {value:#x}"
        );
    }
}

/// Test varint encoding in transaction context
///
/// Verifies that varint encoding works correctly when used for
/// transaction input/output counts and script lengths.
#[test]
fn test_varint_transaction_context() {
    // Test input count encoding (typical values)
    let input_counts = vec![1, 2, 100, 250, 252, 253, 1000, 0xffff];
    for count in input_counts {
        let encoded = encode_varint(count);
        let (decoded, _) = decode_varint(&encoded).unwrap();
        assert_eq!(decoded, count, "Input count encoding failed for {count}");
    }

    // Test output count encoding
    let output_counts = vec![1, 2, 10, 252, 253, 1000];
    for count in output_counts {
        let encoded = encode_varint(count);
        let (decoded, _) = decode_varint(&encoded).unwrap();
        assert_eq!(decoded, count, "Output count encoding failed for {count}");
    }

    // Test script length encoding (can be large)
    let script_lengths = vec![0, 1, 100, 252, 253, 520, 1000, 0xffff, 0x10000];
    for len in script_lengths {
        let encoded = encode_varint(len);
        let (decoded, _) = decode_varint(&encoded).unwrap();
        assert_eq!(decoded, len, "Script length encoding failed for {len}");
    }
}

/// Test little-endian byte order correctness
///
/// Verifies that multi-byte values are encoded in little-endian order.
#[test]
fn test_varint_little_endian() {
    // Test 0x0100 (256) - should be [0x00, 0x01] in little-endian
    let encoded = encode_varint(256);
    assert_eq!(
        encoded,
        vec![0xfd, 0x00, 0x01],
        "256 must be little-endian [0x00, 0x01]"
    );

    // Test 0x010000 (65536) - should be [0x00, 0x00, 0x01, 0x00] in little-endian
    let encoded = encode_varint(0x10000);
    assert_eq!(
        encoded,
        vec![0xfe, 0x00, 0x00, 0x01, 0x00],
        "0x10000 must be little-endian"
    );

    // Test 0x0100000000 - should be little-endian
    let encoded = encode_varint(0x100000000);
    assert_eq!(
        encoded[1..5],
        [0x00, 0x00, 0x00, 0x00],
        "Lower 32 bits of 0x100000000 must be [0x00, 0x00, 0x00, 0x00]"
    );
    assert_eq!(
        encoded[5..9],
        [0x01, 0x00, 0x00, 0x00],
        "Upper 32 bits of 0x100000000 must be [0x01, 0x00, 0x00, 0x00]"
    );
}

/// Test insufficient bytes error handling
#[test]
fn test_varint_insufficient_bytes() {
    // Empty input
    assert!(decode_varint(&[]).is_err(), "Empty input must be rejected");

    // 0xfd prefix but only 1 byte
    assert!(
        decode_varint(&[0xfd]).is_err(),
        "0xfd prefix with 1 byte must be rejected"
    );

    // 0xfd prefix but only 2 bytes
    assert!(
        decode_varint(&[0xfd, 0x00]).is_err(),
        "0xfd prefix with 2 bytes must be rejected"
    );

    // 0xfe prefix but only 1 byte
    assert!(
        decode_varint(&[0xfe]).is_err(),
        "0xfe prefix with 1 byte must be rejected"
    );

    // 0xfe prefix but only 4 bytes
    assert!(
        decode_varint(&[0xfe, 0x00, 0x00, 0x00]).is_err(),
        "0xfe prefix with 4 bytes must be rejected"
    );

    // 0xff prefix but only 1 byte
    assert!(
        decode_varint(&[0xff]).is_err(),
        "0xff prefix with 1 byte must be rejected"
    );

    // 0xff prefix but only 8 bytes
    assert!(
        decode_varint(&[0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]).is_err(),
        "0xff prefix with 8 bytes must be rejected"
    );
}
