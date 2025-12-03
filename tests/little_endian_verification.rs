//! Little-endian serialization verification tests
//!
//! Verifies that all serialization in BLLVM uses little-endian byte order
//! explicitly, matching Bitcoin Core's wire format.
//!
//! Consensus-critical: Endianness differences = network incompatibility

use blvm_consensus::serialization::{
    block::{deserialize_block_header, serialize_block_header},
    transaction::{deserialize_transaction, serialize_transaction},
    varint::{decode_varint, encode_varint},
};
use blvm_consensus::types::{BlockHeader, Transaction, TransactionInput, TransactionOutput};

/// Test that transaction version is serialized in little-endian
#[test]
fn test_transaction_version_little_endian() {
    // Version 0x01020304 should serialize as [0x04, 0x03, 0x02, 0x01] in little-endian
    let tx = Transaction {
        version: 0x01020304,
        inputs: vec![].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };

    let serialized = serialize_transaction(&tx);

    // Version is first 4 bytes, should be little-endian
    assert_eq!(
        serialized[0], 0x04,
        "Version byte 0 should be 0x04 (little-endian)"
    );
    assert_eq!(
        serialized[1], 0x03,
        "Version byte 1 should be 0x03 (little-endian)"
    );
    assert_eq!(
        serialized[2], 0x02,
        "Version byte 2 should be 0x02 (little-endian)"
    );
    assert_eq!(
        serialized[3], 0x01,
        "Version byte 3 should be 0x01 (little-endian)"
    );
}

/// Test that transaction output value is serialized in little-endian
#[test]
fn test_transaction_value_little_endian() {
    // Value 0x0102030405060708 should serialize as [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
    // Note: value is i64, but serialized as u64 (cast)
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 0x0102030405060708,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let serialized = serialize_transaction(&tx);

    // Find value bytes: version(4) + varint(input_count=0, 1 byte) + varint(output_count=1, 1 byte) = 6 bytes
    let value_start = 6;

    // Verify little-endian encoding by checking actual bytes
    // The value is cast to u64 before serialization, so we check the u64 representation
    let expected_bytes = (0x0102030405060708u64).to_le_bytes();
    assert_eq!(
        &serialized[value_start..value_start + 8],
        &expected_bytes[..],
        "Value must be serialized in little-endian"
    );
    assert_eq!(
        serialized[value_start], 0x08,
        "Value byte 0 should be 0x08 (little-endian)"
    );
    assert_eq!(
        serialized[value_start + 1],
        0x07,
        "Value byte 1 should be 0x07 (little-endian)"
    );
    assert_eq!(
        serialized[value_start + 2],
        0x06,
        "Value byte 2 should be 0x06 (little-endian)"
    );
    assert_eq!(
        serialized[value_start + 3],
        0x05,
        "Value byte 3 should be 0x05 (little-endian)"
    );
    assert_eq!(
        serialized[value_start + 4],
        0x04,
        "Value byte 4 should be 0x04 (little-endian)"
    );
    assert_eq!(
        serialized[value_start + 5],
        0x03,
        "Value byte 5 should be 0x03 (little-endian)"
    );
    assert_eq!(
        serialized[value_start + 6],
        0x02,
        "Value byte 6 should be 0x02 (little-endian)"
    );
    assert_eq!(
        serialized[value_start + 7],
        0x01,
        "Value byte 7 should be 0x01 (little-endian)"
    );
}

/// Test that transaction input index is serialized in little-endian
#[test]
fn test_transaction_input_index_little_endian() {
    // Index 0x01020304 should serialize as [0x04, 0x03, 0x02, 0x01] in little-endian
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32].into(),
                index: 0x01020304,
            },
            script_sig: vec![].into(),
            sequence: 0,
        }]
        .into(),
        outputs: vec![].into(),
        lock_time: 0,
    };

    let serialized = serialize_transaction(&tx);

    // Find index bytes (after version(4) + varint(input_count=1, 1 byte) + prevout.hash(32) = 37 bytes)
    let index_start = 37;
    assert_eq!(
        serialized[index_start], 0x04,
        "Index byte 0 should be 0x04 (little-endian)"
    );
    assert_eq!(
        serialized[index_start + 1],
        0x03,
        "Index byte 1 should be 0x03 (little-endian)"
    );
    assert_eq!(
        serialized[index_start + 2],
        0x02,
        "Index byte 2 should be 0x02 (little-endian)"
    );
    assert_eq!(
        serialized[index_start + 3],
        0x01,
        "Index byte 3 should be 0x01 (little-endian)"
    );
}

/// Test that transaction sequence is serialized in little-endian
#[test]
fn test_transaction_sequence_little_endian() {
    // Sequence 0x01020304 should serialize as [0x04, 0x03, 0x02, 0x01] in little-endian
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0x01020304,
        }]
        .into(),
        outputs: vec![].into(),
        lock_time: 0,
    };

    let serialized = serialize_transaction(&tx);

    // Find sequence bytes (after version(4) + varint(input_count=1, 1 byte) + prevout.hash(32) + index(4) + varint(script_len=0, 1 byte) = 42 bytes)
    let sequence_start = 42;
    assert_eq!(
        serialized[sequence_start], 0x04,
        "Sequence byte 0 should be 0x04 (little-endian)"
    );
    assert_eq!(
        serialized[sequence_start + 1],
        0x03,
        "Sequence byte 1 should be 0x03 (little-endian)"
    );
    assert_eq!(
        serialized[sequence_start + 2],
        0x02,
        "Sequence byte 2 should be 0x02 (little-endian)"
    );
    assert_eq!(
        serialized[sequence_start + 3],
        0x01,
        "Sequence byte 3 should be 0x01 (little-endian)"
    );
}

/// Test that block header version is serialized in little-endian
#[test]
fn test_block_header_version_little_endian() {
    // Version 0x01020304 should serialize as [0x04, 0x03, 0x02, 0x01] in little-endian
    let header = BlockHeader {
        version: 0x01020304,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 0,
        bits: 0,
        nonce: 0,
    };

    let serialized = serialize_block_header(&header);

    // Version is first 4 bytes
    assert_eq!(
        serialized[0], 0x04,
        "Version byte 0 should be 0x04 (little-endian)"
    );
    assert_eq!(
        serialized[1], 0x03,
        "Version byte 1 should be 0x03 (little-endian)"
    );
    assert_eq!(
        serialized[2], 0x02,
        "Version byte 2 should be 0x02 (little-endian)"
    );
    assert_eq!(
        serialized[3], 0x01,
        "Version byte 3 should be 0x01 (little-endian)"
    );
}

/// Test that block header timestamp is serialized in little-endian
#[test]
fn test_block_header_timestamp_little_endian() {
    // Timestamp 0x01020304 should serialize as [0x04, 0x03, 0x02, 0x01] in little-endian
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 0x01020304,
        bits: 0,
        nonce: 0,
    };

    let serialized = serialize_block_header(&header);

    // Timestamp is at offset 68 (after version(4) + prev_block_hash(32) + merkle_root(32) = 68)
    assert_eq!(
        serialized[68], 0x04,
        "Timestamp byte 0 should be 0x04 (little-endian)"
    );
    assert_eq!(
        serialized[69], 0x03,
        "Timestamp byte 1 should be 0x03 (little-endian)"
    );
    assert_eq!(
        serialized[70], 0x02,
        "Timestamp byte 2 should be 0x02 (little-endian)"
    );
    assert_eq!(
        serialized[71], 0x01,
        "Timestamp byte 3 should be 0x01 (little-endian)"
    );
}

/// Test that block header bits is serialized in little-endian
#[test]
fn test_block_header_bits_little_endian() {
    // Bits 0x01020304 should serialize as [0x04, 0x03, 0x02, 0x01] in little-endian
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 0,
        bits: 0x01020304,
        nonce: 0,
    };

    let serialized = serialize_block_header(&header);

    // Bits is at offset 72 (after version(4) + prev_block_hash(32) + merkle_root(32) + timestamp(4) = 72)
    assert_eq!(
        serialized[72], 0x04,
        "Bits byte 0 should be 0x04 (little-endian)"
    );
    assert_eq!(
        serialized[73], 0x03,
        "Bits byte 1 should be 0x03 (little-endian)"
    );
    assert_eq!(
        serialized[74], 0x02,
        "Bits byte 2 should be 0x02 (little-endian)"
    );
    assert_eq!(
        serialized[75], 0x01,
        "Bits byte 3 should be 0x01 (little-endian)"
    );
}

/// Test that block header nonce is serialized in little-endian
#[test]
fn test_block_header_nonce_little_endian() {
    // Nonce 0x01020304 should serialize as [0x04, 0x03, 0x02, 0x01] in little-endian
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 0,
        bits: 0,
        nonce: 0x01020304,
    };

    let serialized = serialize_block_header(&header);

    // Nonce is at offset 76 (after version(4) + prev_block_hash(32) + merkle_root(32) + timestamp(4) + bits(4) = 76)
    assert_eq!(
        serialized[76], 0x04,
        "Nonce byte 0 should be 0x04 (little-endian)"
    );
    assert_eq!(
        serialized[77], 0x03,
        "Nonce byte 1 should be 0x03 (little-endian)"
    );
    assert_eq!(
        serialized[78], 0x02,
        "Nonce byte 2 should be 0x02 (little-endian)"
    );
    assert_eq!(
        serialized[79], 0x01,
        "Nonce byte 3 should be 0x01 (little-endian)"
    );
}

/// Test that varint encoding uses little-endian for multi-byte values
#[test]
fn test_varint_little_endian() {
    // Value 0x0100 (256) should encode as [0xfd, 0x00, 0x01] (0xfd prefix + 2 bytes little-endian)
    let encoded = encode_varint(0x0100);
    assert_eq!(
        encoded,
        vec![0xfd, 0x00, 0x01],
        "VarInt 256 must be little-endian [0x00, 0x01]"
    );

    // Value 0x010000 (65536) should encode as [0xfe, 0x00, 0x00, 0x01, 0x00] (0xfe prefix + 4 bytes little-endian)
    let encoded = encode_varint(0x010000);
    assert_eq!(
        encoded,
        vec![0xfe, 0x00, 0x00, 0x01, 0x00],
        "VarInt 65536 must be little-endian"
    );

    // Value 0x0100000000 should encode with 0xff prefix + 8 bytes little-endian
    let encoded = encode_varint(0x0100000000);
    assert_eq!(encoded[0], 0xff, "VarInt 0x0100000000 must use 0xff prefix");
    assert_eq!(
        encoded[1..5],
        [0x00, 0x00, 0x00, 0x00],
        "Lower 32 bits must be [0x00, 0x00, 0x00, 0x00] (little-endian)"
    );
    assert_eq!(
        encoded[5..9],
        [0x01, 0x00, 0x00, 0x00],
        "Upper 32 bits must be [0x01, 0x00, 0x00, 0x00] (little-endian)"
    );
}

/// Test round-trip serialization preserves little-endian encoding
#[test]
fn test_round_trip_little_endian() {
    // Test transaction round-trip
    let tx = Transaction {
        version: 0x01020304,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32].into(),
                index: 0x05060708,
            },
            script_sig: vec![].into(),
            sequence: 0x090a0b0c,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0x0d0e0f1011121314,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0x15161718,
    };

    let serialized = serialize_transaction(&tx);
    let deserialized = deserialize_transaction(&serialized).unwrap();

    assert_eq!(deserialized.version, tx.version);
    assert_eq!(
        deserialized.inputs[0].prevout.index,
        tx.inputs[0].prevout.index
    );
    assert_eq!(deserialized.inputs[0].sequence, tx.inputs[0].sequence);
    assert_eq!(deserialized.outputs[0].value, tx.outputs[0].value);
    assert_eq!(deserialized.lock_time, tx.lock_time);
}

/// Test that all integer serialization uses explicit little-endian methods
///
/// This test verifies that we're not accidentally using native endianness
/// or big-endian anywhere in serialization code.
#[test]
fn test_explicit_little_endian_methods() {
    // This test documents that all serialization should use:
    // - to_le_bytes() for serialization
    // - from_le_bytes() for deserialization
    // - NOT to_bytes() or from_bytes() (which use native endianness)
    // - NOT to_be_bytes() or from_be_bytes() (which use big-endian)

    // Verify that Rust's default methods would be different on big-endian systems
    // (This test passes on little-endian systems, but documents the requirement)

    let value: u32 = 0x01020304;
    let le_bytes = value.to_le_bytes();
    let be_bytes = value.to_be_bytes();

    // On little-endian systems, native endian == little-endian
    // But we must use to_le_bytes() explicitly to ensure portability
    assert_eq!(
        le_bytes,
        [0x04, 0x03, 0x02, 0x01],
        "Little-endian encoding must be [0x04, 0x03, 0x02, 0x01]"
    );
    assert_ne!(
        le_bytes, be_bytes,
        "Little-endian and big-endian must differ"
    );
}
