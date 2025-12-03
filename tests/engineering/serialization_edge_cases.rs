//! Serialization/deserialization edge case tests
//! 
//! Tests for consensus-critical serialization edge cases that could cause
//! consensus divergence if not handled correctly.

use blvm_consensus::*;
use blvm_consensus::serialization::varint::{encode_varint, decode_varint};
use blvm_consensus::serialization::{serialize_transaction, deserialize_transaction, serialize_block_header, deserialize_block_header};

#[test]
fn test_varint_maximum_value() {
    // Maximum VarInt value: u64::MAX
    let max_value = u64::MAX;
    let encoded = encode_varint(max_value);
    let (decoded, bytes_consumed) = decode_varint(&encoded).unwrap();
    
    assert_eq!(decoded, max_value);
    assert_eq!(bytes_consumed, 9); // 0xff + 8 bytes
}

#[test]
fn test_varint_boundary_values() {
    // Test boundary values for encoding format switches
    let test_cases = vec![
        (0xfc, 1),      // Last single-byte value
        (0xfd, 3),      // First two-byte value
        (0xffff, 3),    // Last two-byte value
        (0x10000, 5),  // First four-byte value
        (0xffffffff, 5), // Last four-byte value
        (0x100000000, 9), // First eight-byte value
    ];
    
    for (value, expected_bytes) in test_cases {
        let encoded = encode_varint(value);
        assert_eq!(encoded.len(), expected_bytes, "Wrong encoding length for value {}", value);
        
        let (decoded, bytes_consumed) = decode_varint(&encoded).unwrap();
        assert_eq!(decoded, value);
        assert_eq!(bytes_consumed, expected_bytes);
    }
}

#[test]
fn test_varint_invalid_encoding_rejected() {
    // Value 252 encoded with 0xfd prefix should be rejected
    assert!(decode_varint(&[0xfd, 252, 0]).is_err());
    
    // Value 65535 encoded with 0xfe prefix should be rejected
    assert!(decode_varint(&[0xfe, 255, 255, 0, 0]).is_err());
    
    // Value 0xffffffff encoded with 0xff prefix should be rejected
    assert!(decode_varint(&[0xff, 255, 255, 255, 255, 0, 0, 0, 0]).is_err());
}

#[test]
fn test_varint_truncated_data() {
    // Empty input
    assert!(decode_varint(&[]).is_err());
    
    // Incomplete two-byte encoding
    assert!(decode_varint(&[0xfd]).is_err());
    assert!(decode_varint(&[0xfd, 0]).is_err());
    
    // Incomplete four-byte encoding
    assert!(decode_varint(&[0xfe]).is_err());
    assert!(decode_varint(&[0xfe, 0, 0, 0]).is_err());
    
    // Incomplete eight-byte encoding
    assert!(decode_varint(&[0xff]).is_err());
    assert!(decode_varint(&[0xff, 0, 0, 0, 0, 0, 0, 0]).is_err());
}

#[test]
fn test_transaction_serialization_round_trip() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51], // OP_1
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2; 32],
                    index: 1,
                },
                script_sig: vec![0x51, 0x52], // OP_1 OP_2
                sequence: 0xfffffffe,
            },
        ].into(),
        outputs: vec![
            TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![0x51].into(), // OP_1
            },
            TransactionOutput {
                value: 2500000000,
                script_pubkey: vec![0x51, 0x52], // OP_1 OP_2
            },
        ].into(),
        lock_time: 0,
    };
    
    let serialized = serialize_transaction(&tx);
    let deserialized = deserialize_transaction(&serialized).unwrap();
    
    assert_eq!(deserialized.version, tx.version);
    assert_eq!(deserialized.inputs.len(), tx.inputs.len());
    assert_eq!(deserialized.outputs.len(), tx.outputs.len());
    assert_eq!(deserialized.lock_time, tx.lock_time);
}

#[test]
fn test_transaction_serialization_empty_scripts() {
    // Test transaction with empty scripts
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: vec![],
            sequence: 0,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let serialized = serialize_transaction(&tx);
    let deserialized = deserialize_transaction(&serialized).unwrap();
    
    assert_eq!(deserialized.inputs[0].script_sig, vec![]);
    assert_eq!(deserialized.outputs[0].script_pubkey, vec![]);
}

#[test]
fn test_transaction_serialization_large_scripts() {
    // Test transaction with large scripts (near limits)
    let large_script = vec![0x51; 10000]; // 10KB script
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: large_script.clone(),
            sequence: 0,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: large_script.clone(),
        }].into(),
        lock_time: 0,
    };
    
    let serialized = serialize_transaction(&tx);
    let deserialized = deserialize_transaction(&serialized).unwrap();
    
    assert_eq!(deserialized.inputs[0].script_sig, large_script);
    assert_eq!(deserialized.outputs[0].script_pubkey, large_script);
}

#[test]
fn test_transaction_deserialize_insufficient_bytes() {
    // Empty input
    assert!(deserialize_transaction(&[]).is_err());
    
    // Only version
    assert!(deserialize_transaction(&[0, 0, 0, 0]).is_err());
    
    // Version + incomplete input count
    assert!(deserialize_transaction(&[0, 0, 0, 0, 0xfd]).is_err());
}

#[test]
fn test_block_header_serialization_size() {
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0,
    };
    
    let serialized = serialize_block_header(&header);
    assert_eq!(serialized.len(), 80, "Block header must be exactly 80 bytes");
}

#[test]
fn test_block_header_serialization_round_trip() {
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [1; 32],
        merkle_root: [2; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 12345,
    };
    
    let serialized = serialize_block_header(&header);
    let deserialized = deserialize_block_header(&serialized).unwrap();
    
    assert_eq!(deserialized.version, header.version);
    assert_eq!(deserialized.prev_block_hash, header.prev_block_hash);
    assert_eq!(deserialized.merkle_root, header.merkle_root);
    assert_eq!(deserialized.timestamp, header.timestamp);
    assert_eq!(deserialized.bits, header.bits);
    assert_eq!(deserialized.nonce, header.nonce);
}

#[test]
fn test_block_header_deserialize_insufficient_bytes() {
    // Empty input
    assert!(deserialize_block_header(&[]).is_err());
    
    // Partial header
    assert!(deserialize_block_header(&[0; 40]).is_err());
    assert!(deserialize_block_header(&[0; 79]).is_err());
}

#[test]
fn test_transaction_negative_version() {
    // Bitcoin allows negative transaction versions (though rare)
    let tx = Transaction {
        version: 0xffffffff, // -1 when interpreted as i32
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };
    
    // Serialization should handle this (stores as i32 in wire format)
    let serialized = serialize_transaction(&tx);
    let deserialized = deserialize_transaction(&serialized).unwrap();
    
    // Version should be preserved (as u64)
    assert_eq!(deserialized.version, tx.version);
}

