//! Parser determinism edge case tests
//! 
//! Tests for consensus-critical parser behavior that must reject malformed data
//! deterministically to ensure all nodes agree on what's invalid.

use blvm_consensus::*;
use blvm_consensus::serialization::varint::decode_varint;
use blvm_consensus::serialization::{deserialize_transaction, deserialize_block_header};

#[test]
fn test_varint_truncated_data() {
    // EOF in middle of VarInt encoding
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
fn test_transaction_truncated_data() {
    // Empty input
    assert!(deserialize_transaction(&[]).is_err());
    
    // Only version (incomplete)
    assert!(deserialize_transaction(&[0, 0, 0]).is_err());
    
    // Version but no input count
    assert!(deserialize_transaction(&[0, 0, 0, 0]).is_err());
    
    // Version + incomplete VarInt for input count
    assert!(deserialize_transaction(&[0, 0, 0, 0, 0xfd]).is_err());
    assert!(deserialize_transaction(&[0, 0, 0, 0, 0xfd, 0]).is_err());
    
    // Version + input count (1) but no input data
    assert!(deserialize_transaction(&[0, 0, 0, 0, 1]).is_err());
    
    // Version + input count + partial hash
    let mut data = vec![0, 0, 0, 0, 1]; // Version + 1 input
    data.extend_from_slice(&[0; 31]); // Only 31 bytes of hash
    assert!(deserialize_transaction(&data).is_err());
}

#[test]
fn test_transaction_invalid_length_fields() {
    // Transaction with VarInt length > remaining bytes
    let mut data = vec![0, 0, 0, 0]; // Version
    data.push(1); // 1 input
    data.extend_from_slice(&[0; 32]); // Hash
    data.extend_from_slice(&[0, 0, 0, 0]); // Index
    data.push(0xff); // Script length VarInt prefix (8-byte encoding)
    data.extend_from_slice(&[255, 255, 255, 255, 255, 255, 255, 255]); // u64::MAX length
    // Not enough bytes for script
    assert!(deserialize_transaction(&data).is_err());
    
    // Script length = 0 (valid case - empty script)
    let mut data = vec![0, 0, 0, 0]; // Version
    data.push(1); // 1 input
    data.extend_from_slice(&[0; 32]); // Hash
    data.extend_from_slice(&[0, 0, 0, 0]); // Index
    data.push(0); // Script length = 0
    data.extend_from_slice(&[0, 0, 0, 0]); // Sequence
    data.push(1); // 1 output
    data.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // Value
    data.push(0); // Script length = 0
    data.extend_from_slice(&[0, 0, 0, 0]); // Lock time
    
    // This should succeed (empty scripts are valid)
    let result = deserialize_transaction(&data);
    assert!(result.is_ok());
}

#[test]
fn test_block_header_truncated_data() {
    // Empty input
    assert!(deserialize_block_header(&[]).is_err());
    
    // Partial header
    assert!(deserialize_block_header(&[0; 40]).is_err());
    assert!(deserialize_block_header(&[0; 79]).is_err());
    
    // Exactly 80 bytes should succeed
    assert!(deserialize_block_header(&[0; 80]).is_ok());
}

#[test]
fn test_transaction_negative_input_count() {
    // Transaction with VarInt that would decode to negative (impossible for unsigned)
    // But VarInt is unsigned, so we test with maximum value that would cause issues
    
    // Very large input count that would cause memory issues
    let mut data = vec![0, 0, 0, 0]; // Version
    data.push(0xff); // VarInt prefix for 8-byte encoding
    data.extend_from_slice(&[255, 255, 255, 255, 255, 255, 255, 255]); // u64::MAX
    
    // Should fail during deserialization (too many inputs or memory)
    let result = deserialize_transaction(&data);
    // May fail at various points - the important thing is it fails deterministically
    assert!(result.is_err());
}

#[test]
fn test_transaction_negative_output_count() {
    // Very large output count
    let mut data = vec![0, 0, 0, 0]; // Version
    data.push(0); // 0 inputs
    data.push(0xff); // VarInt prefix for 8-byte encoding
    data.extend_from_slice(&[255, 255, 255, 255, 255, 255, 255, 255]); // u64::MAX outputs
    
    // Should fail during deserialization
    let result = deserialize_transaction(&data);
    assert!(result.is_err());
}

#[test]
fn test_transaction_malformed_prevout_hash() {
    // Transaction with valid structure but malformed hash
    // (Hash is just bytes, so any 32 bytes is valid)
    // This test checks that we can handle any hash value
    
    let mut data = vec![0, 0, 0, 0]; // Version
    data.push(1); // 1 input
    data.extend_from_slice(&[0xff; 32]); // Hash (all 0xff)
    data.extend_from_slice(&[0, 0, 0, 0]); // Index
    data.push(0); // Empty script
    data.extend_from_slice(&[0, 0, 0, 0]); // Sequence
    data.push(1); // 1 output
    data.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // Value
    data.push(0); // Empty script
    data.extend_from_slice(&[0, 0, 0, 0]); // Lock time
    
    // Should succeed - any hash value is valid
    assert!(deserialize_transaction(&data).is_ok());
}

#[test]
fn test_transaction_malformed_script_length() {
    // Script length VarInt that's invalid (e.g., value < 0xfd but using 0xfd prefix)
    let mut data = vec![0, 0, 0, 0]; // Version
    data.push(1); // 1 input
    data.extend_from_slice(&[0; 32]); // Hash
    data.extend_from_slice(&[0, 0, 0, 0]); // Index
    data.push(0xfd); // VarInt prefix for 2-byte encoding
    data.push(252); // Value 252 (should use single byte, not 0xfd)
    data.push(0); // Second byte
    
    // Should fail - invalid VarInt encoding
    assert!(deserialize_transaction(&data).is_err());
}

#[test]
fn test_block_header_malformed_timestamp() {
    // Block header with any timestamp value should be accepted by parser
    // (Validation happens later)
    let mut header = vec![0; 80];
    
    // Set timestamp to arbitrary value (bytes 68-71)
    header[68] = 0xff;
    header[69] = 0xff;
    header[70] = 0xff;
    header[71] = 0xff;
    
    // Should succeed - parser doesn't validate timestamp
    assert!(deserialize_block_header(&header).is_ok());
}

#[test]
fn test_transaction_empty_after_version() {
    // Transaction with only version, nothing else
    let data = vec![0, 0, 0, 0]; // Version only
    
    assert!(deserialize_transaction(&data).is_err());
}

#[test]
fn test_transaction_incomplete_output() {
    // Transaction with incomplete output
    let mut data = vec![0, 0, 0, 0]; // Version
    data.push(0); // 0 inputs
    data.push(1); // 1 output
    data.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0]); // Only 7 bytes of value (need 8)
    
    assert!(deserialize_transaction(&data).is_err());
}

#[test]
fn test_transaction_incomplete_lock_time() {
    // Transaction with incomplete lock time
    let mut data = vec![0, 0, 0, 0]; // Version
    data.push(0); // 0 inputs
    data.push(0); // 0 outputs
    data.extend_from_slice(&[0, 0, 0]); // Only 3 bytes of lock time (need 4)
    
    assert!(deserialize_transaction(&data).is_err());
}

