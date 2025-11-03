#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::{Transaction, BlockHeader};
use consensus_proof::transaction::check_transaction;
use consensus_proof::serialization::transaction::{serialize_transaction, deserialize_transaction};
use consensus_proof::serialization::block::{serialize_block_header, deserialize_block_header};
use consensus_proof::serialization::varint::{encode_varint, decode_varint};
use consensus_proof::segwit::calculate_transaction_weight;

/// Differential fuzzing target for internal consistency testing
/// 
/// Tests that different code paths and functions produce consistent results:
/// - Serialization round-trips preserve transaction properties
/// - Validation results are consistent before/after serialization
/// - Weight calculations are idempotent
/// - Economic calculations are consistent
/// 
/// This does NOT call Bitcoin Core - it tests consensus-proof internally.
fuzz_target!(|data: &[u8]| {
    // Test 1: Transaction serialization round-trip preserves validation
    if data.len() >= 4 {
        if let Ok(tx1) = deserialize_transaction(data) {
            // Validate original
            let validation1 = check_transaction(&tx1);
            
            // Serialize and deserialize
            let serialized = serialize_transaction(&tx1);
            if let Ok(tx2) = deserialize_transaction(&serialized) {
                // Validate after round-trip
                let validation2 = check_transaction(&tx2);
                
                // Differential check: Validation should be consistent
                // Both should be valid or both invalid (may differ in error messages)
                match (&validation1, &validation2) {
                    (Ok(v1), Ok(v2)) => {
                        // Both validations succeeded - check if results match
                        use consensus_proof::ValidationResult;
                        match (v1, v2) {
                            (ValidationResult::Valid, ValidationResult::Valid) => {
                                // Both valid - good
                            }
                            (ValidationResult::Invalid(_), ValidationResult::Invalid(_)) => {
                                // Both invalid - acceptable (errors may differ)
                            }
                            _ => {
                                // Mismatch - this is a bug!
                                panic!("Validation mismatch after round-trip: {:?} vs {:?}", v1, v2);
                            }
                        }
                    }
                    _ => {
                        // One succeeded, one failed - this is a bug!
                        panic!("Validation error inconsistency: {:?} vs {:?}", validation1, validation2);
                    }
                }
                
                // Additional check: Transaction properties should match
                assert_eq!(tx1.version, tx2.version, "Version must match after round-trip");
                assert_eq!(tx1.inputs.len(), tx2.inputs.len(), "Input count must match");
                assert_eq!(tx1.outputs.len(), tx2.outputs.len(), "Output count must match");
                assert_eq!(tx1.lock_time, tx2.lock_time, "Lock time must match");
                
                // Test 1b: Weight calculation should be idempotent
                // Note: Weight calculation requires witness data, skip if not available
                let weight1 = calculate_transaction_weight(&tx1, None).unwrap_or(0);
                let weight2 = calculate_transaction_weight(&tx2, None).unwrap_or(0);
                assert_eq!(weight1, weight2, "Transaction weight must be consistent");
            }
        }
    }
    
    // Test 2: Block header serialization round-trip preserves properties
    if data.len() >= 80 {
        if let Ok(header1) = deserialize_block_header(data) {
            // Serialize and deserialize
            let serialized = serialize_block_header(&header1);
            if let Ok(header2) = deserialize_block_header(&serialized) {
                // All properties should match
                assert_eq!(header1.version, header2.version, "Header version must match");
                assert_eq!(header1.prev_block_hash, header2.prev_block_hash, "Prev hash must match");
                assert_eq!(header1.merkle_root, header2.merkle_root, "Merkle root must match");
                assert_eq!(header1.timestamp, header2.timestamp, "Timestamp must match");
                assert_eq!(header1.bits, header2.bits, "Bits must match");
                assert_eq!(header1.nonce, header2.nonce, "Nonce must match");
                
                // Test 2b: Header properties should be consistent
                // (Economic calculations tested in economic_validation target)
            }
        }
    }
    
    // Test 3: VarInt encoding/decoding round-trip consistency
    if data.len() >= 1 {
        if let Ok((value1, consumed1)) = decode_varint(data) {
            // Encode and decode again
            let encoded = encode_varint(value1);
            if let Ok((value2, consumed2)) = decode_varint(&encoded) {
                // Values must match
                assert_eq!(value1, value2, "VarInt value must match after round-trip");
                
                // Additional check: Multiple encodings should produce same result
                let encoded2 = encode_varint(value1);
                assert_eq!(encoded, encoded2, "VarInt encoding must be deterministic");
            }
        }
    }
    
    // Test 4: Serialization determinism
    // Same transaction should serialize to same bytes
    if data.len() >= 4 {
        if let Ok(tx1) = deserialize_transaction(data) {
            if let Ok(tx2) = deserialize_transaction(data) {
                // Both should be identical
                assert_eq!(tx1.version, tx2.version);
                assert_eq!(tx1.inputs.len(), tx2.inputs.len());
                assert_eq!(tx1.outputs.len(), tx2.outputs.len());
                
                // Serialization must be deterministic
                let serialized1 = serialize_transaction(&tx1);
                let serialized2 = serialize_transaction(&tx2);
                assert_eq!(serialized1, serialized2, "Serialization must be deterministic");
                
                // Weight calculation must be deterministic
                let weight1 = calculate_transaction_weight(&tx1, None).unwrap_or(0);
                let weight2 = calculate_transaction_weight(&tx2, None).unwrap_or(0);
                assert_eq!(weight1, weight2, "Weight calculation must be deterministic");
            }
        }
    }
    
    // Test 5: Validation consistency
    // Transaction should have same validation result when parsed multiple times
    if data.len() >= 4 {
        if let Ok(tx1) = deserialize_transaction(data) {
            if let Ok(tx2) = deserialize_transaction(data) {
                let result1 = check_transaction(&tx1);
                let result2 = check_transaction(&tx2);
                
                // Results should match
                match (&result1, &result2) {
                    (Ok(v1), Ok(v2)) => {
                        use consensus_proof::ValidationResult;
                        match (v1, v2) {
                            (ValidationResult::Valid, ValidationResult::Valid) => {
                                // Both valid
                            }
                            (ValidationResult::Invalid(_), ValidationResult::Invalid(_)) => {
                                // Both invalid - acceptable
                            }
                            _ => {
                                panic!("Validation results must match: {:?} vs {:?}", v1, v2);
                            }
                        }
                    }
                    _ => {
                        panic!("Validation error consistency failed: {:?} vs {:?}", result1, result2);
                    }
                }
            }
        }
    }
    
    // Test 6: Edge case handling
    // Functions should handle edge cases without panicking
    if data.is_empty() {
        // Empty data should fail gracefully
        let _ = deserialize_transaction(data);
        let _ = deserialize_block_header(data);
        let _ = decode_varint(data);
    }
    
    // Test 7: Invalid input handling
    // All functions should handle invalid inputs gracefully (no panics)
    if data.len() > 0 && data.len() < 4 {
        let _ = deserialize_transaction(data);
    }
    
    if data.len() > 0 && data.len() < 80 {
        let _ = deserialize_block_header(data);
    }
});

