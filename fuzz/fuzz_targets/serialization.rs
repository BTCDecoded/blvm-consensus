#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::serialization::transaction::{serialize_transaction, deserialize_transaction};
use consensus_proof::serialization::block::{serialize_block_header, deserialize_block_header};
use consensus_proof::serialization::varint::{encode_varint, decode_varint};

fuzz_target!(|data: &[u8]| {
    // Fuzz serialization/deserialization: round-trip correctness, edge cases, invalid inputs
    
    // Test 1: Transaction serialization round-trip
    if data.len() >= 4 {
        // Try to parse as transaction
        if let Ok(tx) = deserialize_transaction(data) {
            // Serialize and deserialize again - should be identical
            let serialized = serialize_transaction(&tx);
            if let Ok(deserialized) = deserialize_transaction(&serialized) {
                // Round-trip property: deserialize(serialize(tx)) = tx
                assert_eq!(deserialized.version, tx.version, "Version must match after round-trip");
                assert_eq!(deserialized.inputs.len(), tx.inputs.len(), "Input count must match");
                assert_eq!(deserialized.outputs.len(), tx.outputs.len(), "Output count must match");
                assert_eq!(deserialized.lock_time, tx.lock_time, "Lock time must match");
            }
        }
    }
    
    // Test 2: Block header serialization round-trip
    if data.len() >= 80 {
        // Try to parse as block header
        if let Ok(header) = deserialize_block_header(data) {
            // Serialize and deserialize again - should be identical
            let serialized = serialize_block_header(&header);
            if let Ok(deserialized) = deserialize_block_header(&serialized) {
                // Round-trip property: deserialize(serialize(header)) = header
                assert_eq!(deserialized.version, header.version, "Version must match");
                assert_eq!(deserialized.prev_block_hash, header.prev_block_hash, "Prev hash must match");
                assert_eq!(deserialized.merkle_root, header.merkle_root, "Merkle root must match");
                assert_eq!(deserialized.timestamp, header.timestamp, "Timestamp must match");
                assert_eq!(deserialized.bits, header.bits, "Bits must match");
                assert_eq!(deserialized.nonce, header.nonce, "Nonce must match");
            }
        }
    }
    
    // Test 3: VarInt encoding/decoding round-trip
    if data.len() >= 1 {
        // Try to decode VarInt
        if let Ok((value, _consumed)) = decode_varint(data) {
            // Encode and decode again - should be identical
            let encoded = encode_varint(value);
            if let Ok((decoded_value, _)) = decode_varint(&encoded) {
                // Round-trip property: decode(encode(value)) = value
                assert_eq!(decoded_value, value, "VarInt value must match after round-trip");
            }
        }
    }
    
    // Test 4: Serialization determinism
    // Same input should produce same output
    if data.len() >= 4 {
        if let Ok(tx1) = deserialize_transaction(data) {
            if let Ok(tx2) = deserialize_transaction(data) {
                // Same deserialization should produce identical transactions
                assert_eq!(tx1.version, tx2.version);
                assert_eq!(tx1.inputs.len(), tx2.inputs.len());
                assert_eq!(tx1.outputs.len(), tx2.outputs.len());
                assert_eq!(tx1.lock_time, tx2.lock_time);
                
                // Serialization should be deterministic
                let serialized1 = serialize_transaction(&tx1);
                let serialized2 = serialize_transaction(&tx2);
                assert_eq!(serialized1, serialized2, "Serialization must be deterministic");
            }
        }
    }
    
    // Test 5: Invalid input handling
    // Should handle malformed inputs gracefully (no panics)
    if data.len() < 4 {
        // Too short - should fail gracefully
        let _result = deserialize_transaction(data);
        // Should return error, not panic
    }
    
    if data.len() < 80 {
        // Too short for block header - should fail gracefully
        let _result = deserialize_block_header(data);
        // Should return error, not panic
    }
    
    // Test 6: Edge cases for VarInt
    // Test various value ranges
    let test_values = [
        0u64,
        1,
        127,
        128,
        16383,
        16384,
        2097151,
        2097152,
        268435455,
        268435456,
        u64::MAX,
    ];
    
    for &value in &test_values {
        let encoded = encode_varint(value);
        if let Ok((decoded, _)) = decode_varint(&encoded) {
            assert_eq!(decoded, value, "VarInt must preserve value");
        }
    }
});

