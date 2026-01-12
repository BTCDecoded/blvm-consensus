//! Test for block deserialization cache collision bug
//!
//! This test verifies that when processing blocks sequentially, we don't
//! get cache collisions that cause later blocks to have incorrect transaction data.

use blvm_consensus::serialization::block::deserialize_block_with_witnesses;
use blvm_consensus::block::calculate_tx_id;

#[test]
fn test_sequential_blocks_maintain_unique_txids() {
    // Create two blocks with different transactions but same structure
    use blvm_consensus::types::*;
    use blvm_consensus::serialization::transaction::serialize_transaction;
    use blvm_consensus::serialization::varint::encode_varint;
    
    // Block 1: Transaction with script_sig "04ffff001d010b"
    let tx1 = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            script_sig: vec![0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x0b], // 7 bytes
            sequence: 0xffffffff,
        }].into_boxed_slice(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000,
            script_pubkey: vec![0x43, 0x41, 0x04, 0x72, 0x11, 0xa8, 0x24, 0xf5, 0x5b, 0x50, 0x52, 0x28, 0xe4, 0xc3, 0xd5, 0x19, 0x4c, 0x1f, 0xcf, 0xaa, 0x15, 0xa4, 0x56, 0xab, 0xdf, 0x37, 0xf9, 0xb9, 0xd9, 0x7a, 0x40, 0x40, 0xaf, 0xc0, 0x73, 0xde, 0xe6, 0xc8], // Different output
        }].into_boxed_slice(),
        lock_time: 0,
    };
    
    // Block 2: Transaction with same script_sig but DIFFERENT output
    let tx2 = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            script_sig: vec![0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x0b], // Same script_sig!
            sequence: 0xffffffff,
        }].into_boxed_slice(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000,
            script_pubkey: vec![0x43, 0x41, 0x04, 0x97, 0x73, 0x67, 0x16, 0x4c, 0xa2, 0x4f, 0x1f, 0x2d, 0xe2, 0xe2, 0xcf, 0xb9, 0xe5, 0xc3, 0xf2, 0x2d, 0x51, 0x0d, 0x3f, 0x33, 0x68, 0x3d, 0xe2, 0x00, 0x28, 0x31, 0x00, 0xaf, 0x0c, 0x86, 0x67, 0xdb, 0xa7, 0xe4], // Different output!
        }].into_boxed_slice(),
        lock_time: 0,
    };
    
    // Calculate txids
    let tx1_id = calculate_tx_id(&tx1);
    let tx2_id = calculate_tx_id(&tx2);
    
    // They MUST be different (different outputs)
    assert_ne!(tx1_id, tx2_id, "Transactions with different outputs must have different txids");
    
    // Serialize transactions
    let tx1_bytes = serialize_transaction(&tx1);
    let tx2_bytes = serialize_transaction(&tx2);
    
    // Create blocks
    let mut block1_data = Vec::new();
    block1_data.extend_from_slice(&[0u8; 80]); // Header
    block1_data.extend_from_slice(&encode_varint(1)); // 1 transaction
    block1_data.extend_from_slice(&tx1_bytes);
    
    let mut block2_data = Vec::new();
    block2_data.extend_from_slice(&[0u8; 80]); // Header
    block2_data.extend_from_slice(&encode_varint(1)); // 1 transaction
    block2_data.extend_from_slice(&tx2_bytes);
    
    // Deserialize blocks sequentially (simulating the bug scenario)
    let (block1, _) = deserialize_block_with_witnesses(&block1_data).unwrap();
    let (block2, _) = deserialize_block_with_witnesses(&block2_data).unwrap();
    
    // Verify txids match
    let block1_txid = calculate_tx_id(&block1.transactions[0]);
    let block2_txid = calculate_tx_id(&block2.transactions[0]);
    
    assert_eq!(tx1_id, block1_txid, "Block 1 txid must match original");
    assert_eq!(tx2_id, block2_txid, "Block 2 txid must match original");
    assert_ne!(block1_txid, block2_txid, "Blocks must have different txids");
}

