//! Tests for block deserialization offset tracking
//!
//! This module tests that we correctly track bytes consumed when deserializing
//! transactions from blocks, ensuring we don't skip or misread transaction data.
//!
//! Critical bug fixed: Previously we re-serialized transactions to calculate size,
//! which could use cached/wrong data. Now we track actual bytes consumed during deserialization.

use blvm_consensus::serialization::block::deserialize_block_with_witnesses;
use blvm_consensus::serialization::transaction::{deserialize_transaction, serialize_transaction};
use blvm_consensus::block::calculate_tx_id;

#[test]
fn test_deserialize_transaction_returns_correct_offset() {
    // Create a simple transaction
    use blvm_consensus::types::*;
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            script_sig: vec![0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x04], // 7 bytes
            sequence: 0xffffffff,
        }].into_boxed_slice(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000, // 50 BTC
            script_pubkey: vec![0x41, 0x04, 0x67, 0x8a, 0xfd, 0xb0, 0xfe, 0x55, 0x48, 0x27, 0x19, 0x67, 0xf1, 0xa6, 0x71, 0x30, 0xb7, 0x10, 0x5c, 0xd6, 0xa8, 0x28, 0xe0, 0x39, 0x09, 0xa6, 0x79, 0x62, 0xe0, 0xea, 0x1f, 0x61, 0xde, 0xb6, 0x49, 0xf6, 0xbc, 0x3f, 0x4c, 0xef, 0x38, 0xc4, 0xf3, 0x55, 0x04, 0xe5, 0x1e, 0xc1, 0x12, 0xde, 0x5c, 0x38, 0x4d, 0xf7, 0xba, 0x0b, 0x8d, 0x57, 0x8a, 0x4c, 0x70, 0x2b, 0x6b, 0xf1, 0x1d, 0x5f, 0xac], // 65 bytes
        }].into_boxed_slice(),
        lock_time: 0,
    };
    
    // Serialize it
    let serialized = serialize_transaction(&tx);
    let original_len = serialized.len();
    
    // Deserialize with offset tracking
    use blvm_consensus::serialization::transaction::deserialize_transaction_with_offset;
    let (deserialized, bytes_consumed) = deserialize_transaction_with_offset(&serialized).unwrap();
    
    // Verify bytes consumed matches serialized length
    assert_eq!(bytes_consumed, original_len, 
               "Bytes consumed ({}) must match serialized length ({})", 
               bytes_consumed, original_len);
    
    // Verify transaction is correct
    assert_eq!(deserialized.version, tx.version);
    assert_eq!(deserialized.inputs.len(), tx.inputs.len());
    assert_eq!(deserialized.outputs.len(), tx.outputs.len());
    
    // Verify txid matches
    let original_txid = calculate_tx_id(&tx);
    let deserialized_txid = calculate_tx_id(&deserialized);
    assert_eq!(original_txid, deserialized_txid, "TxID must match after round-trip");
}

#[test]
fn test_block_deserialization_tracks_offset_correctly() {
    // Test that when deserializing a block with multiple transactions,
    // we correctly track the offset and don't skip or misread data
    
    // Create a block with 2 transactions
    use blvm_consensus::types::*;
    
    let tx1 = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            script_sig: vec![0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x04], // 7 bytes
            sequence: 0xffffffff,
        }].into_boxed_slice(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000,
            script_pubkey: vec![0x41; 65], // 65 bytes
        }].into_boxed_slice(),
        lock_time: 0,
    };
    
    let tx2 = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1u8; 32], // Different hash
                index: 0,
            },
            script_sig: vec![0x07, 0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x04, 0xff], // 9 bytes (different length)
            sequence: 0xffffffff,
        }].into_boxed_slice(),
        outputs: vec![TransactionOutput {
            value: 25_0000_0000,
            script_pubkey: vec![0x41; 33], // 33 bytes (different length)
        }].into_boxed_slice(),
        lock_time: 0,
    };
    
    // Serialize transactions
    let tx1_bytes = serialize_transaction(&tx1);
    let tx2_bytes = serialize_transaction(&tx2);
    
    // Create a minimal block (just header + transactions for testing)
    use blvm_consensus::serialization::varint::encode_varint;
    let mut block_data = Vec::new();
    
    // Block header (80 bytes of zeros for simplicity)
    block_data.extend_from_slice(&[0u8; 80]);
    
    // Transaction count (2)
    block_data.extend_from_slice(&encode_varint(2));
    
    // Transaction 1
    block_data.extend_from_slice(&tx1_bytes);
    
    // Transaction 2
    block_data.extend_from_slice(&tx2_bytes);
    
    // Deserialize block
    let (block, _witnesses) = deserialize_block_with_witnesses(&block_data).unwrap();
    
    // Verify we got 2 transactions
    assert_eq!(block.transactions.len(), 2, "Block must have 2 transactions");
    
    // Verify transaction 1 matches
    let tx1_txid = calculate_tx_id(&tx1);
    let block_tx1_txid = calculate_tx_id(&block.transactions[0]);
    assert_eq!(tx1_txid, block_tx1_txid, "First transaction txid must match");
    
    // Verify transaction 2 matches
    let tx2_txid = calculate_tx_id(&tx2);
    let block_tx2_txid = calculate_tx_id(&block.transactions[1]);
    assert_eq!(tx2_txid, block_tx2_txid, "Second transaction txid must match");
    
    // Verify transactions are different
    assert_ne!(block_tx1_txid, block_tx2_txid, "Transactions must have different txids");
}

#[test]
fn test_sequential_block_processing_maintains_correct_txids() {
    // This test specifically targets the bug where processing blocks sequentially
    // from height 0 would cause later blocks to have incorrect txids due to offset tracking issues
    
    use blvm_consensus::types::*;
    
    // Create 3 different transactions with different script_sig lengths
    let transactions: Vec<Transaction> = (0..3).map(|i| {
        Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32], // Different hash for each
                    index: 0,
                },
                script_sig: vec![0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x04 + i], // Different script
                sequence: 0xffffffff,
            }].into_boxed_slice(),
            outputs: vec![TransactionOutput {
                value: (50 - i as i64) * 1000_0000,
                script_pubkey: vec![0x41; 65],
            }].into_boxed_slice(),
            lock_time: 0,
        }
    }).collect();
    
    // Serialize all transactions
    let tx_bytes: Vec<Vec<u8>> = transactions.iter()
        .map(|tx| serialize_transaction(tx))
        .collect();
    
    // Create blocks, each with one transaction
    let mut blocks = Vec::new();
    for tx_bytes in &tx_bytes {
        use blvm_consensus::serialization::varint::encode_varint;
        let mut block_data = Vec::new();
        block_data.extend_from_slice(&[0u8; 80]); // Header
        block_data.extend_from_slice(&encode_varint(1)); // 1 transaction
        block_data.extend_from_slice(tx_bytes);
        blocks.push(block_data);
    }
    
    // Deserialize all blocks sequentially (simulating the bug scenario)
    let mut deserialized_txids = Vec::new();
    for block_data in &blocks {
        let (block, _) = deserialize_block_with_witnesses(block_data).unwrap();
        let txid = calculate_tx_id(&block.transactions[0]);
        deserialized_txids.push(txid);
    }
    
    // Verify all txids are different
    for i in 0..deserialized_txids.len() {
        for j in (i+1)..deserialized_txids.len() {
            assert_ne!(deserialized_txids[i], deserialized_txids[j],
                      "Transaction {} and {} must have different txids", i, j);
        }
    }
    
    // Verify txids match original transactions
    for (i, (original_tx, deserialized_txid)) in transactions.iter().zip(deserialized_txids.iter()).enumerate() {
        let original_txid = calculate_tx_id(original_tx);
        assert_eq!(original_txid, *deserialized_txid,
                  "Block {} transaction txid must match original", i);
    }
}

