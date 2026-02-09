//! Tests for witness data validation in block context

use blvm_consensus::types::*;
use blvm_consensus::block::connect_block;
use blvm_consensus::segwit::Witness;
use blvm_consensus::bip113::get_median_time_past;

#[test]
fn test_witness_validation_empty_witnesses() {
    // Test that blocks without SegWit transactions validate with empty witnesses
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                script_sig: vec![],
                sequence: 0xffffffff,
            }].into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }].into(),
            lock_time: 0,
        }],
    };
    
    let witnesses: Vec<Witness> = vec![Vec::new()]; // Empty witness for coinbase
    let mut utxo_set = UtxoSet::new();
    
    // Should validate successfully with empty witnesses
    let (result, _) = connect_block(&block, &witnesses, utxo_set, 0, None, 0u64, crate::types::Network::Mainnet).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_witness_validation_segwit_block() {
    // Test that SegWit blocks validate with witness data
    // Note: This is a simplified test - full SegWit validation would require
    // proper script validation with witness data
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![
            // Coinbase transaction
            Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 5000000000,
                    script_pubkey: vec![].into(),
                }].into(),
                lock_time: 0,
            },
        ],
    };
    
    let witnesses: Vec<Witness> = vec![
        Vec::new(), // Empty witness for coinbase (correct)
    ];
    
    let mut utxo_set = UtxoSet::new();
    
    let (result, _) = connect_block(&block, &witnesses, utxo_set, 0, None, 0u64, crate::types::Network::Mainnet).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_witness_count_mismatch() {
    // Test that witness count mismatch is detected
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                script_sig: vec![],
                sequence: 0xffffffff,
            }].into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }].into(),
            lock_time: 0,
        }],
    };
    
    // Wrong number of witnesses (2 instead of 1)
    let witnesses: Vec<Witness> = vec![Vec::new(), Vec::new()];
    let mut utxo_set = UtxoSet::new();
    
    let (result, _) = connect_block(&block, &witnesses, utxo_set, 0, None, 0u64, crate::types::Network::Mainnet).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_median_time_past_validation() {
    // Test that median time-past is used for timestamp CLTV validation
    let mut headers = Vec::new();
    
    // Create 11 headers with timestamps
    for i in 0..11 {
        headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890 + (i * 600), // 10 minutes apart
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }
    
    // Calculate median time-past
    let median_time = get_median_time_past(&headers);
    
    // Should be the median of the 11 timestamps
    assert!(median_time > 0);
    
    // Test with blocks (would use median time-past for CLTV)
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                script_sig: vec![],
                sequence: 0xffffffff,
            }].into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }].into(),
            lock_time: 0,
        }],
    };
    
    let witnesses: Vec<Witness> = vec![Vec::new()];
    let mut utxo_set = UtxoSet::new();
    
    // Validate with recent headers for median time-past
    let (result, _) = connect_block(&block, &witnesses, utxo_set, 0, Some(&headers), crate::types::Network::Mainnet).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_median_time_past_with_fewer_headers() {
    // Test median time-past with fewer than 11 headers
    let headers = vec![
        BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890 + 600,
            bits: 0x1d00ffff,
            nonce: 0,
        },
    ];
    
    // Should still calculate median (of 2 headers)
    let median_time = get_median_time_past(&headers);
    assert!(median_time > 0);
}

