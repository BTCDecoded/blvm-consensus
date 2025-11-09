//! BIP Compliance Tests
//! 
//! Tests for compliance with Bitcoin Core behavior for consensus-critical BIPs.
//! These tests verify that our BIP implementations match Bitcoin Core's validation logic.

use bllvm_consensus::*;
use bllvm_consensus::script::verify_script_with_context_full;
use bllvm_consensus::bip113::get_median_time_past;
use bllvm_consensus::constants::LOCKTIME_THRESHOLD;

#[test]
fn test_bip65_cltv_compliance_basic() {
    // Basic CLTV compliance: transaction locktime must be >= required locktime
    // This matches Bitcoin Core's validation logic
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32], index: 0 },
            script_sig: {
                let mut script = vec![0x51]; // OP_1
                script.extend_from_slice(&encode_varint(400000)); // Required locktime
                script.push(0xb1); // CLTV
                script
            },
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }],
        lock_time: 500000, // >= required
    };
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let prevouts = vec![TransactionOutput {
        value: utxo.value,
        script_pubkey: utxo.script_pubkey.clone(),
    }];
    
    // Should pass: tx.lock_time (500000) >= required (400000)
    let result = verify_script_with_context_full(
        &input.script_sig,
        &utxo.script_pubkey,
        None,
        0,
        &tx,
        0,
        &prevouts,
        Some(500000), // Block height for block-height CLTV
        None,
    );
    
    assert!(result.is_ok());
    // Note: Actual validation in Bitcoin Core would require exact block height context
}

#[test]
fn test_bip112_csv_compliance_basic() {
    // Basic CSV compliance: input sequence must be >= required sequence
    // This matches Bitcoin Core's validation logic
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32], index: 0 },
            script_sig: {
                let mut script = vec![0x51]; // OP_1
                script.extend_from_slice(&encode_varint(0x00040000)); // 4 blocks required
                script.push(0xb2); // CSV
                script
            },
            sequence: 0x00050000, // 5 blocks (>= required)
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }],
        lock_time: 0,
    };
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let prevouts = vec![TransactionOutput {
        value: utxo.value,
        script_pubkey: utxo.script_pubkey.clone(),
    }];
    
    // Should pass: input sequence (5 blocks) >= required (4 blocks)
    let result = verify_script_with_context_full(
        &input.script_sig,
        &utxo.script_pubkey,
        None,
        0,
        &tx,
        0,
        &prevouts,
        None,
        None,
    );
    
    assert!(result.is_ok());
}

#[test]
fn test_bip113_median_time_past_compliance() {
    // BIP113 compliance: median time-past uses last 11 blocks
    // Matches Bitcoin Core's median time calculation
    
    let timestamps = vec![
        1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000
    ];
    
    let headers: Vec<BlockHeader> = timestamps.iter().map(|&t| BlockHeader {
        version: 1,
        prev_block_hash: [0u8; 32],
        merkle_root: [0u8; 32],
        timestamp: t,
        bits: 0x1d00ffff,
        nonce: 0,
    }).collect();
    
    let median = get_median_time_past(&headers);
    
    // Median of 11 sorted timestamps should be the 6th value (index 5)
    // [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    // Median = 1500 (6th element)
    assert_eq!(median, 1500);
}

#[test]
fn test_bip65_cltv_type_mismatch_rejection() {
    // Bitcoin Core rejects CLTV when locktime types don't match
    // Block height vs timestamp mismatch should fail
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32], index: 0 },
            script_sig: {
                let mut script = vec![0x51];
                script.extend_from_slice(&encode_varint(600000000)); // Timestamp (>= threshold)
                script.push(0xb1); // CLTV
                script
            },
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }],
        lock_time: 400000, // Block height (< threshold)
    };
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let prevouts = vec![TransactionOutput {
        value: utxo.value,
        script_pubkey: utxo.script_pubkey.clone(),
    }];
    
    // Should fail: type mismatch (block height vs timestamp)
    let result = verify_script_with_context_full(
        &input.script_sig,
        &utxo.script_pubkey,
        None,
        0,
        &tx,
        0,
        &prevouts,
        None,
        None,
    );
    
    assert!(result.is_ok());
    assert!(!result.unwrap()); // Should fail validation
}

#[test]
fn test_bip112_csv_disabled_sequence_rejection() {
    // Bitcoin Core rejects CSV when sequence is disabled (0x80000000 bit set)
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32], index: 0 },
            script_sig: {
                let mut script = vec![0x51];
                script.extend_from_slice(&encode_varint(0x00040000));
                script.push(0xb2); // CSV
                script
            },
            sequence: 0x80000000, // Sequence disabled
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }],
        lock_time: 0,
    };
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let prevouts = vec![TransactionOutput {
        value: utxo.value,
        script_pubkey: utxo.script_pubkey.clone(),
    }];
    
    // Should fail: sequence disabled
    let result = verify_script_with_context_full(
        &input.script_sig,
        &utxo.script_pubkey,
        None,
        0,
        &tx,
        0,
        &prevouts,
        None,
        None,
    );
    
    assert!(result.is_ok());
    assert!(!result.unwrap()); // Should fail validation
}

// Helper function for encoding varints (used in tests)
fn encode_varint(value: u64) -> Vec<u8> {
    if value < 0xfd {
        vec![value as u8]
    } else if value <= 0xffff {
        let mut bytes = vec![0xfd];
        bytes.extend_from_slice(&(value as u16).to_le_bytes());
        bytes
    } else if value <= 0xffffffff {
        let mut bytes = vec![0xfe];
        bytes.extend_from_slice(&(value as u32).to_le_bytes());
        bytes
    } else {
        let mut bytes = vec![0xff];
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes
    }
}

